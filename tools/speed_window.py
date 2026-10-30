"""Cooperative dispatch gate for one exclusive speed-grid window."""
from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ENV = "RNAJEV_SPEED_WINDOW_DIR"
DEFAULT_DIR = Path("/tmp")


def control_dir() -> Path:
    return Path(os.environ.get(ENV, str(DEFAULT_DIR)))


def paths() -> tuple[Path, Path]:
    root = control_dir()
    return root / "rnajev_speed_window_round34.lock", root / "rnajev_speed_window_round34.json"


class SpeedWindowConflict(RuntimeError):
    pass


class SpeedWindow:
    """An flock owned by the queue and inherited by its runner subprocess."""

    def __init__(self) -> None:
        self.handle = None
        self.owner_pid = os.getpid()
        self.owner_start_ticks = int(Path(f"/proc/{self.owner_pid}/stat").read_text().split()[21])

    @property
    def fd(self) -> int:
        if self.handle is None:
            raise RuntimeError("speed window is not active")
        return self.handle.fileno()

    def _write(self, phase: str) -> None:
        _, state = paths()
        payload = {
            "schema_version": 1, "phase": phase, "owner_pid": self.owner_pid,
            "owner_start_ticks": self.owner_start_ticks,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = state.with_name(f"{state.name}.{self.owner_pid}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
        temporary.replace(state)

    def __enter__(self) -> "SpeedWindow":
        lock, state = paths()
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.handle = lock.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            raise SpeedWindowConflict(f"speed window already owned: {lock}") from None
        try:
            self._write("waiting")
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        print(f"SPEED_WINDOW_REQUESTED owner_pid={self.owner_pid} state={state}", flush=True)
        return self

    def set_phase(self, phase: str) -> None:
        if phase not in {"waiting", "running"}:
            raise ValueError(phase)
        self._write(phase)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is None:
            return
        _, state = paths()
        try:
            try:
                payload = json.loads(state.read_text(encoding="utf-8"))
                if (payload.get("owner_pid"), payload.get("owner_start_ticks")) == (self.owner_pid, self.owner_start_ticks):
                    state.unlink(missing_ok=True)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        finally:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        print(f"SPEED_WINDOW_RELEASED owner_pid={self.owner_pid}", flush=True)


def speed_window_requested() -> bool:
    """Fail open only after the owner lock is gone; stale JSON cannot deadlock queues."""
    lock, _ = paths()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def wait_for_speed_window(queue: str, interval: int, *, sleep=time.sleep) -> None:
    waited = False
    while speed_window_requested():
        waited = True
        print(f"WAIT_SPEED_WINDOW queue={queue} state={paths()[1]}", flush=True)
        sleep(interval)
    if waited:
        print(f"RESUME_AFTER_SPEED_WINDOW queue={queue}", flush=True)
