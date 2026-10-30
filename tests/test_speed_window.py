import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import run_speed_grid_queue_round34 as speed_queue  # noqa: E402
import speed_window  # noqa: E402
from speed_window import SpeedWindow, SpeedWindowConflict  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_control(tmp_path, monkeypatch):
    monkeypatch.setenv(speed_window.ENV, str(tmp_path / "control"))


def test_single_owner_state_and_stale_file_cannot_deadlock():
    assert not speed_window.speed_window_requested()
    with SpeedWindow() as window:
        assert speed_window.speed_window_requested()
        with pytest.raises(SpeedWindowConflict):
            SpeedWindow().__enter__()
        state = json.loads(speed_window.paths()[1].read_text())
        assert state["phase"] == "waiting" and state["owner_pid"] == os.getpid()
        window.set_phase("running")
        assert json.loads(speed_window.paths()[1].read_text())["phase"] == "running"
    assert not speed_window.speed_window_requested()
    speed_window.paths()[1].write_text("stale corrupt state")
    assert not speed_window.speed_window_requested()


def test_waiter_resumes_after_release(monkeypatch):
    states = iter((True, True, False))
    sleeps = []
    monkeypatch.setattr(speed_window, "speed_window_requested", lambda: next(states))
    speed_window.wait_for_speed_window("synthetic", 5, sleep=sleeps.append)
    assert sleeps == [5, 5]


def run_fake_speed_queue(tmp_path, monkeypatch, runner_body):
    runner = tmp_path / "fake_speed_runner.py"
    runner.write_text(
        f"import sys\nsys.path.insert(0, {str(_ROOT / 'tools')!r})\n"
        "import speed_window\n"
        "assert speed_window.speed_window_requested(), 'runner lost inherited gate'\n"
        + runner_body + "\n"
    )
    monkeypatch.setattr(speed_queue, "RUNNER", str(runner))
    monkeypatch.setattr(speed_queue, "gpu_processes", lambda: set())
    monkeypatch.setattr(sys, "argv", ["speed-queue", "--interval", "1"])
    return speed_queue.main()


def test_success_and_failure_always_release_gate(tmp_path, monkeypatch, capsys):
    assert run_fake_speed_queue(tmp_path, monkeypatch, "print('SYNTHETIC_RUNNER_OK')") == 0
    assert "SPEED_QUEUE_COMPLETE" in capsys.readouterr().out
    assert not speed_window.speed_window_requested()
    assert run_fake_speed_queue(tmp_path, monkeypatch, "raise SystemExit(7)") == 7
    assert "SPEED_RUNNER_STOP rc=7" in capsys.readouterr().out
    assert not speed_window.speed_window_requested()


def test_exclusivity_race_retries_without_second_owner(tmp_path, monkeypatch):
    count = tmp_path / "calls"
    body = (
        "from pathlib import Path\n"
        f"p = Path({str(count)!r})\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "if n == 0:\n"
        "    print('exclusive RTX 4090 required')\n"
        "    raise SystemExit(1)"
    )
    assert run_fake_speed_queue(tmp_path, monkeypatch, body) == 0
    assert count.read_text() == "2"
    assert not speed_window.speed_window_requested()


def test_duplicate_queue_is_refused_before_gpu_work(monkeypatch):
    with SpeedWindow():
        monkeypatch.setattr(sys, "argv", ["speed-queue"])
        monkeypatch.setattr(speed_queue, "gpu_processes", lambda: (_ for _ in ()).throw(AssertionError("GPU queried")))
        assert speed_queue.main() == 2


def test_crashed_queue_cannot_release_live_runner_gate(tmp_path):
    code = textwrap.dedent(f"""\
        import os, subprocess, sys
        sys.path.insert(0, {str(_ROOT / 'tools')!r})
        from speed_window import SpeedWindow
        window = SpeedWindow()
        window.__enter__()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'], pass_fds=(window.fd,))
        print('READY', flush=True)
        os._exit(9)
    """)
    env = os.environ.copy()
    parent = subprocess.Popen([sys.executable, "-c", code], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert parent.stdout.readline().startswith("SPEED_WINDOW_REQUESTED")
    assert parent.stdout.readline().strip() == "READY"
    assert parent.wait(timeout=5) == 9
    assert speed_window.speed_window_requested(), "live runner did not retain the gate"
    deadline = time.monotonic() + 5
    while speed_window.speed_window_requested() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not speed_window.speed_window_requested(), "runner exit did not release the gate"


def test_every_gpu_dispatcher_gates_before_work(tmp_path, monkeypatch):
    class Blocked(Exception):
        pass

    def blocked(*args):
        raise Blocked

    def no_work(*args, **kwargs):
        raise AssertionError("work was attempted before the speed-window gate")

    import run_nb_secstr_queue_round34 as nb
    monkeypatch.setattr(nb, "next_action", lambda: ("train", ["no-work"], tmp_path / "nb.log", 0))
    monkeypatch.setattr(nb, "free_mib", no_work)
    monkeypatch.setattr(nb, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["nb-queue", "--interval", "1"])
    with pytest.raises(Blocked):
        nb.main()

    import run_archiveii_family_queue_round34 as archive
    monkeypatch.setattr(archive, "FAMILIES", ("tRNA",))
    monkeypatch.setattr(archive, "EXISTING", set())
    monkeypatch.setattr(archive, "result_status", lambda family: "missing")
    monkeypatch.setattr(archive, "queue_qa_status", lambda family: "pending")
    monkeypatch.setattr(archive, "external_process", no_work)
    monkeypatch.setattr(archive, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["archive-queue", "--interval", "1"])
    with pytest.raises(Blocked):
        archive.main()

    import run_te_el_cell_queue_round34 as te_el
    monkeypatch.setattr(te_el, "CELLS", ("Muscle",))
    monkeypatch.setattr(te_el, "smoke_qa_status", lambda task, cell: "passed")
    monkeypatch.setattr(te_el, "result_status", lambda task, cell, fold: "missing")
    monkeypatch.setattr(te_el, "full_qa_status", lambda task, cell, fold: "pending")
    monkeypatch.setattr(te_el, "external_process", no_work)
    monkeypatch.setattr(te_el, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["te-el-queue", "--interval", "1"])
    with pytest.raises(Blocked):
        te_el.main()

    import run_hek_fold_queue_round34 as hek
    monkeypatch.setattr(hek, "EXTERNAL_FOLD2_PIDS", {})
    monkeypatch.setattr(hek, "fold_state", lambda *args, **kwargs: "ready_to_launch")
    monkeypatch.setattr(hek, "free_mib", no_work)
    monkeypatch.setattr(hek, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["hek-queue", "--interval", "1"])
    with pytest.raises(Blocked):
        hek.main()

    import run_regression_smoke_queue_round34 as smoke
    monkeypatch.setattr(smoke, "result_status", lambda task, cell: "missing")
    monkeypatch.setattr(smoke, "external_process", lambda task, cell: None)
    monkeypatch.setattr(smoke, "free_memory_mib", no_work)
    monkeypatch.setattr(smoke, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["smoke-queue", "--interval", "1"])
    with pytest.raises(Blocked):
        smoke.main()

    import watch_round34_results as watcher
    result = tmp_path / "result.json"
    result.write_text('{"status":"PASS"}')
    monkeypatch.setattr(watcher, "jobs", lambda: {"synthetic": {"result": result, "command": [], "env": {}}})
    monkeypatch.setattr(watcher, "load_state", lambda: {"jobs": {}})
    monkeypatch.setattr(watcher, "run_qa", no_work)
    monkeypatch.setattr(watcher, "wait_for_speed_window", blocked)
    monkeypatch.setattr(sys, "argv", ["qa-watcher", "--interval", "1"])
    with pytest.raises(Blocked):
        watcher.main()
