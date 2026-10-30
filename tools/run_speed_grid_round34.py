#!/usr/bin/env python3
"""Run the missing RTX 4090 full-TS0 speed grid without repeating verified results."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/root/miniconda3/envs/lucaone/bin/python"
TOOLS = Path("/home/rna_jev/tools")
os.environ["SPEED_4090_ROUND33_DIR"] = os.environ.get(
    "SPEED_4090_ROUND33_DIR", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/speed_4090_round33")
os.environ["SPEED_4090_ROUND34_DIR"] = os.environ.get(
    "SPEED_4090_ROUND34_DIR", "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/speed_4090_round34")
sys.path.insert(0, str(TOOLS))
import make_figure5_round34 as figure  # noqa: E402
from speed_end_to_end_4090 import EXPECTED_F1  # noqa: E402


def gpu_processes() -> set[int]:
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def require_exclusive() -> None:
    others = gpu_processes() - {os.getpid()}
    if others:
        raise RuntimeError(f"exclusive RTX 4090 required; other GPU processes present: {sorted(others)}")


def load_validated(model: str, batch: int) -> dict:
    row = figure.load_result(model, batch)
    if abs(row["f1"] - EXPECTED_F1[model]) > 0.002:
        raise RuntimeError(f"F1 mismatch for {model} B={batch}: {row['f1']} vs {EXPECTED_F1[model]}")
    payload = json.loads(figure.result_path(model, batch).read_text())
    for run in payload["runs"]:
        if not 0 < run["n_timed"] < payload["protocol"]["n_sequences"]:
            raise RuntimeError(f"invalid timed sequence count for {model} B={batch}: {run['n_timed']}")
        if run["timed_wall_s"] <= 0 or run["throughput_seq_per_s"] <= 0:
            raise RuntimeError(f"invalid timing values for {model} B={batch}")
    return row


def prediction_sets(payload: dict) -> dict[str, set[tuple[int, int]]]:
    parsed = {}
    for record in payload["predictions"]:
        parsed[record["name"]] = {tuple(int(value) for value in pair) for pair in record["pairs"]}
    return parsed


def validate_complete_models(rows: dict) -> None:
    """Cross-batch equivalence, not bitwise identity.

    CUDA kernels select batch-shape-dependent reduction orders, so identical
    weights can change the last bits of a logit and flip a borderline pair.
    We require >=99% pair agreement and |delta F1| <= 1e-3 across B=1/2/4 and
    record the exact agreement instead of pretending the predictions are
    bitwise invariant.
    """
    for model in figure.MODELS:
        if all((model, batch) in rows for batch in (1, 2, 4)):
            payloads = [json.loads(figure.result_path(model, batch).read_text()) for batch in (1, 2, 4)]
            sets = [prediction_sets(payload) for payload in payloads]
            names = [set(item) for item in sets]
            if len({frozenset(item) for item in names}) != 1:
                raise RuntimeError(f"cross-batch sequence names differ for {model}")
            reference = sets[0]
            agreements = {}
            for batch, item in zip((2, 4), sets[1:]):
                intersection = 0
                union = 0
                for name in reference:
                    merged = reference[name] | item[name]
                    shared = reference[name] & item[name]
                    intersection += len(shared)
                    union += len(merged)
                agreements[batch] = intersection / union if union else 1.0
            f1_values = [payload["runs"][0]["strict_micro_f1"] for payload in payloads]
            spread = max(f1_values) - min(f1_values)
            if min(agreements.values()) < 0.99 or spread > 1e-3:
                raise RuntimeError(
                    f"cross-batch equivalence failed for {model}: pairwise_vs_B1={agreements} spread={spread:.2e}")
            print(f"SPEED_MODEL_BATCH_EQUIVALENCE_PASS model={model} pairwise_vs_B1={agreements} "
                  f"f1_spread={spread:.2e} f1={f1_values}", flush=True)


def quarantine_result(path: Path) -> Path | None:
    if not path.is_file():
        return None
    directory = path.parent / "quarantine"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{path.stem}.invalid_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}{path.suffix}"
    path.replace(destination)
    print(f"SPEED_RESULT_QUARANTINED path={destination}", flush=True)
    return destination


def terminate_process(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.wait()


def run_speed_job(model: str, batch: int, path: Path) -> None:
    try:
        require_exclusive()
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.parent / "logs" / f"{model}_b{batch}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, str(TOOLS / "speed_end_to_end_4090.py"), "--model", model,
                   "--batch-size", str(batch), "--repeats", "3", "--out", str(path)]
        env = {**os.environ, "PYTHONPATH": "/home/rna_jev/src:/home/rna_jev/tools", "CUDA_VISIBLE_DEVICES": "0"}
        print(f"SPEED_JOB_START model={model} B={batch} log={log}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"START cmd={' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                       cwd="/home/rna_jev", start_new_session=True)
            # nvidia-smi reports host-namespace PIDs, so the runner's own GPU
            # context appears as an unrelated PID inside this container.  We
            # enforce exclusivity at start and after release instead of
            # treating the child's own context as a competitor.
            process.wait()
            handle.write(f"END rc={process.returncode}\n")
        if process.returncode != 0:
            raise RuntimeError(f"speed job failed rc={process.returncode}; no retry: {log}")
        for _ in range(60):
            if not gpu_processes():
                break
            time.sleep(1)
        require_exclusive()
    except Exception:
        quarantine_result(path)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate existing results and list missing jobs without launching GPU work")
    args = parser.parse_args()
    rows = {}
    missing = []
    for model in figure.MODELS:
        for batch in (1, 2, 4):
            path = figure.result_path(model, batch)
            if path.is_file():
                rows[model, batch] = load_validated(model, batch)
                print(f"SKIP_VERIFIED model={model} B={batch} sha256={rows[model, batch]['result_sha256']}", flush=True)
            else:
                missing.append((model, batch, path))
    validate_complete_models(rows)
    print("SPEED_GRID_PLAN", json.dumps({"verified": len(rows), "missing": [f"{model}_b{batch}" for model, batch, _ in missing]}), flush=True)
    if args.dry_run:
        return 0
    if not missing:
        print("SPEED_GRID_ALL_RESULTS_ALREADY_VERIFIED", flush=True)
        return 0
    require_exclusive()
    for model, batch, path in missing:
        try:
            run_speed_job(model, batch, path)
            rows[model, batch] = load_validated(model, batch)
            validate_complete_models(rows)
        except Exception:
            quarantine_result(path)
            raise
    if len(rows) != 18:
        raise RuntimeError(f"speed grid incomplete: {len(rows)}/18")
    validate_complete_models(rows)
    require_exclusive()
    print("SPEED_GRID_ALL_18_RESULTS_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
