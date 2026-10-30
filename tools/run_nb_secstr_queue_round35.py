#!/usr/bin/env python3
"""Round 35 NucleicBERT secondary-structure queue: batch-4, 3-way parallel, fail-closed.

Why this queue exists: the round-34 queue trained at batch=1, which is ~0.16 s/step
over 47,963 structures (about 2 h per epoch; 10 paired runs would take days). A batch-4
smoke on this 4090 completes 16 steps in ~2 s with ~2.7 GB peak, so batch=4 keeps both
heads identical while making the 5+5 protocol feasible. No step cap is added; the
convergence contract (validation BCE, patience=5, min_delta=1e-6) is unchanged.
Everything else (data, seeds, QA, aggregation) is the round-34 protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PY = "/root/miniconda3/envs/lucaone/bin/python"
TOOLS = Path("/home/rna_jev/tools")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/nucleicbert_secondary_structure_round34"
OUT = ROOT / "eval_decision/nb_secstr_round35_b4"
SMOKE = ROOT / "eval_decision/nb_secstr_round35_b4_smoke32"
STATE = ROOT / "audits/nb_secstr_round35_b4/state.json"
FULL_SPLITS = ("train_rnalign_tr0", "vl0", "ts0", "archiveii600")
ENV = {**os.environ,
       "PYTHONPATH": "/home/rna_jev/src:/home/rna_jev/tools",
       "NB_CKPT": "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt"}
BATCH = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_pass(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text()).get("status") == "PASS"


def qa_pass(run_dir: Path) -> bool:
    run, qa = run_dir / "run.json", run_dir / "qa.json"
    return (result_pass(run) and result_pass(qa)
            and json.loads(qa.read_text()).get("run_sha256") == sha256(run))


def extraction_done() -> bool:
    manifest = DATA / "embeddings/manifest.json"
    if not manifest.is_file():
        return False
    entries = json.loads(manifest.read_text()).get("entries", [])
    return all(any(e.get("split") == split and Path(e["path"]).is_file()
                   and Path(e["path"]).with_suffix(".json").is_file() for e in entries)
               for split in FULL_SPLITS)


def free_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return int([line for line in text.splitlines() if line.strip()][0])


_LOG_HANDLES: list = []


def launch(head: str, seed: int, limit: int, out_root: Path, log_dir: Path) -> subprocess.Popen:
    out = out_root / f"{head}_s{seed}"
    command = [PY, str(TOOLS / "nb_secstr_dual_round34.py"), "--head", head, "--seed", str(seed),
               "--out-dir", str(out), "--device", "cuda", "--batch-size", str(BATCH)]
    if limit:
        command.extend(["--limit", str(limit)])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{head}_s{seed}.launch.log"
    handle = log_path.open("a")
    _LOG_HANDLES.append(handle)
    handle.write(f"\nSTART {datetime.now(timezone.utc).isoformat()} cmd={' '.join(command)}\n")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=ENV)


def run_qa(head: str, seed: int, out_root: Path, log_dir: Path) -> None:
    run_dir = out_root / f"{head}_s{seed}"
    log_path = log_dir / f"{head}_s{seed}.qa.log"
    with log_path.open("a") as handle:
        completed = subprocess.run(
            [PY, str(TOOLS / "qa_nb_secstr_round34.py"), "--run-dir", str(run_dir), "--device", "cuda"],
            stdout=handle, stderr=subprocess.STDOUT, env=ENV, check=False)
    if completed.returncode != 0 or not qa_pass(run_dir):
        raise RuntimeError(f"QA failed for {run_dir}; queue stopped without retry")


def record(entry: dict) -> None:
    state = json.loads(STATE.read_text()) if STATE.is_file() else {"jobs": {}}
    state["jobs"].setdefault("runs", {})[f"{entry['head']}_s{entry['seed']}"] = entry
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(STATE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=8000)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not extraction_done():
        raise RuntimeError("full extraction is not complete; refusing to train")
    if not args.dry_run:
        SMOKE.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)

    # Smoke gate: official then JEV, batch 4, 32 longest structures.
    for head in ("official", "jev"):
        run_dir = SMOKE / f"{head}_s0"
        if not result_pass(run_dir / "run.json"):
            if (run_dir / "run.json").is_file():
                raise RuntimeError(f"smoke exists but not PASS: {run_dir}")
            if args.dry_run:
                print(f"DRY smoke {head}")
                continue
            if free_mib() < args.min_free_mib:
                raise RuntimeError("not enough free memory for smoke")
            process = launch(head, 0, 32, SMOKE, SMOKE)
            if process.wait() != 0:
                raise RuntimeError(f"smoke training failed: {run_dir}")
        if not qa_pass(run_dir) and not args.dry_run:
            run_qa(head, 0, SMOKE, SMOKE)
        print(f"SMOKE_PASS {head}", flush=True)

    pending = [(head, seed) for head in ("official", "jev") for seed in range(5)]
    pending = [item for item in pending if not result_pass(OUT / f"{item[0]}_s{item[1]}/run.json")]
    if args.dry_run:
        print("DRY full pending", pending, "parallel", args.parallel)
        return 0

    running: dict[tuple[str, int], subprocess.Popen] = {}
    while pending or running:
        for item in list(running):
            if running[item].poll() is not None:
                if running[item].returncode != 0:
                    raise RuntimeError(f"training failed: {item}")
                run_qa(item[0], item[1], OUT, OUT)
                record({"head": item[0], "seed": item[1], "status": "PASS",
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
                print(f"RUN_QA_PASS {item[0]}_s{item[1]}", flush=True)
                del running[item]
        while pending and len(running) < args.parallel:
            if free_mib() < args.min_free_mib:
                print(f"WAIT free={free_mib()} required={args.min_free_mib}", flush=True)
                break
            item = pending.pop(0)
            running[item] = launch(item[0], item[1], 0, OUT, OUT)
            print(f"RUN_START {item[0]}_s{item[1]} pid={running[item].pid}", flush=True)
        time.sleep(args.interval if running else 1)

    aggregate = OUT / "result.json"
    if not result_pass(aggregate):
        command = [PY, str(TOOLS / "aggregate_nb_secstr_round34.py"),
                   "--runs-root", str(OUT), "--out", str(aggregate)]
        with (OUT / "aggregate.launch.log").open("a") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=ENV, check=False)
        if completed.returncode != 0:
            raise RuntimeError("aggregation failed")
    print("NB_SECSTR_ROUND35_B4_QUEUE_DONE", aggregate, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
