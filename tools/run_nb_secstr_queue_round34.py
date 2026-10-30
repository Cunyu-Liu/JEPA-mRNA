#!/usr/bin/env python3
"""Memory-gated smoke -> full extraction -> five-seed SSP queue; any failure stops the queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PYTHON = "/root/miniconda3/envs/lucaone/bin/python"
TOOLS = Path("/home/rna_jev/tools")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/nucleicbert_secondary_structure_round34"
EMBEDDINGS = DATA / "embeddings"
OUT = ROOT / "eval_decision/nb_secstr_round34"
SMOKE_OUT = ROOT / "eval_decision/nb_secstr_round34_smoke32"
ROLES = ("train", "validation", "ts0", "archiveii600")
FILES = {"train": "train_rnalign_tr0", "validation": "vl0", "ts0": "ts0", "archiveii600": "archiveii600"}
ENV = {**os.environ, "PYTHONPATH": "/home/rna_jev/src:/home/rna_jev/tools", "NB_CKPT": "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt"}


def free_mib() -> int:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0]
    total, used = (int(x.strip()) for x in text.split(","))
    return total - used


def extraction_done(split: str) -> bool:
    path = EMBEDDINGS / "manifest.json"
    if not path.is_file():
        return False
    entries = [e for e in json.loads(path.read_text()).get("entries", []) if e.get("split") == split]
    return bool(entries) and all(Path(e["path"]).is_file() and Path(e["path"]).with_suffix(".json").is_file() for e in entries)


def result_pass(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text()).get("status") == "PASS"


def qa_pass(run_dir: Path) -> bool:
    run = run_dir / "run.json"
    qa = run_dir / "qa.json"
    if not result_pass(run) or not result_pass(qa):
        return False
    import hashlib
    return json.loads(qa.read_text()).get("run_sha256") == hashlib.sha256(run.read_bytes()).hexdigest()


def extraction_job(role: str, limit: int) -> list[str]:
    split = f"smoke{limit}_{role}" if limit else FILES[role]
    return [
        PYTHON, str(TOOLS / "extract_nb_secstr_round34.py"),
        "--jsonl", str(DATA / "jsonl" / f"{FILES[role]}.jsonl"), "--split", split,
        "--out-dir", str(EMBEDDINGS), "--batch-size", "4", "--device", "cuda",
        *(["--limit", str(limit)] if limit else []),
    ]


def training_job(head: str, seed: int, limit: int) -> list[str]:
    out = (SMOKE_OUT if limit else OUT) / f"{head}_s{seed}"
    return [
        PYTHON, str(TOOLS / "nb_secstr_dual_round34.py"), "--head", head, "--seed", str(seed),
        "--out-dir", str(out), "--device", "cuda", "--batch-size", "1",
        *(["--limit", str(limit)] if limit else []),
    ]


def qa_job(head: str, seed: int, limit: int) -> list[str]:
    out = (SMOKE_OUT if limit else OUT) / f"{head}_s{seed}"
    return [PYTHON, str(TOOLS / "qa_nb_secstr_round34.py"), "--run-dir", str(out), "--device", "cuda"]


def next_action():
    for role in ROLES:
        split = f"smoke32_{role}"
        if not extraction_done(split):
            return "extract", extraction_job(role, 32), ROOT / "audits/nb_secstr_round34" / f"smoke32_extract_{role}.log", 8192
    for head in ("official", "jev"):
        run = SMOKE_OUT / f"{head}_s0/run.json"
        qa = SMOKE_OUT / f"{head}_s0/qa.json"
        if not result_pass(run):
            if run.is_file():
                raise RuntimeError(f"smoke run exists but is not PASS: {run}")
            return "train", training_job(head, 0, 32), SMOKE_OUT / f"{head}_s0.launch.log", 8192
        if not qa_pass(SMOKE_OUT / f"{head}_s0"):
            return "qa", qa_job(head, 0, 32), SMOKE_OUT / f"{head}_s0.qa.launch.log", 8192
    for role in ROLES:
        if not extraction_done(FILES[role]):
            return "extract", extraction_job(role, 0), ROOT / "audits/nb_secstr_round34" / f"full_extract_{role}.log", 8192
    for head, seed in [(h, s) for h in ("official", "jev") for s in range(5)]:
        run = OUT / f"{head}_s{seed}/run.json"
        qa = OUT / f"{head}_s{seed}/qa.json"
        if not result_pass(run):
            if run.is_file():
                raise RuntimeError(f"full run exists but is not PASS: {run}")
            return "train", training_job(head, seed, 0), OUT / f"{head}_s{seed}.launch.log", 8192
        if not qa_pass(OUT / f"{head}_s{seed}"): 
            return "qa", qa_job(head, seed, 0), OUT / f"{head}_s{seed}.qa.launch.log", 8192
    aggregate = OUT / "result.json"
    if not result_pass(aggregate):
        return "aggregate", [
            PYTHON, str(TOOLS / "aggregate_nb_secstr_round34.py"), "--runs-root", str(OUT), "--out", str(aggregate),
        ], OUT / "aggregate.launch.log", 0
    return "done", None, None, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    while True:
        kind, command, log, required = next_action()
        if kind == "done":
            print("NB_SECSTR_QUEUE_DONE", flush=True)
            return 0
        if args.dry_run:
            print(json.dumps({"kind": kind, "command": command, "log": str(log), "required_free_mib": required}), flush=True)
            return 0
        available = free_mib()
        if available < required:
            print(f"WAIT kind={kind} free={available} required={required}", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"START kind={kind} free={available} log={log} cmd={' '.join(command)}", flush=True)
        with log.open("a") as handle:
            handle.write(f"\nSTART {time.strftime('%Y-%m-%dT%H:%M:%S')} free_mib={available} cmd={' '.join(command)}\n")
            handle.flush()
            code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, env=ENV)
            handle.write(f"END {time.strftime('%Y-%m-%dT%H:%M:%S')} rc={code}\n")
        if code != 0:
            raise RuntimeError(f"job failed rc={code}; queue stopped without retry: {log}")
        print(f"PASS kind={kind} log={log}", flush=True)
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
