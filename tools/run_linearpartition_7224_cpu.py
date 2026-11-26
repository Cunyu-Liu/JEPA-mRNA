#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from longseq_acceptance_contract import file_record, pair_recall, sha256_file, sha256_text
from recover_linearpartition_longseq import full_length_structures, parse_bpp

ALLOWED_TRAIN_TARGETS = {"9MR4_5", "9Q7Q_5"}


def parse_gnu_time(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    def capture(pattern: str) -> str:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"missing GNU time field: {pattern}")
        return match.group(1)
    elapsed = capture(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
    pieces = [float(piece) for piece in elapsed.split(":")]
    wall_seconds = 0.0
    for piece in pieces:
        wall_seconds = wall_seconds * 60 + piece
    return {
        "wall_seconds": wall_seconds,
        "user_seconds": float(capture(r"User time \(seconds\): (\S+)")),
        "system_seconds": float(capture(r"System time \(seconds\): (\S+)")),
        "peak_cpu_rss_bytes": int(capture(r"Maximum resident set size \(kbytes\): (\d+)")) * 1024,
        "exit_status": int(capture(r"Exit status: (\d+)")),
        "raw_elapsed": elapsed,
    }


def mea_pairs(structure: str) -> list[dict]:
    stack, pairs = [], []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced MEA structure")
            pairs.append({"pair": [stack.pop(), index], "probability": None})
    if stack:
        raise ValueError("unbalanced MEA structure")
    return sorted(pairs, key=lambda item: item["pair"])


def load_frozen_target(repo: Path, manifest_path: Path, name: str) -> tuple[dict, str]:
    if name not in ALLOWED_TRAIN_TARGETS:
        raise ValueError("this runner is restricted to the two preregistered 7224nt train chains")
    document = json.loads(manifest_path.read_text())
    matches = [target for target in document["targets"] if target["name"] == name]
    if len(matches) != 1:
        raise ValueError("target identity is not unique")
    target = matches[0]
    if target["length"] != 7224 or target["split"] != "train":
        raise ValueError("unexpected target role or length")
    frozen = target["frozen_jsonl"]
    path = Path(frozen["path"])
    if sha256_file(path) != frozen["sha256"]:
        raise ValueError("frozen JSONL hash mismatch")
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == target["row_index_zero_based"]:
                row = json.loads(line)
                break
        else:
            raise ValueError("frozen target row missing")
    sequence = row["seq"].upper().replace("T", "U")
    if row["name"] != name or len(sequence) != 7224 or sha256_text(sequence) != target["sequence_sha256"]:
        raise ValueError("frozen sequence identity mismatch")
    if row["pairs"] != target["pairs"]:
        raise ValueError("frozen reference pair mismatch")
    return target, sequence


def run_target(repo: Path, manifest_path: Path, recovery_path: Path, upstream: Path,
               name: str, run_dir: Path, time_binary: Path, perf_probe_log: Path) -> dict:
    target, sequence = load_frozen_target(repo, manifest_path, name)
    recovery = json.loads(recovery_path.read_text())
    binary = upstream / "bin/linearpartition_c"
    expected_binary = recovery["binaries"]["LinearPartition-C"]
    if recovery["commit"] != "b450fb3e63189073b68d385589035f992080aa3a" \
            or sha256_file(binary) != expected_binary["sha256"]:
        raise ValueError("official binary identity mismatch")
    if not perf_probe_log.is_file():
        raise ValueError("FLOPs capability evidence missing")
    run_dir.mkdir(parents=True, exist_ok=False)
    fasta = run_dir / "input_full_7224.fasta"
    fasta.write_text(f">{name}\n{sequence}\n", encoding="utf-8")
    bpp = run_dir / "output_full_7224.bpp"
    stdout_path = run_dir / "output_full_7224.stdout.log"
    stderr_path = run_dir / "output_full_7224.stderr.log"
    time_path = run_dir / "gnu_time_v.txt"
    command = [str(time_binary), "-v", "-o", str(time_path), str(binary),
               "100", "0", "1", str(bpp), "", "0", "0.0", "", "1", "3.0",
               "0", "0.3", "", "", "0", "", "1", "2", ""]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with fasta.open("rb") as stdin, stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, cwd=upstream, stdin=stdin, stdout=stdout,
                                   stderr=stderr, env=environment, check=False)
    timing = parse_gnu_time(time_path)
    if completed.returncode or timing["exit_status"]:
        raise RuntimeError(f"LinearPartition failed: {completed.returncode}")
    bpp_rows_1based = parse_bpp(bpp, len(sequence))
    bpp_rows = [{"pair": [row["pair_1based_inclusive"][0] - 1,
                           row["pair_1based_inclusive"][1] - 1],
                 "probability": row["probability"]} for row in bpp_rows_1based]
    stdout = stdout_path.read_text(encoding="utf-8")
    structures = full_length_structures(stdout, len(sequence))
    if len(structures) != 1:
        raise ValueError("expected exactly one full-length MEA structure")
    structure = structures[0]
    structure_path = run_dir / "output_full_7224.mea.dotbracket"
    structure_path.write_text(f">{name}\n{sequence}\n{structure}\n", encoding="utf-8")
    structure_pairs = mea_pairs(structure)
    metrics = {
        "scope": "TRAIN_ONLY_EXECUTION_AND_RECALL_NOT_INDEPENDENT_SCIENTIFIC_EVALUATION",
        "execution_column": "APPROXIMATE_SPARSE",
        "coverage": {"expected_targets": 1, "records": 1, "record_coverage": 1.0,
                     "completed_targets": 1, "completed_target_coverage": 1.0,
                     "input_nucleotides": len(sequence), "output_nucleotides": len(structure),
                     "completed_nucleotide_coverage": len(structure) / len(sequence),
                     "truncated": False},
        "bpp_candidate_pair_rows": len(bpp_rows),
        "bpp_candidate_pair_recall": pair_recall(target["pairs"], bpp_rows),
        "mea_pair_count": len(structure_pairs),
        "mea_pair_recall": pair_recall(target["pairs"], structure_pairs),
        "span_definition": "j-i, zero-based inclusive; empty buckets null",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    probe_text = perf_probe_log.read_text(encoding="utf-8")
    report = {
        "schema_version": "linearpartition_7224_cpu_run_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FULL_LENGTH_EXECUTION_AND_OUTPUT__FLOPS_NOT_MEASURED",
        "name": name, "split": "train", "train_only": True,
        "independent_scientific_evaluation": False,
        "execution": {"column": "APPROXIMATE_SPARSE", "variant": "LinearPartition-C",
                      "beam_size": 100, "bpp_cutoff": 0.0, "device": "cpu", "threads": 1,
                      "command": command, "official_binary": file_record(binary)},
        "input": {**file_record(fasta), "length": len(sequence), "sequence_sha256": sha256_text(sequence),
                  "frozen_jsonl_sha256": target["frozen_jsonl"]["sha256"], "truncated": False},
        "outputs": {"bpp": file_record(bpp), "mea": file_record(structure_path),
                    "stdout": file_record(stdout_path), "stderr": file_record(stderr_path),
                    "bpp_pair_rows": len(bpp_rows), "mea_length": len(structure)},
        "resources": {
            "latency": {"status": "MEASURED", "wall_seconds": timing["wall_seconds"],
                        "user_seconds": timing["user_seconds"], "system_seconds": timing["system_seconds"]},
            "memory": {"status": "MEASURED", "peak_cpu_rss_bytes": timing["peak_cpu_rss_bytes"],
                       "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None},
            "flops": {"status": "NOT_MEASURED", "value": None, "unit": "FLOPs", "estimated": False,
                      "method": None, "reason": "perf CLI absent; perf_event_open instructions probe returned Operation not permitted with kernel.perf_event_paranoid=4; no short-chain scaling estimate substituted",
                      "capability_evidence": file_record(perf_probe_log)},
            "gnu_time": {**file_record(time_path), "version": "GNU time 1.7", "parsed": timing},
        },
        "metrics": file_record(run_dir / "metrics.json"),
        "gpu_execution": False,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--name", choices=sorted(ALLOWED_TRAIN_TARGETS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--time-binary", type=Path, required=True)
    parser.add_argument("--perf-probe-log", type=Path, required=True)
    args = parser.parse_args()
    report = run_target(args.repo, args.target_manifest, args.recovery_report, args.upstream,
                        args.name, args.run_dir, args.time_binary, args.perf_probe_log)
    print(json.dumps({"status": report["status"], "name": report["name"],
                      "resources": report["resources"], "outputs": report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
