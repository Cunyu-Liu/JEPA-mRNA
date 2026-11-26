#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from longseq_acceptance_contract import file_record, pair_recall, sha256_file, sha256_text
from recover_linearpartition_longseq import full_length_structures, parse_bpp
from run_linearpartition_7224_cpu import mea_pairs, parse_gnu_time

EXPECTED_NAME = "7AM2_1"
EXPECTED_LENGTH = 19000
MINIMUM_AVAILABLE_BYTES = 8 * 1024**3
TIMEOUT_SECONDS = 600


def memavailable_bytes(meminfo: str) -> int:
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable missing")


def load_holdout(manifest_path: Path) -> tuple[dict, str]:
    document = json.loads(manifest_path.read_text())
    matches = [target for target in document["targets"] if target["name"] == EXPECTED_NAME]
    if len(matches) != 1:
        raise ValueError("holdout identity is not unique")
    target = matches[0]
    if target["split"] != "holdout" or target["length"] != EXPECTED_LENGTH:
        raise ValueError("unexpected holdout role or length")
    frozen = target["frozen_jsonl"]
    path = Path(frozen["path"])
    if sha256_file(path) != frozen["sha256"]:
        raise ValueError("frozen holdout JSONL hash mismatch")
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == target["row_index_zero_based"]:
                row = json.loads(line)
                break
        else:
            raise ValueError("holdout row missing")
    sequence = row["seq"].upper().replace("T", "U")
    if row["name"] != EXPECTED_NAME or len(sequence) != EXPECTED_LENGTH \
            or sha256_text(sequence) != target["sequence_sha256"] or row["pairs"] != target["pairs"]:
        raise ValueError("holdout frozen identity mismatch")
    return target, sequence


def optional_file_record(path: Path) -> dict | None:
    return file_record(path) if path.is_file() else None


def run_holdout(manifest_path: Path, recovery_path: Path, upstream: Path, run_dir: Path,
                time_binary: Path, timeout_binary: Path, perf_probe_log: Path) -> dict:
    target, sequence = load_holdout(manifest_path)
    recovery = json.loads(recovery_path.read_text())
    binary = upstream / "bin/linearpartition_c"
    if recovery["commit"] != "b450fb3e63189073b68d385589035f992080aa3a" \
            or sha256_file(binary) != recovery["binaries"]["LinearPartition-C"]["sha256"]:
        raise ValueError("official binary identity mismatch")
    available = memavailable_bytes(Path("/proc/meminfo").read_text())
    run_dir.mkdir(parents=True, exist_ok=False)
    gate = {"required_available_bytes": MINIMUM_AVAILABLE_BYTES,
            "observed_available_bytes": available, "passed": available >= MINIMUM_AVAILABLE_BYTES,
            "timeout_seconds": TIMEOUT_SECONDS}
    (run_dir / "resource_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["passed"]:
        raise RuntimeError("resource gate failed; no model process started")
    fasta = run_dir / "input_full_19000.fasta"
    fasta.write_text(f">{EXPECTED_NAME}\n{sequence}\n", encoding="utf-8")
    bpp = run_dir / "output_full_19000.bpp"
    stdout_path = run_dir / "output_full_19000.stdout.log"
    stderr_path = run_dir / "output_full_19000.stderr.log"
    time_path = run_dir / "gnu_time_v.txt"
    command = [str(time_binary), "-v", "-o", str(time_path), str(timeout_binary),
               "--signal=TERM", "--kill-after=10s", f"{TIMEOUT_SECONDS}s", str(binary),
               "100", "0", "1", str(bpp), "", "0", "0.0", "", "1", "3.0",
               "0", "0.3", "", "", "0", "", "1", "2", ""]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with fasta.open("rb") as stdin, stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, cwd=upstream, stdin=stdin, stdout=stdout,
                                   stderr=stderr, env=environment, check=False, timeout=630)
    timing = parse_gnu_time(time_path)
    timed_out = completed.returncode == 124 or timing["exit_status"] == 124
    succeeded = completed.returncode == 0 and timing["exit_status"] == 0
    outputs = {"bpp": optional_file_record(bpp), "stdout": file_record(stdout_path),
               "stderr": file_record(stderr_path)}
    metrics = None
    failure_reason = None
    if succeeded:
        try:
            bpp_rows_1based = parse_bpp(bpp, len(sequence))
            bpp_rows = [{"pair": [row["pair_1based_inclusive"][0] - 1,
                                   row["pair_1based_inclusive"][1] - 1],
                         "probability": row["probability"]} for row in bpp_rows_1based]
            structures = full_length_structures(stdout_path.read_text(), len(sequence))
            if len(structures) != 1:
                raise ValueError("expected one complete MEA")
            structure = structures[0]
            structure_path = run_dir / "output_full_19000.mea.dotbracket"
            structure_path.write_text(f">{EXPECTED_NAME}\n{sequence}\n{structure}\n", encoding="utf-8")
            structure_pairs = mea_pairs(structure)
            metrics = {
                "scope": "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION",
                "execution_column": "APPROXIMATE_SPARSE",
                "coverage": {"expected_targets": 1, "records": 1, "record_coverage": 1.0,
                             "completed_targets": 1, "completed_target_coverage": 1.0,
                             "input_nucleotides": len(sequence), "output_nucleotides": len(structure),
                             "completed_nucleotide_coverage": 1.0, "truncated": False},
                "bpp_candidate_pair_rows": len(bpp_rows),
                "bpp_candidate_pair_recall": pair_recall(target["pairs"], bpp_rows),
                "mea_pair_count": len(structure_pairs),
                "mea_pair_recall": pair_recall(target["pairs"], structure_pairs),
                "span_definition": "j-i, zero-based inclusive; empty buckets null",
                "not_for_model_selection": True,
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            outputs.update({"mea": file_record(structure_path), "bpp_pair_rows": len(bpp_rows),
                            "mea_length": len(structure)})
        except Exception as error:
            succeeded = False
            failure_reason = f"OUTPUT_VALIDATION_FAILED: {error}"
    else:
        failure_reason = "TIMEOUT_600S" if timed_out else f"NONZERO_EXIT_{completed.returncode}"
    report = {
        "schema_version": "linearpartition_19000_cpu_run_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FULL_LENGTH_HOLDOUT_EXECUTION_AND_OUTPUT__FLOPS_NOT_MEASURED" if succeeded else "FAILED_EXPLICIT_NO_ZERO_FILL",
        "name": EXPECTED_NAME, "split": "holdout", "holdout_descriptive_only": True,
        "not_for_model_selection": True, "failure_reason": failure_reason,
        "execution": {"column": "APPROXIMATE_SPARSE", "variant": "LinearPartition-C",
                      "beam_size": 100, "bpp_cutoff": 0.0, "device": "cpu", "threads": 1,
                      "timeout_seconds": TIMEOUT_SECONDS, "timed_out": timed_out,
                      "returncode": completed.returncode, "command": command,
                      "official_binary": file_record(binary)},
        "resource_gate": file_record(run_dir / "resource_gate.json"),
        "input": {**file_record(fasta), "length": len(sequence), "sequence_sha256": sha256_text(sequence),
                  "frozen_jsonl_sha256": target["frozen_jsonl"]["sha256"], "truncated": False},
        "outputs": outputs,
        "resources": {
            "latency": {"status": "MEASURED", "wall_seconds": timing["wall_seconds"],
                        "user_seconds": timing["user_seconds"], "system_seconds": timing["system_seconds"]},
            "memory": {"status": "MEASURED", "peak_cpu_rss_bytes": timing["peak_cpu_rss_bytes"],
                       "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None},
            "flops": {"status": "NOT_MEASURED", "value": None, "unit": "FLOPs", "estimated": False,
                      "method": None, "reason": "hardware counter access denied by host; no estimate substituted",
                      "capability_evidence": file_record(perf_probe_log)},
            "gnu_time": {**file_record(time_path), "version": "GNU time 1.7", "parsed": timing},
        },
        "metrics": file_record(run_dir / "metrics.json") if metrics else None,
        "gpu_execution": False,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--time-binary", type=Path, required=True)
    parser.add_argument("--timeout-binary", type=Path, default=Path("/usr/bin/timeout"))
    parser.add_argument("--perf-probe-log", type=Path, required=True)
    args = parser.parse_args()
    report = run_holdout(args.target_manifest, args.recovery_report, args.upstream, args.run_dir,
                         args.time_binary, args.timeout_binary, args.perf_probe_log)
    print(json.dumps({"status": report["status"], "name": report["name"],
                      "failure_reason": report["failure_reason"], "resources": report["resources"],
                      "outputs": report["outputs"]}, indent=2))
    if not report["status"].startswith("PASS"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
