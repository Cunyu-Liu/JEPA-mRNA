#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

BUCKETS = ("0-49", "50-199", "200-999", ">=1000")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def check_file(record: dict) -> None:
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"file identity mismatch: {path}")


def read_bpp(path: Path, length: int) -> list[tuple[int, int, float]]:
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith(">"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"bad BPP row {line_number}")
        left, right, probability = int(fields[0]), int(fields[1]), float(fields[2])
        if not 1 <= left < right <= length or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"invalid BPP row {line_number}")
        if (left, right) in seen:
            raise ValueError("duplicate BPP pair")
        seen.add((left, right))
        rows.append((left - 1, right - 1, probability))
    return rows


def read_mea(path: Path, sequence: str) -> tuple[str, list[tuple[int, int]]]:
    lines = path.read_text().splitlines()
    if len(lines) != 3 or lines[1] != sequence or len(lines[2]) != len(sequence):
        raise ValueError("MEA does not preserve the complete input")
    structure = lines[2]
    stack, pairs = [], []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced MEA")
            pairs.append((stack.pop(), index))
        elif symbol != ".":
            raise ValueError("invalid MEA symbol")
    if stack:
        raise ValueError("unbalanced MEA")
    return structure, pairs


def bucket(pair: tuple[int, int]) -> str:
    span = pair[1] - pair[0]
    return "0-49" if span < 50 else "50-199" if span < 200 else "200-999" if span < 1000 else ">=1000"


def recall(reference: list[list[int]], candidates: list[tuple[int, int]]) -> dict:
    ref, pred = {tuple(pair) for pair in reference}, set(candidates)
    result = {"overall": {"reference": len(ref), "true_positive": len(ref & pred),
                          "recall": len(ref & pred) / len(ref) if ref else None}}
    for name in BUCKETS:
        selected = {pair for pair in ref if bucket(pair) == name}
        true_positive = len(selected & pred)
        result[name] = {"reference": len(selected), "true_positive": true_positive,
                        "recall": true_positive / len(selected) if selected else None}
    return result


def parse_time(path: Path) -> tuple[float, int]:
    text = path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if not elapsed or not rss:
        raise ValueError("incomplete GNU time evidence")
    seconds = 0.0
    for part in elapsed.group(1).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds, int(rss.group(1)) * 1024


def verify(run_dir: Path, target_manifest: Path) -> dict:
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text())
    name = report["name"]
    targets = json.loads(target_manifest.read_text())["targets"]
    target = next(target for target in targets if target["name"] == name)
    if target["split"] != "train" or target["length"] != 7224 or not report["train_only"] \
            or report["independent_scientific_evaluation"] is not False \
            or report["execution"]["column"] != "APPROXIMATE_SPARSE":
        raise ValueError("scope/column contract mismatch")
    check_file(report["input"])
    input_lines = Path(report["input"]["path"]).read_text().splitlines()
    if len(input_lines) != 2:
        raise ValueError("unexpected FASTA")
    sequence = input_lines[1]
    if len(sequence) != 7224 or hashlib.sha256(sequence.encode()).hexdigest() != target["sequence_sha256"]:
        raise ValueError("untruncated input identity mismatch")
    for output in report["outputs"].values():
        if isinstance(output, dict) and "path" in output:
            check_file(output)
    check_file(report["execution"]["official_binary"])
    bpp_path = Path(report["outputs"]["bpp"]["path"])
    bpp = read_bpp(bpp_path, 7224)
    structure, mea = read_mea(Path(report["outputs"]["mea"]["path"]), sequence)
    stdout = Path(report["outputs"]["stdout"]["path"]).read_text()
    if structure not in stdout.splitlines():
        raise ValueError("standalone MEA differs from raw stdout")
    if len(bpp) != report["outputs"]["bpp_pair_rows"] or len(structure) != report["outputs"]["mea_length"]:
        raise ValueError("output counts disagree with run report")
    metrics_path = Path(report["metrics"]["path"])
    check_file(report["metrics"])
    metrics = json.loads(metrics_path.read_text())
    recomputed = {
        "bpp_candidate_pair_recall": recall(target["pairs"], [(left, right) for left, right, _ in bpp]),
        "mea_pair_recall": recall(target["pairs"], mea),
        "bpp_candidate_pair_rows": len(bpp), "mea_pair_count": len(mea),
    }
    for key, value in recomputed.items():
        if metrics[key] != value:
            raise ValueError(f"metric mismatch: {key}")
    coverage = metrics["coverage"]
    if coverage["record_coverage"] != 1.0 or coverage["completed_target_coverage"] != 1.0 \
            or coverage["completed_nucleotide_coverage"] != 1.0 or coverage["truncated"] is not False:
        raise ValueError("full coverage contract failed")
    time_record = report["resources"]["gnu_time"]
    check_file(time_record)
    wall, rss = parse_time(Path(time_record["path"]))
    if wall != report["resources"]["latency"]["wall_seconds"] \
            or rss != report["resources"]["memory"]["peak_cpu_rss_bytes"]:
        raise ValueError("resource values disagree with raw GNU time")
    flops = report["resources"]["flops"]
    check_file(flops["capability_evidence"])
    if flops["status"] != "NOT_MEASURED" or flops["value"] is not None or flops["estimated"] is not False \
            or "Operation not permitted" not in Path(flops["capability_evidence"]["path"]).read_text():
        raise ValueError("FLOPs unavailable evidence is inconsistent")
    return {
        "schema_version": "linearpartition_7224_independent_verification_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_INDEPENDENT_RAW_OUTPUT_HASH_RESOURCE_AND_METRIC_VERIFICATION",
        "name": name, "split": "train", "train_only": True,
        "independent_scientific_evaluation": False,
        "execution_column": "APPROXIMATE_SPARSE", "full_length": 7224, "truncated": False,
        "run_report_sha256": digest(report_path),
        "verified_files": [report["input"], report["execution"]["official_binary"],
                           report["outputs"]["bpp"], report["outputs"]["mea"],
                           report["outputs"]["stdout"], report["outputs"]["stderr"],
                           report["metrics"], time_record, flops["capability_evidence"]],
        "recomputed": {**recomputed, "coverage": coverage,
                       "wall_seconds": wall, "peak_cpu_rss_bytes": rss,
                       "flops_status": "NOT_MEASURED"},
        "gpu_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run_dir, args.target_manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "name": report["name"],
                      "recomputed": report["recomputed"]}, indent=2))


if __name__ == "__main__":
    main()
