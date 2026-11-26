#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_REPOSITORY = "https://github.com/LinearFold/LinearPartition.git"
EXPECTED_COMMIT = "b450fb3e63189073b68d385589035f992080aa3a"
EXPECTED_SHORT_LENGTH = 76
DEFAULT_PYTHON = "/root/miniconda3/envs/codonflow/bin/python"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "hash_scope": "full_content"}


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def official_short_sequence(upstream: Path) -> str:
    lines = (upstream / "ecoli_tRNA").read_text(encoding="utf-8").splitlines()
    sequence = next(line.strip().upper() for line in lines if line.strip() and not line.startswith(">"))
    if len(sequence) != EXPECTED_SHORT_LENGTH or set(sequence) - set("ACGU"):
        raise ValueError("unexpected official ecoli_tRNA sequence")
    return sequence


def parse_bpp(path: Path, sequence_length: int) -> list[dict]:
    pairs = {}
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(">"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"invalid BPP row {row_number}")
            left, right = int(fields[0]), int(fields[1])
            probability = float(fields[2])
            if not 1 <= left < right <= sequence_length:
                raise ValueError(f"out-of-range BPP row {row_number}")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid BPP probability row {row_number}")
            pair = (left, right)
            if pair in pairs:
                raise ValueError(f"duplicate BPP pair row {row_number}")
            pairs[pair] = probability
    return [{"pair_1based_inclusive": list(pair), "probability": pairs[pair]}
            for pair in sorted(pairs)]


def full_length_structures(stdout: str, sequence_length: int) -> list[str]:
    structures = [line.strip() for line in stdout.splitlines()
                  if re.fullmatch(r"[().]+", line.strip())]
    if not structures or any(len(structure) != sequence_length for structure in structures):
        raise ValueError("missing or truncated full-length MEA structure")
    for structure in structures:
        balance = 0
        for symbol in structure:
            balance += symbol == "("
            balance -= symbol == ")"
            if balance < 0:
                raise ValueError("unbalanced MEA structure")
        if balance:
            raise ValueError("unbalanced MEA structure")
    return structures


def classify_execution(beam_size: int, cutoff: float) -> dict:
    if beam_size != 100 or cutoff != 0.0:
        raise ValueError("this recovery protocol is frozen at beam=100 and cutoff=0.0")
    return {
        "column": "APPROXIMATE_SPARSE",
        "exact_global_dp": False,
        "reason": "finite beam pruning; emitted BPP support is sparse and absent pairs are not exact zeros",
    }


def recover(upstream: Path, evidence: Path, python: Path) -> dict:
    upstream = upstream.resolve()
    evidence = evidence.resolve()
    if git_output(upstream, "remote", "get-url", "origin") != EXPECTED_REPOSITORY:
        raise AssertionError("unexpected LinearPartition origin")
    if git_output(upstream, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise AssertionError("unexpected LinearPartition commit")
    if git_output(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise AssertionError("tracked official source is modified")
    evidence.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(["make", "-j2"], cwd=upstream, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    (evidence / "build.log").write_text(build.stdout, encoding="utf-8")

    sequence = official_short_sequence(upstream)
    fasta = f">official_ecoli_tRNA\n{sequence}\n"
    input_path = evidence / "official_short_input.fasta"
    input_path.write_text(fasta, encoding="utf-8")
    rows = []
    for variant, flag in (("LinearPartition-C", None), ("LinearPartition-V", "-V")):
        bpp_path = evidence / f"short_{variant}.bpp"
        command = [str(python), str(upstream / "linearpartition"), "--fasta",
                   "-b", "100", "-c", "0.0", "-M", "--rewrite", str(bpp_path)]
        if flag:
            command.append(flag)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        completed = subprocess.run(command, cwd=upstream, input=fasta, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=environment, check=True)
        (evidence / f"short_{variant}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (evidence / f"short_{variant}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        pairs = parse_bpp(bpp_path, len(sequence))
        structures = full_length_structures(completed.stdout, len(sequence))
        rows.append({
            "variant": variant, "command": command,
            "execution": classify_execution(100, 0.0),
            "scope": "OFFICIAL_SHORT_CPU_SMOKE_ONLY_NOT_LONG_CHAIN_PREDICTION",
            "sequence_length": len(sequence), "sequence_sha256": sha256_text(sequence),
            "bpp_pair_rows": len(pairs), "bpp_pairs_sha256": sha256_text(
                json.dumps(pairs, sort_keys=True, separators=(",", ":"))),
            "bpp_output": file_record(bpp_path),
            "full_length_mea_structures": structures,
            "stdout": file_record(evidence / f"short_{variant}.stdout.log"),
            "stderr": file_record(evidence / f"short_{variant}.stderr.log"),
            "returncode": completed.returncode,
        })
    source_names = ["README.md", "LICENSE", "Makefile", "linearpartition",
                    "src/LinearPartition.cpp", "src/LinearPartition.h", "src/bpp.cpp",
                    "src/Utils/feature_weight.h", "src/Utils/energy_parameter.h"]
    report = {
        "schema_version": "linearpartition_recovery_v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_OFFICIAL_SOURCE_BUILD_AND_SHORT_CPU_SMOKE",
        "official_repository": EXPECTED_REPOSITORY,
        "commit": EXPECTED_COMMIT,
        "commit_utc": git_output(upstream, "show", "-s", "--format=%cI", "HEAD"),
        "tree": git_output(upstream, "rev-parse", "HEAD^{tree}"),
        "version_selection": "official master pinned by full commit; v1.0 is not substituted",
        "v1_0_commit": git_output(upstream, "rev-parse", "v1.0^{}"),
        "source_files": [file_record(upstream / name) for name in source_names],
        "license": {
            "status": "VERIFIED_CUSTOM_LICENSE_NOT_SPDX_ASSIGNED",
            **file_record(upstream / "LICENSE"),
            "terms_summary": "research/educational/commercial use and modification permitted with conditions; no fee redistribution beyond media costs; proper credit; commercial-product inclusion requires contacting corresponding author",
        },
        "embedded_parameters": {
            "external_checkpoint_required": False,
            "LinearPartition-C": "embedded feature_weight.h; not parameter-free",
            "LinearPartition-V": "embedded Vienna energy_parameter.h",
        },
        "binaries": {variant: file_record(upstream / "bin" / ("linearpartition_c" if variant.endswith("-C") else "linearpartition_v"))
                     for variant in ("LinearPartition-C", "LinearPartition-V")},
        "build_log": file_record(evidence / "build.log"),
        "official_short_input": {**file_record(input_path), "sequence_sha256": sha256_text(sequence),
                                 "sequence_length": len(sequence)},
        "rows": rows,
        "long_chain_inference_executed": False,
        "gpu_execution": False,
        "scientific_long_sequence_completion": False,
    }
    output = evidence / "recovery_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(DEFAULT_PYTHON))
    args = parser.parse_args()
    report = recover(args.upstream, args.evidence, args.python)
    print(json.dumps({"status": report["status"], "variants": [
        {"variant": row["variant"], "length": row["sequence_length"],
         "bpp_pair_rows": row["bpp_pair_rows"],
         "structure_lengths": [len(value) for value in row["full_length_mea_structures"]]}
        for row in report["rows"]], "long_chain_inference_executed": False}, indent=2))


if __name__ == "__main__":
    main()
