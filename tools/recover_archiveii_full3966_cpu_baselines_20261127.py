#!/usr/bin/env python3
"""Resumable per-ID CPU baseline recovery for ArchiveII official-only 1422 records."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_PANEL = ROOT / "source_data/archiveii_official_reference_20261127/official_reference_3966.jsonl.gz"
CLEAN_PANEL = ROOT / "source_data/archiveii_official_reference_20261127/official_reference_clean_2544.jsonl.gz"
PLAN_PATH = ROOT / "source_data/archiveii_full3966_cpu_recovery_20261127/recovery_plan.json"
DEFAULT_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_full3966_cpu_20261127")
BASELINES = ("contrafold", "eternafold", "mxfold2", "vienna_mfe", "vienna_centroid", "vienna_mea",
             "rnastructure_fold", "rnastructure_maxexpect")
SCHEMA = "archiveii.full3966.cpu_recovery.v1"
ETERNA_REPO = Path("/home/rna_jev/third_party/EternaFold")
RNASTRUCTURE_REPO = Path("/home/rna_jev/packages/rnastructure-6.6/RNAstructure")
MX_ENV_PYTHON = Path("/root/miniconda3/envs/lucaone/bin/python")
MX_CONF = Path("/root/miniconda3/envs/lucaone/lib/python3.10/site-packages/mxfold2/models/TrainSetAB.conf")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical(row) + "\n")
    os.replace(temporary, path)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recovery_rows():
    full = read_gzip(FULL_PANEL)
    clean = {row["name"] for row in read_gzip(CLEAN_PANEL)}
    rows = [{"name": row["name"], "seq": row["seq"], "sequence_sha256": row["sequence_sha256"],
             "pair_set_sha256": row["pair_set_sha256"], "reference_member": row["bpfold_member"],
             "reference_member_sha256": row["bpfold_member_sha256"]} for row in full if row["name"] not in clean]
    if len(rows) != 1422 or len({row["name"] for row in rows}) != 1422:
        raise ValueError("recovery population changed")
    return rows


def build_plan():
    rows = recovery_rows()
    plan = {"schema_version": SCHEMA + ".plan", "source_panel": str(FULL_PANEL.relative_to(ROOT)),
            "source_panel_sha256": sha256(FULL_PANEL), "clean_panel_sha256": sha256(CLEAN_PANEL),
            "n_records": len(rows), "non_acgu_records": sum(bool(set(row["seq"]) - set("ACGU")) for row in rows),
            "id_order_sha256": value_sha([row["name"] for row in rows]),
            "policy": {"sequence": "official BPSEQ sequence unchanged; per-ID sequence SHA required",
                       "non_acgu": "REJECTED_NON_ACGU without normalization or prediction",
                       "compute": "CPU only; one thread per worker; no GPU visibility",
                       "checkpoint": "one validated JSON per ID; resume skips only matching name/sequence/pairs/baseline"},
            "baselines": list(BASELINES), "rows": rows}
    write_json(PLAN_PATH, plan)
    return plan


def load_plan():
    if not PLAN_PATH.is_file():
        return build_plan()
    plan = json.loads(PLAN_PATH.read_text())
    expected = recovery_rows()
    if plan["rows"] != expected or plan["n_records"] != 1422:
        raise ValueError("recovery plan does not match official panels")
    return plan


def record_path(work_root, baseline, name):
    return work_root / baseline / "records" / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".json")


def pair_set(values, length):
    result, occupied = set(), set()
    for value in values:
        i, j = map(int, value)
        if not 0 <= i < j < length or i in occupied or j in occupied:
            raise ValueError(f"invalid pair {(i, j)} for length {length}")
        occupied.update((i, j))
        result.add((i, j))
    return sorted(result)


def dot_bracket_pairs(structure, length):
    if len(structure) != length:
        raise ValueError("dot-bracket length mismatch")
    stack, result = [], []
    for index, token in enumerate(structure):
        if token == "(":
            stack.append(index)
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket")
            result.append((stack.pop(), index))
        elif token != ".":
            raise ValueError("unsupported dot-bracket token")
    if stack:
        raise ValueError("unbalanced dot-bracket")
    return pair_set(result, length)


def parse_ct(text, sequence):
    lines = text.splitlines()
    length = len(sequence)
    if not lines or len(lines) % (length + 1):
        raise ValueError("CT block accounting mismatch")
    blocks = [lines[start:start + length + 1] for start in range(0, len(lines), length + 1)]
    pairs = []
    for index, line in enumerate(blocks[0][1:], start=1):
        fields = line.split()
        if len(fields) != 6 or int(fields[0]) != index or fields[1].upper().replace("T", "U") != sequence[index - 1]:
            raise ValueError("CT sequence/index mismatch")
        if int(fields[2]) != index - 1 or int(fields[3]) != (0 if index == length else index + 1) or int(fields[5]) != index:
            raise ValueError("CT neighboring indices mismatch")
        partner = int(fields[4]) - 1
        if partner > index - 1:
            pairs.append((index - 1, partner))
    return pair_set(pairs, length), len(blocks)


def parse_mxfold2_dbn(text, expected_rows):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records, index = [], 0
    while index < len(lines):
        if not lines[index].startswith(">") or index + 2 >= len(lines):
            raise ValueError("invalid MXfold2 DBN stream")
        name = lines[index][1:]
        sequence = lines[index + 1].upper().replace("T", "U")
        structure = lines[index + 2].split()[0]
        records.append((name, sequence, structure))
        index += 3
    if len(records) != len(expected_rows):
        raise ValueError(f"MXfold2 output count {len(records)} != {len(expected_rows)}")
    output = {}
    for expected, (_output_name, sequence, structure) in zip(expected_rows, records):
        if sequence != expected["seq"]:
            raise ValueError("MXfold2 output sequence/order mismatch")
        output[expected["name"]] = dot_bracket_pairs(structure, len(sequence))
    return output


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                DATAPATH=str(RNASTRUCTURE_REPO / "data_tables"))


def metadata(baseline):
    binary = ETERNA_REPO / "src/contrafold"
    common = {"cpu_threads": 1, "gpu_visible": False}
    if baseline in {"contrafold", "eternafold"}:
        params = ETERNA_REPO / "parameters/EternaFoldParams.v1"
        return {**common, "binary": str(binary), "binary_sha256": sha256(binary),
                "parameters": "native contrafold.params.*" if baseline == "contrafold" else str(params),
                "parameters_sha256": None if baseline == "contrafold" else sha256(params),
                "protocol": "official serial CPU binary and MEA decode"}
    if baseline == "mxfold2":
        weights = MX_CONF.with_suffix(".pth")
        return {**common, "python": str(MX_ENV_PYTHON), "config": str(MX_CONF), "config_sha256": sha256(MX_CONF),
                "weights": str(weights), "weights_sha256": sha256(weights), "model": "MixC",
                "protocol": "official CLI @TrainSetAB.conf; --gpu=-1 default; full-length FASTA"}
    if baseline.startswith("vienna_"):
        sys.path.insert(0, "/root/rna_jepa_data/pylibs")
        import RNA
        return {**common, "vienna_version": RNA.__version__, "protocol": f"official ViennaRNA {baseline} full-length sequence"}
    binary = RNASTRUCTURE_REPO / "exe" / ("Fold" if baseline == "rnastructure_fold" else "MaxExpect")
    return {**common, "binary": str(binary), "binary_sha256": sha256(binary),
            "datapath": str(RNASTRUCTURE_REPO / "data_tables"),
            "protocol": "official RNAstructure CPU binary; --MFE" if baseline == "rnastructure_fold" else "official RNAstructure CPU binary; --sequence"}


def success_record(baseline, row, pairs, extra=None):
    return {"schema_version": SCHEMA, "baseline": baseline, "name": row["name"],
            "sequence_sha256": row["sequence_sha256"], "sequence_length": len(row["seq"]),
            "status": "PASS", "pairs": pairs, "pair_set_sha256": value_sha(pairs),
            "non_acgu_characters": [], **(extra or {})}


def rejected_record(baseline, row):
    return {"schema_version": SCHEMA, "baseline": baseline, "name": row["name"],
            "sequence_sha256": row["sequence_sha256"], "sequence_length": len(row["seq"]),
            "status": "REJECTED_NON_ACGU", "pairs": None, "pair_set_sha256": None,
            "non_acgu_characters": sorted(set(row["seq"]) - set("ACGU"))}


def validate_saved(record, baseline, row):
    if record["baseline"] != baseline or record["name"] != row["name"] or record["sequence_sha256"] != row["sequence_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    if record["status"] == "REJECTED_NON_ACGU":
        if record["pairs"] is not None or not record["non_acgu_characters"]:
            raise ValueError("invalid rejection checkpoint")
        return True
    if record["status"] != "PASS":
        raise ValueError("unknown checkpoint status")
    expected = pair_set(record["pairs"], len(row["seq"]))
    if expected != [tuple(pair) for pair in record["pairs"]] or record["pair_set_sha256"] != value_sha(expected):
        raise ValueError("checkpoint pair identity mismatch")
    if set(row["seq"]) - set("ACGU"):
        raise ValueError("PASS checkpoint for non-ACGU sequence")
    return True


def run_command(command, env, cwd=None):
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env, cwd=cwd)


def predict_external(baseline, row, work_directory):
    env = cpu_environment()
    sequence_path = work_directory / "input.seq"
    sequence_path.write_text(row["seq"] + "\n")
    if baseline in {"contrafold", "eternafold"}:
        command = [str(ETERNA_REPO / "src/contrafold"), "predict", str(sequence_path)]
        if baseline == "eternafold":
            command.extend(["--params", str(ETERNA_REPO / "parameters/EternaFoldParams.v1")])
        completed = run_command(command, env)
        lines = completed.stdout.splitlines()
        structure = lines[lines.index(">structure") + 1].strip()
        return dot_bracket_pairs(structure, len(row["seq"])), {}
    fasta = work_directory / "input.fa"
    fasta.write_text(f">{row['name']}\n{row['seq']}\n")
    output = work_directory / "output.ct"
    output.unlink(missing_ok=True)
    binary = RNASTRUCTURE_REPO / "exe" / ("Fold" if baseline == "rnastructure_fold" else "MaxExpect")
    command = [str(binary), str(fasta), str(output), "--MFE" if baseline == "rnastructure_fold" else "--sequence"]
    completed = run_command(command, env)
    pairs, blocks = parse_ct(output.read_text(), row["seq"])
    return pairs, {"ct_structures": blocks}


def predict_vienna(baseline, row, rna_module):
    if baseline == "vienna_mfe":
        structure, _ = rna_module.fold(row["seq"])
    else:
        compound = rna_module.fold_compound(row["seq"])
        compound.pf()
        if baseline == "vienna_centroid":
            structure, _ = compound.centroid()
        else:
            structure, _ = compound.MEA()
    return dot_bracket_pairs(structure, len(row["seq"]))


def predict_mxfold2_shard(rows, work_directory):
    fasta = work_directory / "input.fa"
    fasta.write_text("".join(f">{row['name']}\n{row['seq']}\n" for row in rows))
    command = [str(MX_ENV_PYTHON), "-m", "mxfold2", "predict", str(fasta), "@" + MX_CONF.name]
    completed = run_command(command, cpu_environment(), cwd=MX_CONF.parent)
    return parse_mxfold2_dbn(completed.stdout, rows)


def consolidate(work_root, baseline, rows):
    records, remaining = [], []
    for row in rows:
        path = record_path(work_root, baseline, row["name"])
        if not path.is_file():
            remaining.append(row["name"])
            continue
        record = json.loads(path.read_text())
        validate_saved(record, baseline, row)
        records.append(record)
    write_gzip_jsonl(work_root / baseline / "predictions.jsonl.gz", records)
    coverage = {"schema_version": SCHEMA + ".coverage", "baseline": baseline,
                "n_target": len(rows), "n_completed": len(records),
                "n_pass": sum(row["status"] == "PASS" for row in records),
                "n_rejected_non_acgu": sum(row["status"] == "REJECTED_NON_ACGU" for row in records),
                "coverage": len(records) / len(rows), "remaining_count": len(remaining), "remaining_names": remaining,
                "id_order_sha256": value_sha([row["name"] for row in records]),
                "predictions_sha256": sha256(work_root / baseline / "predictions.jsonl.gz"),
                "status": "COMPLETE" if not remaining else "PARTIAL"}
    write_json(work_root / baseline / "coverage.json", coverage)
    return coverage


def acquire_lock(work_root, baseline, command):
    directory = work_root / baseline
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "RUNNING.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"baseline already running or stale lock: {lock}") from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "command": command}) + "\n")
    return lock


def run_baseline(baseline, work_root, rows, shard_size, limit=None):
    if baseline not in BASELINES:
        raise ValueError(f"unknown baseline {baseline}")
    selected = rows[:limit] if limit else rows
    lock = acquire_lock(work_root, baseline, " ".join(sys.argv))
    try:
        if baseline.startswith("vienna_"):
            sys.path.insert(0, "/root/rna_jepa_data/pylibs")
            import RNA
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        else:
            RNA = None
        groups = [selected[index:index + shard_size] for index in range(0, len(selected), shard_size)] if baseline == "mxfold2" else [[row] for row in selected]
        for group in groups:
            pending = [row for row in group if not record_path(work_root, baseline, row["name"]).is_file()]
            if not pending:
                continue
            for row in pending:
                path = record_path(work_root, baseline, row["name"])
                if path.is_file():
                    validate_saved(json.loads(path.read_text()), baseline, row)
            if baseline == "mxfold2":
                valid_pending = []
                for row in pending:
                    if set(row["seq"]) - set("ACGU"):
                        write_json(record_path(work_root, baseline, row["name"]), rejected_record(baseline, row))
                    else:
                        valid_pending.append(row)
                if valid_pending:
                    with tempfile.TemporaryDirectory(prefix="archiveii_mxfold2_") as temporary:
                        shard_predictions = predict_mxfold2_shard(valid_pending, Path(temporary))
                    for row in valid_pending:
                        record = success_record(baseline, row, shard_predictions[row["name"]])
                        write_json(record_path(work_root, baseline, row["name"]), record)
            else:
                for row in pending:
                    if set(row["seq"]) - set("ACGU"):
                        record = rejected_record(baseline, row)
                    elif baseline.startswith("vienna_"):
                        record = success_record(baseline, row, predict_vienna(baseline, row, RNA))
                    else:
                        with tempfile.TemporaryDirectory(prefix="archiveii_cpu_baseline_") as temporary:
                            pairs, extra = predict_external(baseline, row, Path(temporary))
                        record = success_record(baseline, row, pairs, extra)
                    write_json(record_path(work_root, baseline, row["name"]), record)
            coverage = consolidate(work_root, baseline, rows)
            print(json.dumps({"baseline": baseline, "completed": coverage["n_completed"], "target": len(rows),
                              "remaining": coverage["remaining_count"]}), flush=True)
        write_json(work_root / baseline / "predictor_metadata.json", metadata(baseline))
        return consolidate(work_root, baseline, rows)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=8)
    args = parser.parse_args()
    plan = load_plan()
    if args.build_plan:
        print(json.dumps({"status": "PLAN_READY", "path": str(PLAN_PATH), "n_records": plan["n_records"],
                          "id_order_sha256": plan["id_order_sha256"]}, sort_keys=True))
        return
    if not args.baseline:
        parser.error("--baseline is required unless --build-plan")
    if args.status_only or args.consolidate_only:
        coverage = consolidate(args.work_root, args.baseline, plan["rows"])
        print(json.dumps(coverage, sort_keys=True))
        return
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    coverage = run_baseline(args.baseline, args.work_root, plan["rows"], args.shard_size, args.limit)
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
