#!/usr/bin/env python3
"""MMseqs 80%-identity / 80%-bidirectional-coverage leakage screen.

A negative screen is NOT_DETECTED_UNKNOWN, never clean. No performance data,
checkpoint, model selection or external-set tuning is involved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

JSONL_ROOT = Path("/root/rna_jepa_data/ss_data/jsonl")
TR1C = JSONL_ROOT / "bprna_tr1c.jsonl"
DATASET_NAMES = [
    "archiveii_embok_clean", "bprna_new", "bprna_ts0", "ref_pdb_ts1", "ref_pdb_ts2",
    "ref_pdb_ts3", "ref_pdb_ts_hard", "ref_synthetic_test", "pdb669", "rfam_fam",
    "rfam_temporal", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf",
]
FILES = {name: JSONL_ROOT / f"{name}.jsonl" for name in DATASET_NAMES}
CHANRG_EVAL = {"chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"}
CHANRG_TRAIN = [JSONL_ROOT / "chanrg_train_train.jsonl", JSONL_ROOT / "chanrg_train_validation.jsonl"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(row):
    return str(row.get("seq", row.get("sequence", ""))).upper().replace("T", "U")


def make_fasta(records, path, prefix, dataset=None):
    mapping, lengths = {}, {}
    with Path(path).open("w") as handle:
        for idx, row in enumerate(records):
            seq = norm(row)
            if not seq:
                continue
            key = f"{prefix}{idx:08d}"
            name = str(row.get("name", row.get("id", row.get("sequence_id", ""))))
            mapping[key] = {"dataset": dataset, "record_index": idx, "original_id": name,
                            "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest()}
            lengths[key] = len(seq)
            handle.write(f">{key}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start:start + 80] + "\n")
    return mapping, lengths


def run_easy_search(mmseqs, query, target, result, tmp_root, threads, log_path):
    tmp = Path(tmp_root) / (Path(result).stem + "_tmp")
    command = [str(mmseqs), "easy-search", str(query), str(target), str(result), str(tmp),
               "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0", "--search-type", "3", "-s", "7.5",
               "--max-seqs", "1000", "--threads", str(threads)]
    with Path(log_path).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return command


def parse_m8(path, query_lengths, target_lengths, query_map, target_map, identity=0.8, coverage=0.8):
    hits = []
    path = Path(path)
    if not path.exists():
        return hits
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.rstrip().split("\t")
            if len(fields) < 12:
                raise ValueError(f"{path}:{line_no}: expected 12 MMseqs fields")
            q, t = fields[0], fields[1]
            fident = float(fields[2])
            qstart, qend, tstart, tend = map(int, fields[6:10])
            qcov = (qend - qstart + 1) / query_lengths[q]
            tcov = (tend - tstart + 1) / target_lengths[t]
            if fident + 1e-12 < identity or qcov + 1e-12 < coverage or tcov + 1e-12 < coverage:
                continue
            hits.append({"query_key": q, "target_key": t,
                         "query_dataset": query_map[q]["dataset"],
                         "query_original_id": query_map[q]["original_id"],
                         "target_original_id": target_map[t]["original_id"],
                         "identity": fident, "query_coverage": qcov, "target_coverage": tcov,
                         "alignment_length": int(fields[3]), "evalue": float(fields[10]),
                         "bits": float(fields[11])})
    return hits


def summarize_hits(hits, query_map):
    grouped = collections.defaultdict(list)
    for hit in hits:
        grouped[hit["query_dataset"]].append(hit)
    output = {}
    for dataset in sorted(set(row["dataset"] for row in query_map.values())):
        rows = grouped.get(dataset, [])
        query_keys = {row["query_key"] for row in rows}
        query_hashes = {query_map[key]["sequence_sha256"] for key in query_keys}
        best = sorted(rows, key=lambda row: (-row["identity"], -min(row["query_coverage"], row["target_coverage"]),
                                            row["evalue"], row["query_key"]))[:20]
        output[dataset] = {
            "contract_status": "HOMOLOGY" if rows else "NOT_DETECTED_UNKNOWN",
            "n_hit_records": len(query_keys), "n_hit_unique_sequences": len(query_hashes),
            "n_alignments_passing_postfilter": len(rows),
            "n_query_records": sum(row["dataset"] == dataset for row in query_map.values()),
            "examples_top20": best,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True,
                        help="Local filesystem supporting MMseqs temporary symlinks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.work_dir.exists() or args.tmp_root.exists():
        raise FileExistsError("Refusing to overwrite an existing homology artifact/work/tmp directory")
    args.out.parent.mkdir(parents=True, exist_ok=False)
    args.work_dir.mkdir(parents=True)
    args.tmp_root.mkdir(parents=True)
    mmseqs = args.mmseqs.resolve()
    version = subprocess.check_output([str(mmseqs), "version"], text=True).strip()
    queries = {name: load_jsonl(path) for name, path in FILES.items()}
    train = load_jsonl(TR1C)
    chan_train = [row for path in CHANRG_TRAIN for row in load_jsonl(path)]
    empty = [name for name, rows in queries.items() if not rows]
    if not train or not chan_train or empty:
        raise ValueError(f"Empty required input: train={len(train)}, chanrg_train={len(chan_train)}, empty_eval={empty}")
    train_map, train_lengths = make_fasta(train, args.work_dir / "tr1c.fasta", "tr")
    chan_map, chan_lengths = make_fasta(chan_train, args.work_dir / "chanrg_train_validation.fasta", "ch")
    all_records = [(name, row) for name, rows in queries.items() for row in rows]
    # Build a combined query FASTA while preserving dataset identity.
    query_map, query_lengths = {}, {}
    with (args.work_dir / "all_eval.fasta").open("w") as handle:
        for idx, (dataset, row) in enumerate(all_records):
            seq = norm(row)
            if not seq:
                continue
            key = f"q{idx:08d}"
            query_map[key] = {"dataset": dataset, "record_index": idx,
                              "original_id": str(row.get("name", row.get("id", row.get("sequence_id", "")))),
                              "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest()}
            query_lengths[key] = len(seq)
            handle.write(f">{key}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start:start + 80] + "\n")
    commands = {}
    commands["vs_tr1c"] = run_easy_search(mmseqs, args.work_dir / "all_eval.fasta", args.work_dir / "tr1c.fasta",
                                          args.work_dir / "vs_tr1c.tsv", args.tmp_root, args.threads,
                                          args.work_dir / "vs_tr1c.log")
    chan_query = args.work_dir / "chanrg_eval.fasta"
    chan_query_map, chan_query_lengths = {}, {}
    with chan_query.open("w") as source, (args.work_dir / "all_eval.fasta").open() as combined:
        # Reuse only CHANRG entries; IDs and metadata remain identical.
        selected = {key for key, row in query_map.items() if row["dataset"] in CHANRG_EVAL}
        current = None
        for line in combined:
            if line.startswith(">"):
                current = line[1:].strip()
            if current in selected:
                source.write(line)
        chan_query_map = {key: row for key, row in query_map.items() if key in selected}
        chan_query_lengths = {key: value for key, value in query_lengths.items() if key in selected}
    commands["chanrg_vs_train_validation"] = run_easy_search(
        mmseqs, chan_query, args.work_dir / "chanrg_train_validation.fasta",
        args.work_dir / "chanrg_vs_train_validation.tsv", args.tmp_root, args.threads,
        args.work_dir / "chanrg_vs_train_validation.log")
    hits_tr1c = parse_m8(args.work_dir / "vs_tr1c.tsv", query_lengths, train_lengths, query_map, train_map)
    hits_chan = parse_m8(args.work_dir / "chanrg_vs_train_validation.tsv", chan_query_lengths, chan_lengths,
                         chan_query_map, chan_map)
    doc = {
        "schema": "rna-jev-mmseqs-homology-audit-v1",
        "status": "COMPLETE_WITH_PRETRAINING_EXPOSURE_UNCHECKED",
        "not_clean_claim": True, "not_sota": True, "not_model_selection": True,
        "threshold": {"identity_min": 0.8, "query_coverage_min": 0.8, "target_coverage_min": 0.8,
                      "cov_mode": 0, "search_type_nucleotide": 3, "sensitivity": 7.5},
        "mmseqs": {"path": str(mmseqs), "version": version, "sha256": sha256_file(mmseqs)},
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "commands": commands, "work_dir": str(args.work_dir), "tmp_root": str(args.tmp_root),
        "input_files": {str(path): sha256_file(path) for path in [TR1C, *FILES.values(), *CHANRG_TRAIN] if Path(path).exists()},
        "result_files": {name: sha256_file(args.work_dir / name) for name in ["vs_tr1c.tsv", "chanrg_vs_train_validation.tsv"]},
        "query_map": query_map, "query_lengths": query_lengths,
        "target_tr1c_map": train_map, "target_tr1c_lengths": train_lengths,
        "target_chanrg_map": chan_map, "target_chanrg_lengths": chan_lengths,
        "vs_tr1c": summarize_hits(hits_tr1c, query_map),
        "chanrg_vs_train_validation": summarize_hits(hits_chan, chan_query_map),
        "scope": "MMseqs local-alignment screen only; no pretraining-exposure or temporal-provenance conclusion",
    }
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": doc["status"], "out": str(args.out),
                      "vs_tr1c": {key: value["n_hit_records"] for key, value in doc["vs_tr1c"].items()},
                      "chanrg_vs_train_validation": {key: value["n_hit_records"] for key, value in doc["chanrg_vs_train_validation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
