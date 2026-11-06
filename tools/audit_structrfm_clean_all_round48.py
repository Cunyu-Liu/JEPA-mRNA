import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from audit_structrfm_clean_round48 import normalise, sha256


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fasta", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True,
                        help="name=jsonl; repeat for every A/B/C split")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    names = [item.split("=", 1)[0] for item in args.split]
    if len(names) != len(set(names)) or not all(names):
        raise ValueError("split names must be nonempty and unique")
    if len(names) != 13:
        raise ValueError(f"the registered A/B/C contract requires 13 splits, got {len(names)}")
    datasets = []
    query_rows = []
    query_fasta = args.work_dir / "all_splits.fa"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with query_fasta.open("w") as handle:
        for item in args.split:
            name, filename = item.split("=", 1)
            path = Path(filename)
            rows = [json.loads(line) for line in path.open() if line.strip()]
            if not rows:
                raise ValueError(f"empty split: {name}")
            datasets.append({"name": name, "path": path, "rows": rows,
                             "query_start": len(query_rows)})
            for index, row in enumerate(rows):
                handle.write(f">query_{len(query_rows)}\n{normalise(row['seq'])}\n")
                query_rows.append((name, index))
    alignment = args.work_dir / "alignment.tsv"
    command = [str(args.mmseqs), "easy-search", str(query_fasta), str(args.train_fasta),
               str(alignment), tempfile.mkdtemp(prefix="structrfm_mmseqs_all_"),
               "--min-seq-id", "0.8", "--cov-mode", "2", "-c", "0.8", "--max-seqs", "1",
               "--threads", str(args.threads), "--search-type", "3"]
    subprocess.run(command, check=True)
    homologous = set()
    if alignment.exists():
        with alignment.open() as handle:
            for line in handle:
                if line.strip() and not line.startswith("#"):
                    homologous.add(int(line.split("\t", 1)[0].rsplit("_", 1)[1]))
    train_sequences = set()
    with args.train_fasta.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                train_sequences.add(normalise(line.strip()))
    train_sha256 = sha256(args.train_fasta)
    audits = {}
    for dataset in datasets:
        name = dataset["name"]
        path = dataset["path"]
        rows = dataset["rows"]
        start = dataset["query_start"]
        exact = []
        kept = []
        split_homologous = set()
        for index, row in enumerate(rows):
            query_index = start + index
            if normalise(row["seq"]) in train_sequences:
                exact.append(index)
            elif query_index in homologous:
                split_homologous.add(index)
            else:
                kept.append(row)
        homology_only = sorted(split_homologous - set(exact))
        if len(rows) != len(kept) + len(exact) + len(homology_only):
            raise RuntimeError(f"attrition does not balance: {name}")
        output = args.out_dir / f"{name}.jsonl"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "train": str(args.train_fasta),
            "train_sha256": train_sha256,
            "split": str(path),
            "split_sha256": sha256(path),
            "protocol": "normalized exact identity plus MMseqs2 nucleotide identity>=0.8 and query coverage>=0.8",
            "command": command,
            "query_start": start,
            "raw_n": len(rows),
            "clean_n": len(kept),
            "removed_exact": len(exact),
            "removed_homology_only": len(homology_only),
            "removed_exact_indices": exact,
            "removed_homology_only_indices": homology_only,
            "clean_sha256": sha256(output),
        }
        atomic_json(args.out_dir / f"{name}.audit.json", audit)
        audits[name] = audit
    if set(audits) != set(names):
        raise RuntimeError("split attrition is incomplete")
    summary = {"status": "PASS_STRUCTRFM_PRETRAIN_CLEAN_ABC_13_SPLITS",
               "joint_query": str(query_fasta), "joint_query_sha256": sha256(query_fasta),
               "splits": audits}
    atomic_json(args.out_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "splits": {
        name: {key: audit[key] for key in ("raw_n", "clean_n", "removed_exact",
                                           "removed_homology_only")}
        for name, audit in audits.items()}}, indent=2))


if __name__ == "__main__":
    main()
