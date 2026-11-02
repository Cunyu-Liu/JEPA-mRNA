#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise(seq):
    return seq.upper().replace("T", "U")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    args = parser.parse_args()
    train = [json.loads(line) for line in args.train.open() if line.strip()]
    split = [json.loads(line) for line in args.split.open() if line.strip()]
    assert train and split
    args.work_dir.mkdir(parents=True, exist_ok=True)
    train_fasta = args.work_dir / "train.fa"
    split_fasta = args.work_dir / "split.fa"
    with train_fasta.open("w") as handle:
        for index, row in enumerate(train):
            handle.write(f">train_{index}\n{normalise(row['seq'])}\n")
    with split_fasta.open("w") as handle:
        for index, row in enumerate(split):
            handle.write(f">split_{index}\n{normalise(row['seq'])}\n")
    alignment = args.work_dir / "alignment.tsv"
    tmp = args.work_dir / "tmp"
    command = [str(args.mmseqs), "easy-search", str(split_fasta), str(train_fasta),
               str(alignment), str(tmp), "--min-seq-id", "0.8", "--cov-mode", "2",
               "-c", "0.8", "--max-seqs", "1", "--threads", "2", "--search-type", "3"]
    subprocess.run(command, check=True)
    homologous = set()
    if alignment.exists():
        with alignment.open() as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                homologous.add(int(line.split("\t", 1)[0].rsplit("_", 1)[1]))
    train_seqs = {normalise(row["seq"]) for row in train}
    kept = []
    exact = []
    for index, row in enumerate(split):
        if normalise(row["seq"]) in train_seqs:
            exact.append(index)
        elif index not in homologous:
            kept.append(row)
    homology_only = sorted(homologous - set(exact))
    assert len(split) == len(kept) + len(exact) + len(homology_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema_version": 1, "status": "PASS",
        "train": str(args.train), "train_sha256": sha256(args.train),
        "split": str(args.split), "split_sha256": sha256(args.split),
        "protocol": "normalized exact identity plus MMseqs2 nucleotide hits with identity>=0.8 and query coverage>=0.8",
        "command": command, "raw_n": len(split), "clean_n": len(kept),
        "removed_exact": len(exact), "removed_homology_only": len(homology_only),
        "removed_exact_indices": exact, "removed_homology_only_indices": homology_only,
        "clean_sha256": sha256(args.out),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if not key.endswith("indices")}, indent=2))


if __name__ == "__main__":
    main()
