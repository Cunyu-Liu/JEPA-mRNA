import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(sequence):
    return sequence.upper().replace("T", "U")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fasta", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.split.open() if line.strip()]
    if not rows:
        raise ValueError("empty split")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    split_fasta = args.work_dir / "split.fa"
    with split_fasta.open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(f">split_{index}\n{normalise(row['seq'])}\n")
    alignment = args.work_dir / "alignment.tsv"
    mmseqs_tmp = tempfile.mkdtemp(prefix="structrfm_mmseqs_")
    command = [str(args.mmseqs), "easy-search", str(split_fasta), str(args.train_fasta),
               str(alignment), mmseqs_tmp, "--min-seq-id", "0.8",
               "--cov-mode", "2", "-c", "0.8", "--max-seqs", "1",
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
    kept = []
    exact = []
    for index, row in enumerate(rows):
        if normalise(row["seq"]) in train_sequences:
            exact.append(index)
        elif index not in homologous:
            kept.append(row)
    homology_only = sorted(homologous - set(exact))
    if len(rows) != len(kept) + len(exact) + len(homology_only):
        raise RuntimeError("attrition does not balance")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "train": str(args.train_fasta),
        "train_sha256": sha256(args.train_fasta),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "protocol": "normalized exact identity plus MMseqs2 nucleotide identity>=0.8 and query coverage>=0.8",
        "command": command,
        "raw_n": len(rows),
        "clean_n": len(kept),
        "removed_exact": len(exact),
        "removed_homology_only": len(homology_only),
        "removed_exact_indices": exact,
        "removed_homology_only_indices": homology_only,
        "clean_sha256": sha256(args.out),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if not key.endswith("indices")}))


if __name__ == "__main__":
    main()
