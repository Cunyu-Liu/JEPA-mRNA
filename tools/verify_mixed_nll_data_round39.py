#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def normalize(seq):
    return seq.upper().replace("T", "U")


def parse_structure(structure):
    opening = "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    closing = ")]}>abcdefghijklmnopqrstuvwxyz"
    partner = dict(zip(opening, closing))
    reverse = {value: key for key, value in partner.items()}
    stacks = {key: [] for key in opening}
    pairs = set()
    for index, char in enumerate(structure):
        if char in stacks:
            stacks[char].append(index)
        elif char in reverse:
            key = reverse[char]
            if not stacks[key]:
                raise ValueError("unbalanced_structure")
            pairs.add((stacks[key].pop(), index))
        elif char != ".":
            raise ValueError("unknown_structure_character")
    if any(stacks.values()):
        raise ValueError("unbalanced_structure")
    return pairs


def validate_addition(row):
    seq = normalize(row["seq"])
    assert seq and not set(seq) - set("ACGU")
    pairs = set()
    occupied = set()
    for i, j in row["pairs"]:
        i, j = sorted((i, j))
        assert 0 <= i < j < len(seq) and j - i > 3
        assert i not in occupied and j not in occupied
        occupied.update((i, j))
        pairs.add((i, j))
    assert len(row["structure"]) == len(seq)
    assert parse_structure(row["structure"]) == pairs
    for i, j in pairs:
        assert not any(i < k < j < l or k < i < l < j for k, l in pairs)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit["status"] == "PASS" and audit["out_sha256"] == sha256(args.data)
    assert audit["source_sha256"] == sha256(Path(audit["source"]))
    assert audit["base_sha256"] == sha256(args.base)
    assert audit["validation_sha256"] == sha256(args.validation)
    base = load(args.base)
    combined = load(args.data)
    assert len(base) == audit["base_n"] and len(combined) == audit["combined_n"]
    assert combined[:len(base)] == base
    additions = combined[len(base):]
    assert len(additions) == audit["additions_n"]
    addition_sequences = [validate_addition(row) for row in additions]
    assert len(set(addition_sequences)) == len(additions)
    assert not set(addition_sequences) & {normalize(row["seq"]) for row in base}
    evaluations = {"bprna_vl0": load(args.validation)}
    for split in audit["split_sha256"]:
        path = args.split_dir / f"{split}.jsonl"
        assert sha256(path) == audit["split_sha256"][split]
        evaluations[split] = load(path)
    args.work.mkdir(parents=True, exist_ok=True)
    fasta = args.work / "additions.fa"
    with fasta.open("w") as handle:
        for index, seq in enumerate(addition_sequences):
            handle.write(f">addition_{index}\n{seq}\n")
    reports = {}
    for split, rows in evaluations.items():
        assert not set(addition_sequences) & {normalize(row["seq"]) for row in rows}
        directory = args.work / split
        directory.mkdir(exist_ok=True)
        target = directory / "evaluation.fa"
        with target.open("w") as handle:
            for index, row in enumerate(rows):
                handle.write(f">evaluation_{index}\n{normalize(row['seq'])}\n")
        alignment = directory / "alignment.tsv"
        command = [str(args.mmseqs), "easy-search", str(fasta), str(target), str(alignment),
                   str(directory / "tmp"), "--min-seq-id", "0.8", "--cov-mode", "2", "-c", "0.8",
                   "--max-seqs", "1", "--threads", "2", "--search-type", "3"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        hits = [line for line in alignment.open() if line.strip()] if alignment.exists() else []
        assert not hits, (split, hits[:3])
        reports[split] = {"additions_n": len(additions), "evaluation_n": len(rows),
                          "exact_overlap": 0, "homology_hits": 0, "command": command}
    payload = {"schema_version": 1, "status": "PASS_INDEPENDENT_LABEL_AND_LEAKAGE_RECHECK",
               "audit": str(args.audit), "audit_sha256": sha256(args.audit),
               "data": str(args.data), "data_sha256": sha256(args.data),
               "base_prefix_exact": True, "additions_n": len(additions),
               "labels_valid_noncrossing": True, "per_split": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "additions_n": len(additions),
                      "splits": len(reports), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
