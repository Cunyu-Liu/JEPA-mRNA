#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalize(seq):
    return seq.upper().replace("T", "U")


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def structure_pairs(structure):
    stacks = {opening: [] for opening in "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    closing.update({chr(code + 32): chr(code) for code in range(ord("A"), ord("Z") + 1)})
    pairs = []
    for index, char in enumerate(structure):
        if char in stacks:
            stacks[char].append(index)
        elif char in closing:
            opening = closing[char]
            if not stacks[opening]:
                raise ValueError("unbalanced_structure")
            pairs.append((stacks[opening].pop(), index))
        elif char != ".":
            raise ValueError("unknown_structure_character")
    if any(stacks.values()):
        raise ValueError("unbalanced_structure")
    return set(pairs)


def validated(row):
    seq = normalize(row["seq"])
    if not seq or set(seq) - set("ACGU"):
        raise ValueError("invalid_alphabet")
    pairs = set()
    positions = set()
    for value in row["pairs"]:
        i, j = sorted(value)
        if not (0 <= i < j < len(seq)):
            raise ValueError("pair_out_of_range")
        if j - i <= 3:
            raise ValueError("hairpin_loop_below_3")
        if i in positions or j in positions:
            raise ValueError("nucleotide_in_multiple_pairs")
        positions.update((i, j))
        pairs.add((i, j))
    if row.get("structure") is not None:
        if len(row["structure"]) != len(seq) or structure_pairs(row["structure"]) != pairs:
            raise ValueError("structure_pair_mismatch")
    ordered = sorted(pairs)
    for index, (i, j) in enumerate(ordered):
        for k, l in ordered[index + 1:]:
            if k >= j:
                break
            if i < k < j < l:
                raise ValueError("crossing_pairs")
    return seq, pairs


def fasta(rows, path, prefix):
    with path.open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(f">{prefix}{index}\n{normalize(row['seq'])}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    source = load(args.source)
    base = load(args.base)
    evaluations = {"bprna_vl0": load(args.validation)}
    evaluations.update({split: load(args.split_dir / f"{split}.jsonl") for split in SPLITS})
    rejection_counts = collections.Counter()
    groups = {}
    for row in source:
        try:
            seq, pairs = validated(row)
        except ValueError as error:
            rejection_counts[str(error)] += 1
            continue
        groups.setdefault(seq, []).append((row, pairs))
    unique = []
    for seq, members in groups.items():
        pair_sets = {tuple(sorted(pairs)) for _, pairs in members}
        if len(pair_sets) != 1:
            rejection_counts["conflicting_duplicate_labels"] += len(members)
            continue
        rejection_counts["duplicate_same_labels"] += len(members) - 1
        unique.append(members[0][0])
    base_sequences = {normalize(row["seq"]) for row in base}
    evaluation_sequences = {split: {normalize(row["seq"]) for row in rows}
                            for split, rows in evaluations.items()}
    candidates = []
    removed_base = 0
    exact_by_split = {split: set() for split in evaluations}
    for row in unique:
        seq = normalize(row["seq"])
        if seq in base_sequences:
            removed_base += 1
            continue
        exact_splits = {split for split, sequences in evaluation_sequences.items() if seq in sequences}
        for split in exact_splits:
            exact_by_split[split].add(len(candidates))
        candidates.append(row)
    fasta(candidates, args.work / "candidates.fa", "candidate_")
    homology_by_split = {}
    commands = {}
    for split, rows in evaluations.items():
        directory = args.work / split
        directory.mkdir(exist_ok=True)
        query = directory / "candidates.fa"
        target = directory / "evaluation.fa"
        fasta(candidates, query, "candidate_")
        fasta(rows, target, "evaluation_")
        alignment = directory / "alignment.tsv"
        command = [str(args.mmseqs), "easy-search", str(query), str(target), str(alignment),
                   str(directory / "tmp"), "--min-seq-id", "0.8", "--cov-mode", "2", "-c", "0.8",
                   "--max-seqs", "1", "--threads", str(args.threads), "--search-type", "3"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        removed = set()
        if alignment.exists():
            for line in alignment.open():
                if line.strip():
                    removed.add(int(line.split("\t", 1)[0].removeprefix("candidate_")))
        homology_by_split[split] = removed
        commands[split] = command
        print(json.dumps({"split": split, "homology_hits": len(removed)}, ensure_ascii=False), flush=True)
    exact_union = set().union(*exact_by_split.values())
    homology_union = set().union(*homology_by_split.values())
    additions = [row for index, row in enumerate(candidates) if index not in exact_union | homology_union]
    assert len(source) == len(unique) + sum(rejection_counts.values())
    assert len(unique) == removed_base + len(candidates)
    assert len(candidates) == len(additions) + len(exact_union | homology_union)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in base + additions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    combined = load(args.out)
    assert len(combined) == len(base) + len(additions)
    assert len({normalize(row["seq"]) for row in combined}) == len(combined)
    for split, rows in evaluations.items():
        test_sequences = {normalize(row["seq"]) for row in rows}
        assert not any(normalize(row["seq"]) in test_sequences for row in additions)
    per_split = {}
    for split, rows in evaluations.items():
        removed_exact = len(exact_by_split[split])
        removed_homology = len(homology_by_split[split] - exact_by_split[split])
        per_split[split] = {"evaluation_n": len(rows), "removed_exact": removed_exact,
                            "removed_homology_only": removed_homology,
                            "candidate_n_before_filter": len(candidates),
                            "candidate_n_after_filter": len(candidates) - len(exact_by_split[split] | homology_by_split[split]),
                            "command": commands[split]}
    audit = {"schema_version": 1, "status": "PASS",
             "protocol": "noncrossing labels, hairpin loop>=3, exact dedup, MMseqs2 identity>=0.8 and query coverage>=0.8 against VL0 and all eight test splits",
             "source": str(args.source), "source_sha256": sha256(args.source),
             "base": str(args.base), "base_sha256": sha256(args.base), "base_n": len(base),
             "validation": str(args.validation), "validation_sha256": sha256(args.validation),
             "split_sha256": {split: sha256(args.split_dir / f"{split}.jsonl") for split in SPLITS},
             "source_raw_n": len(source), "source_unique_before_test_filter": len(unique),
             "source_rejections": dict(sorted(rejection_counts.items())),
             "removed_base_exact": removed_base, "candidate_n": len(candidates),
             "additions_n": len(additions), "combined_n": len(combined),
             "out": str(args.out), "out_sha256": sha256(args.out),
             "per_split": per_split}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: audit[key] for key in
                      ("status", "source_raw_n", "source_unique_before_test_filter", "removed_base_exact",
                       "candidate_n", "additions_n", "combined_n", "out_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
