"""Remove test-split rows that duplicate or closely match the training split.

Why this exists
---------------
`spec/benchmark_decision.md` 4.4 requires that "test set is same-source as training"
be avoided by identity filtering, and the checklist has carried "ArchiveII 去冗余"
as an open item for a long time.  Measuring it turned out to matter a lot: ArchiveII
shares **812 of 3,950 rows (20.6%) with bpRNA TR0 by exact sequence identity**, and
24.8% of the <=100 nt bucket.  The examples are unambiguous -- ArchiveII's
`5s_Acholeplasma-laidlawii-2.bpseq` is byte-identical to TR0's `bpRNA_RFAM_654.bpseq`.

Any F1 measured on the raw split is therefore partly a memorisation score, and the
"we beat ViennaRNA centroid on ArchiveII" comparison is not reportable until the
duplicates are gone.  TS0, VL0 and bpRNA-new were checked the same way and are clean
(exact matches 0; mean 20-mer containment 0.040, 0.042 and 0.0001).

Two filters, reported separately
--------------------------------
1. **Exact identity** against the training split (after ``T`` -> ``U`` and upper-casing).
2. **k-mer containment** above a threshold, which catches near-duplicates that exact
   matching misses.  ``containment(x) = |kmers(x) & kmers(train)| / |kmers(x)|``.

The attrition table obeys ``total = kept + removed_exact + removed_homology``, and the
script asserts that identity so a silent loss cannot pass unnoticed.

Threshold choice is reported, not hidden: 0.5 is the default because it is the loosest
defensible cut, and the script also prints the kept counts at stricter thresholds so a
reader can see how sensitive the result is to it.

Usage
-----
    python tools/dedup_against_train.py \
        --train  /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl \
        --split  /mnt/cunyuliu/rna-jepa/ss_data/jsonl/archiveii_embok.jsonl \
        --out    /mnt/cunyuliu/rna-jepa/ss_data/jsonl/archiveii_clean.jsonl \
        --threshold 0.5 --k 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalise(seq: str) -> str:
    return seq.upper().replace("T", "U")


def kmers(seq: str, k: int) -> Iterable[str]:
    s = normalise(seq)
    for i in range(len(s) - k + 1):
        yield s[i:i + k]


def build_table(sequences: Sequence[str], k: int) -> set:
    table = set()
    for seq in sequences:
        table.update(kmers(seq, k))
    return table


def containment(seq: str, table: set, k: int) -> float:
    ks = list(kmers(seq, k))
    if not ks:
        return 0.0          # a sequence shorter than k has no evidence of homology
    return sum(1 for x in ks if x in table) / len(ks)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", default="", help="write the de-duplicated subset here")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="drop rows with k-mer containment above this (default 0.5)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--json", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    train = read_jsonl(args.train)
    split = read_jsonl(args.split)

    train_seqs = {normalise(r["seq"]) for r in train}
    table = build_table([r["seq"] for r in train], args.k)
    print(f"train {os.path.basename(args.train)}: {len(train)} sequences, "
          f"{len(table)} distinct {args.k}-mers")
    print(f"split {os.path.basename(args.split)}: {len(split)} sequences")
    print()

    kept: List[dict] = []
    removed_exact = 0
    removed_homology = 0
    conts: List[float] = []
    for row in split:
        if normalise(row["seq"]) in train_seqs:
            removed_exact += 1
            continue
        c = containment(row["seq"], table, args.k)
        conts.append(c)
        if c > args.threshold:
            removed_homology += 1
            continue
        kept.append(row)

    total = len(split)
    if total != len(kept) + removed_exact + removed_homology:
        raise AssertionError("attrition does not balance: "
                             f"{total} != {len(kept)} + {removed_exact} + {removed_homology}")

    conts.sort()
    n = len(conts)
    print(f"containment over the non-exact rows (n={n}): "
          f"mean={statistics.mean(conts):.4f} median={conts[n // 2]:.4f} "
          f"p90={conts[int(n * 0.9)]:.4f} max={conts[-1]:.4f}")
    print()
    print("attrition (balances by construction):")
    print(f"  total                      {total:>6d}")
    print(f"  - removed, exact identity  {removed_exact:>6d}  "
          f"({removed_exact / total:.1%})")
    print(f"  - removed, containment > {args.threshold:<4} {removed_homology:>6d}  "
          f"({removed_homology / total:.1%})")
    print(f"  = kept                     {len(kept):>6d}  ({len(kept) / total:.1%})")
    print()
    print("sensitivity to the threshold (rows kept):")
    for t in (0.1, 0.2, 0.3, 0.5, 0.8):
        k_ = sum(1 for c in conts if c <= t)
        print(f"  containment <= {t:<4} -> {k_:>6d} kept "
              f"({k_ / total:.1%} of the split)")
    print("  (these counts exclude the exact matches, which are always removed)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row) + "\n")
        print(f"\nwrote {args.out} ({len(kept)} sequences)")

    if args.json:
        payload = {
            "train": args.train, "split": args.split, "k": args.k,
            "threshold": args.threshold, "total": total,
            "removed_exact": removed_exact, "removed_homology": removed_homology,
            "kept": len(kept),
            "containment": {
                "mean": statistics.mean(conts), "median": conts[n // 2],
                "p90": conts[int(n * 0.9)], "max": conts[-1],
            },
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
