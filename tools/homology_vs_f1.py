"""Is our test-split F1 actually a function of homology to the training set?

Motivation
----------
Two anomalies are currently unexplained and both block claims:

1. **§14.15 problem 2.**  The same checkpoint scores micro F1 0.9230 on bpRNA VL0 but
   0.6188 on bpRNA TS0 *within the same length bucket* (<=100 nt).  Exact overlap is
   zero, and length / pair density / family labels were matched.  A 0.30 gap with no
   identified cause.

2. **§14.18.**  On bpRNA-new the model collapses from 0.4953 (TS0) to 0.3094, which is
   indistinguishable from our own Nussinov+Turner prior (0.3015), while ViennaRNA
   centroid *rises* from 0.5393 to 0.6770.

Both become the *same* observation if F1 tracks sequence-level homology to the training
split: VL0 may simply be more homologous to TR0 than TS0 is, and bpRNA-new is
homology-free by construction.  That is a single testable mechanism instead of two
separate mysteries, so it is worth measuring before writing either off.

What is measured
----------------
For each test split, every sequence gets a **k-mer containment** score

    containment(x) = |{ k-mers of x } ∩ { k-mers of TR0 }| / |{ k-mers of x }|

and sequences are then binned by that score.  Micro F1 is recomputed *within each bin*
by pooling TP/FP/FN, using the per-sequence records already written by the evaluator
(``precision`` and ``n_pred_pairs`` give TP exactly, so pooling is exact rather than an
average of ratios).

k = 20 by default.  k must be large enough that a chance match is not automatic: with
k = 8 an RNA alphabet saturates and containment stops discriminating (the project has
already been bitten by that).  Only k >= 15 should be read.

Usage
-----
    python tools/homology_vs_f1.py \
        --train /mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl \
        --eval-root /mnt/cunyuliu/rna-jepa/eval_decision \
        --pair sel_ff3500_bprna_ts0 \
        --pair sel_ff3500_bprna_vl0 \
        --pair sel_ff3500_bprna_new \
        --k 20

Every pair is ``<eval output directory>``; the split file name is read from the
evaluator's own record so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

#: Bin edges on k-mer containment.  The last bin is closed on the right.
BIN_EDGES: Tuple[float, ...] = (0.05, 0.2, 0.5, 0.8)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def kmers(seq: str, k: int) -> Iterable[str]:
    seq = seq.upper().replace("T", "U")
    for i in range(len(seq) - k + 1):
        yield seq[i:i + k]


def build_kmer_set(sequences: Sequence[str], k: int) -> set:
    table = set()
    for seq in sequences:
        table.update(kmers(seq, k))
    return table


def containment(seq: str, table: set, k: int) -> Tuple[float, int]:
    """Fraction of ``seq``'s k-mers that occur in ``table``, plus the k-mer count.

    A sequence shorter than ``k`` has no k-mers and therefore no defined containment;
    it is reported with ``n_kmers == 0`` and excluded from the bins by the caller
    rather than silently scored as 0.0 or 1.0.
    """
    ks = list(kmers(seq, k))
    if not ks:
        return float("nan"), 0
    hit = sum(1 for x in ks if x in table)
    return hit / len(ks), len(ks)


def bin_index(value: float) -> int:
    for i, edge in enumerate(BIN_EDGES):
        if value < edge:
            return i
    return len(BIN_EDGES)


def bin_label(i: int) -> str:
    if i == 0:
        return f"<{BIN_EDGES[0]:.2f}"
    if i == len(BIN_EDGES):
        return f">={BIN_EDGES[-1]:.2f}"
    return f"{BIN_EDGES[i - 1]:.2f}-{BIN_EDGES[i]:.2f}"


def tp_of(rec: dict) -> float:
    """True positives, recovered exactly from the evaluator's own record."""
    return float(rec["precision"]) * float(rec["n_pred_pairs"])


def micro_f1(records: Sequence[dict]) -> float:
    tp = sum(tp_of(r) for r in records)
    pred = sum(float(r["n_pred_pairs"]) for r in records)
    gt = sum(float(r["n_gt_pairs"]) for r in records)
    denom = pred + gt
    return (2.0 * tp / denom) if denom > 0 else float("nan")


def analyse(eval_root: str, tag: str, train_table: set, train_seqs: set,
            data_root: str, k: int) -> Dict[str, object]:
    out_dir = os.path.join(eval_root, tag)
    result_path = os.path.join(out_dir, "result.json")
    if not os.path.isfile(result_path):
        return {"tag": tag, "error": f"missing {result_path}"}
    with open(result_path, encoding="utf-8") as fh:
        result = json.load(fh)

    split_file = os.path.basename(result["data"])
    rows = read_jsonl(os.path.join(data_root, split_file))
    by_name = {r["name"]: r for r in rows}
    per_seq = result["per_sequence"]

    records: List[Tuple[dict, float]] = []
    n_no_kmer = 0
    n_missing = 0
    for rec in per_seq:
        row = by_name.get(rec["name"])
        if row is None:
            n_missing += 1
            continue
        c, nk = containment(row["seq"], train_table, k)
        if nk == 0:
            n_no_kmer += 1
            continue
        records.append((rec, c))

    exact = sum(1 for rec, _ in records
                if by_name[rec["name"]]["seq"].upper().replace("T", "U") in train_seqs)

    buckets: Dict[int, List[dict]] = {}
    for rec, c in records:
        buckets.setdefault(bin_index(c), []).append(rec)

    table = []
    for i in sorted(buckets):
        group = buckets[i]
        conts = [c for rec, c in records if bin_index(c) == i]
        table.append({
            "bin": bin_label(i),
            "n": len(group),
            "mean_containment": sum(conts) / len(conts) if conts else float("nan"),
            "micro_f1": micro_f1(group),
        })

    all_conts = [c for _, c in records]
    return {
        "tag": tag,
        "checkpoint": os.path.basename(str(result.get("checkpoint", "?"))),
        "prior_weight": result.get("prior_weight"),
        "prior_weight_effective": result.get("prior_weight_effective"),
        "split": split_file,
        "n_scored": len(records),
        "n_no_kmer": n_no_kmer,
        "n_name_not_in_split": n_missing,
        "n_exact_train_match": exact,
        "containment_mean": sum(all_conts) / len(all_conts) if all_conts else float("nan"),
        "containment_median": sorted(all_conts)[len(all_conts) // 2] if all_conts else float("nan"),
        "micro_f1_all": micro_f1([r for r, _ in records]),
        "bins": table,
    }


def print_report(reports: Sequence[Dict[str, object]], k: int) -> None:
    print(f"k-mer containment vs micro F1   (k={k})")
    print()
    for rep in reports:
        if "error" in rep:
            print(f"!! {rep['tag']}: {rep['error']}")
            continue
        print(f"--- {rep['tag']}  [{rep['split']}]")
        print(f"    checkpoint={rep['checkpoint']}  "
              f"prior_weight={rep['prior_weight']} "
              f"(effective {rep['prior_weight_effective']})")
        print(f"    scored={rep['n_scored']}  no_kmer={rep['n_no_kmer']}  "
              f"name_not_in_split={rep['n_name_not_in_split']}  "
              f"exact_train_match={rep['n_exact_train_match']}")
        print(f"    containment mean={rep['containment_mean']:.4f} "
              f"median={rep['containment_median']:.4f}   "
              f"micro F1 (all)={rep['micro_f1_all']:.4f}")
        print(f"    {'bin':>10s} {'n':>6s} {'mean_cont':>10s} {'microF1':>8s}")
        for row in rep["bins"]:
            print(f"    {row['bin']:>10s} {row['n']:>6d} "
                  f"{row['mean_containment']:>10.4f} {row['micro_f1']:>8.4f}")
        print()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="training split jsonl (TR0)")
    p.add_argument("--eval-root", required=True, help="directory holding eval outputs")
    p.add_argument("--data-root", default="", help="directory holding the split jsonl "
                                                   "files (default: same as --train)")
    p.add_argument("--pair", action="append", required=True,
                   help="an evaluation output directory name (repeatable)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--json", default="", help="write the full report here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.k < 15:
        print(f"WARNING: k={args.k} < 15; short k-mers saturate and containment stops "
              f"discriminating.  Read the numbers with that in mind.")
    train_rows = read_jsonl(args.train)
    train_seqs = {r["seq"].upper().replace("T", "U") for r in train_rows}
    table = build_kmer_set([r["seq"] for r in train_rows], args.k)
    print(f"training split: {args.train}  n={len(train_rows)}  "
          f"distinct {args.k}-mers={len(table)}")
    print()

    data_root = args.data_root or os.path.dirname(os.path.abspath(args.train))
    reports = [analyse(args.eval_root, tag, table, train_seqs, data_root, args.k)
               for tag in args.pair]
    print_report(reports, args.k)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"k": args.k, "train": args.train, "reports": reports},
                      fh, indent=1, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
