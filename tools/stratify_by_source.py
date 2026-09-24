"""Stratify an evaluation result by the source database recorded in the sequence name.

Why this is a tool and not a one-off
------------------------------------
``spec/benchmark_decision.md`` 4.2b makes source-stratified reporting **mandatory**:
the pooled F1 of this model is misleading in both directions, because the two large
bpRNA-1m source populations differ by roughly 0.4 F1 and appear in very different
proportions in different splits (``CRW`` is 6.0% of TR0, 7.3% of TS0, 50.5% of VL0).
The measured consequence is that the pooled TS0 number was read as "we are below the
physical baselines", when on the ``CRW`` stratum we are 0.29 *above* ViennaRNA
centroid.  See ``records/DECISION_TRAINING_LOG.md`` 14.21.

``CRW`` and ``RFAM`` are **source-database labels, not verified family labels.**
Sequence names are unique within each split (measured: TS0's 1,288 names reduce to
1,279 distinct numeric ids), so no family field is recoverable from the names.  This
stratification is a reproducible proxy that correlates with molecule type and
structural conservation.  It is not a family-level split.

Pooling is exact, not an average of ratios
------------------------------------------
micro F1 is recomputed per stratum by pooling TP/FP/FN.  TP is recovered exactly from
the evaluator's own record as ``precision * n_pred_pairs``, so the strata sum back to
the pooled number instead of drifting from it.

Usage
-----
    python tools/stratify_by_source.py \
        --eval-root /mnt/cunyuliu/rna-jepa/eval_decision \
        --data-dir  /mnt/cunyuliu/rna-jepa/ss_data/jsonl \
        --run astrained_ff3500_bprna_ts0 --run sel_ff3500_bprna_ts0 \
        --max-length 100
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


#: The source tokens bpRNA-1m actually uses in ``bpRNA_<source>_<id>``.  A token
#: outside this set is NOT a source.  This matters: bpRNA-new names are GenBank
#: accessions such as ``NC_000913.3_1234-1300.bpseq``, which contain underscores, so
#: naively taking the second ``_``-separated token yields ``000913.3`` and the tool
#: would happily print ~3,000 one-sequence "strata" as if they meant something.
#: The vocabulary is closed on purpose; anything else is ``OTHER``.
KNOWN_SOURCES = frozenset({"CRW", "RFAM", "SPR", "SRP", "RNP", "tmRNA"})


def source_of(name: str) -> str:
    """``bpRNA_CRW_15573.bpseq`` -> ``CRW``; anything unrecognised -> ``OTHER``.

    Returns ``OTHER`` unless the name is literally ``bpRNA_<known source>_<id>``.
    Splits whose names do not follow that convention (bpRNA-new uses GenBank
    accessions) therefore come back as all-``OTHER``, and the caller reports the
    stratification as unavailable rather than presenting it as informative.
    """
    parts = name.split("_")
    if len(parts) > 2 and parts[0] == "bpRNA" and parts[1] in KNOWN_SOURCES:
        return parts[1]
    return "OTHER"


def tp_of(rec: dict) -> float:
    return float(rec["precision"]) * float(rec["n_pred_pairs"])


def micro_f1(records: Sequence[dict]) -> Tuple[float, int]:
    """Pooled micro F1 and the number of sequences it was pooled over."""
    if not records:
        return float("nan"), 0
    tp = sum(tp_of(r) for r in records)
    pred = sum(float(r["n_pred_pairs"]) for r in records)
    gt = sum(float(r["n_gt_pairs"]) for r in records)
    denom = pred + gt
    return (2.0 * tp / denom) if denom > 0 else float("nan"), len(records)


def analyse(eval_root: str, data_dir: str, run: str,
            max_length: Optional[int]) -> Dict[str, object]:
    result_path = os.path.join(eval_root, run, "result.json")
    if not os.path.isfile(result_path):
        return {"run": run, "error": f"missing {result_path}"}
    with open(result_path, encoding="utf-8") as fh:
        result = json.load(fh)

    split_file = os.path.basename(str(result["data"]))
    split_path = os.path.join(data_dir, split_file)
    if not os.path.isfile(split_path):
        return {"run": run, "error": f"missing split file {split_path}"}

    by_name = {r["name"]: r for r in read_jsonl(split_path)}
    per_seq = result["per_sequence"]

    pairs: List[Tuple[dict, str]] = []
    missing = 0
    for rec in per_seq:
        row = by_name.get(rec["name"])
        if row is None:
            missing += 1
            continue
        if max_length is not None and len(row["seq"]) > max_length:
            continue
        pairs.append((rec, source_of(rec["name"])))

    overall, n_overall = micro_f1([rec for rec, _ in pairs])
    strata: Dict[str, List[dict]] = collections.defaultdict(list)
    for rec, src in pairs:
        strata[src].append(rec)

    sources = sorted(strata)
    table = []
    for src in sources:
        f1, n = micro_f1(strata[src])
        table.append({"source": src, "n": n, "micro_f1": f1})

    return {
        "run": run,
        "split": split_file,
        "checkpoint": os.path.basename(str(result.get("checkpoint", "?"))),
        "prior_weight": result.get("prior_weight"),
        "prior_weight_effective": result.get("prior_weight_effective"),
        "max_length": max_length,
        "n_missing_from_split": missing,
        "overall_micro_f1": overall,
        "overall_n": n_overall,
        "strata": table,
        # True when the split carries no recognisable source labels, i.e. the
        # stratification is unavailable and must be reported as such.
        "stratification_unavailable": sources in ([], ["OTHER"]),
    }


def print_report(reports: Sequence[Dict[str, object]]) -> None:
    for rep in reports:
        if "error" in rep:
            print(f"!! {rep['run']}: {rep['error']}")
            continue
        lim = rep["max_length"]
        lim_txt = f"<= {lim} nt" if lim is not None else "no length limit"
        print(f"--- {rep['run']}   [{rep['split']}, {lim_txt}]")
        print(f"    checkpoint={rep['checkpoint']}  prior_weight={rep['prior_weight']} "
              f"(effective {rep['prior_weight_effective']})")
        print(f"    pooled micro F1 = {rep['overall_micro_f1']:.4f}  (n={rep['overall_n']})")
        if rep["stratification_unavailable"]:
            print("    STRATIFICATION UNAVAILABLE: this split carries no source labels "
                  "(names are not bpRNA_<source>_<id>).  Report that, do not treat "
                  "'OTHER' as a stratum.")
            print()
            continue
        for row in rep["strata"]:
            print(f"      {row['source']:>6s}  n={row['n']:>5d}  "
                  f"micro F1={row['micro_f1']:.4f}")
        print()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run", action="append", required=True,
                   help="evaluation output directory name (repeatable)")
    p.add_argument("--max-length", type=int, default=0,
                   help="0 means no length limit")
    p.add_argument("--json", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    max_length = args.max_length or None
    reports = [analyse(args.eval_root, args.data_dir, run, max_length)
               for run in args.run]
    print_report(reports)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"max_length": max_length, "reports": reports},
                      fh, indent=1, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
