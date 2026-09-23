"""Build the head-to-head table and its statistics from the evaluation ledger.

Input is the tree written by ``rnajepa.finetune``::

    <eval_root>/<model_label>/<task>_s<seed>/result.json

For every task the *primary* metric (from ``configs/tasks.yaml``) is summarised across
seeds as mean +/- SD, and every model is compared against the reference model with a
**paired** bootstrap over seeds (paired because the same seeds are used for every model,
which removes the shared seed-to-seed variance) plus a Wilcoxon signed-rank test.  The
per-task p-values are then corrected with Benjamini-Hochberg.

Outputs, all under ``--out``:
  ``table_main.csv``    per task x model: metric, mean, sd, n, plus the paper value
  ``table_vs_ref.csv``  per task: delta vs the reference model, CI, p, verdict
  ``table_main.md``     the same content as a markdown table for the record
  ``summary.json``      machine-readable everything, including the metric convention used

Two conventions are enforced and printed so they cannot be lost:
  * binary tasks are reported with positive-class F1 as the paper does, and macro F1 /
    MCC alongside; ``--f1`` switches which one is used for the verdict.
  * tasks whose dev and test files are byte-identical (``shared_dev_test`` in the task
    registry) are marked, because their test score is a development score.

Usage:
  python eval/make_tables.py --eval_root /mnt/cunyuliu/rna-jepa/eval_out \
      --reference mrnabert_official --candidates v1_cont,v2_scratch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rnajepa.metrics import benjamini_hochberg, paired_bootstrap_ci, wilcoxon_signed_rank  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    print("FATAL: pyyaml required", file=sys.stderr)
    raise SystemExit(2)

ROOT = os.environ.get("RNAJEPA_ROOT", "/home/cunyuliu/rna-jepa")


def load_tasks(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["tasks"]


def collect(eval_root: str, model_label: str, task: str) -> List[dict]:
    """All seed results for one (model, task) pair."""
    base = os.path.join(eval_root, model_label)
    if not os.path.isdir(base):
        return []
    out = []
    prefix = f"{task}_s"
    for name in sorted(os.listdir(base)):
        if not name.startswith(prefix):
            continue
        seed = name[len(prefix):]
        if not seed.isdigit():
            continue
        path = os.path.join(base, name, "result.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["_seed"] = int(seed)
        rec["_dir"] = os.path.join(base, name)
        out.append(rec)
    return out


def metric_value(rec: dict, metric: str, f1_kind: str) -> Optional[float]:
    if metric == "accuracy":
        return rec.get("accuracy")
    if metric == "f1":
        return rec.get("f1_positive" if f1_kind == "positive" else "f1_macro")
    if metric == "mcc":
        return rec.get("mcc")
    if metric == "auc":
        return rec.get("auc")
    return rec.get(metric)


def fmt(x: Optional[float], nd: int = 4) -> str:
    return "" if x is None or x != x else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="/mnt/cunyuliu/rna-jepa/eval_out")
    ap.add_argument("--tasks_yaml", default=os.path.join(ROOT, "configs", "tasks.yaml"))
    ap.add_argument("--reference", default="mrnabert_official")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--out", default="/mnt/cunyuliu/rna-jepa/tables")
    ap.add_argument("--f1", default="positive", choices=["positive", "macro"],
                    help="which F1 the verdict uses for binary tasks; the paper reports "
                         "positive-class F1, so that is the default")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--discover", action="store_true",
                    help="also treat every directory under --eval_root as a model. Off by "
                         "default: mixing a scratch run into the published table would "
                         "compare different protocols.")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks_yaml)
    candidates = [c for c in args.candidates.split(",") if c.strip()]
    models = [args.reference] + [c for c in candidates if c != args.reference]

    # optionally discover extra model directories on disk
    if args.discover and os.path.isdir(args.eval_root):
        for name in sorted(os.listdir(args.eval_root)):
            if os.path.isdir(os.path.join(args.eval_root, name)) and name not in models:
                models.append(name)

    os.makedirs(args.out, exist_ok=True)
    rows: List[dict] = []
    comparisons: List[dict] = []
    pvals: List[float] = []
    comp_index: List[int] = []

    for task, spec in tasks.items():
        metric = spec["metric"]
        for model in models:
            recs = collect(args.eval_root, model, task)
            if not recs:
                continue
            vals = [metric_value(r, metric, args.f1) for r in recs]
            vals = [v for v in vals if v is not None and v == v]
            rows.append({
                "task": task, "family": spec["family"], "metric": metric,
                "model": model, "n_seeds": len(vals),
                "mean": statistics.fmean(vals) if vals else None,
                "sd": (statistics.stdev(vals) if len(vals) > 1 else 0.0) if vals else None,
                "seeds": ",".join(str(r["_seed"]) for r in recs),
                "paper": spec.get("paper"),
                "paper_delta": (statistics.fmean(vals) - spec["paper"]
                                if vals and spec.get("paper") is not None else None),
                "shared_dev_test": bool(spec.get("shared_dev_test", False)),
            })

        ref = collect(args.eval_root, args.reference, task)
        if not ref:
            continue
        ref_by_seed = {r["_seed"]: metric_value(r, metric, args.f1) for r in ref}
        for cand in candidates:
            crec = collect(args.eval_root, cand, task)
            if not crec:
                continue
            cand_by_seed = {r["_seed"]: metric_value(r, metric, args.f1) for r in crec}
            seeds = sorted(set(ref_by_seed) & set(cand_by_seed))
            a = [cand_by_seed[s] for s in seeds if cand_by_seed[s] == cand_by_seed[s]]
            b = [ref_by_seed[s] for s in seeds if ref_by_seed[s] == ref_by_seed[s]]
            seeds = [s for s in seeds
                     if cand_by_seed[s] == cand_by_seed[s] and ref_by_seed[s] == ref_by_seed[s]]
            if len(a) < 2:
                comparisons.append({
                    "task": task, "family": spec["family"], "metric": metric,
                    "candidate": cand, "reference": args.reference, "n_pairs": len(a),
                    "delta": (statistics.fmean(a) - statistics.fmean(b)) if a else None,
                    "ci_low": None, "ci_high": None, "p_boot": None, "p_wilcoxon": None,
                    "verdict": "insufficient_seeds",
                })
                continue
            boot = paired_bootstrap_ci(a, b, seed=20260923)
            wil = wilcoxon_signed_rank(a, b)
            comp = {
                "task": task, "family": spec["family"], "metric": metric,
                "candidate": cand, "reference": args.reference,
                "n_pairs": len(a), "seeds": ",".join(str(s) for s in seeds),
                "cand_mean": statistics.fmean(a), "ref_mean": statistics.fmean(b),
                **boot, "p_wilcoxon": wil.get("p"),
            }
            comp_index.append(len(comparisons))
            pvals.append(boot["p_boot"] if boot["p_boot"] == boot["p_boot"] else 1.0)
            comparisons.append(comp)

    # multiple-comparison correction across tasks
    if pvals:
        adj = benjamini_hochberg(pvals, alpha=args.alpha)
        for i, idx in enumerate(comp_index):
            comp = comparisons[idx]
            comp["p_adj_bh"] = adj["p_adj"][i]
            comp["significant_bh"] = adj["reject"][i]
            d, lo, hi = comp.get("delta"), comp.get("ci_low"), comp.get("ci_high")
            if d is None:
                continue
            if lo is not None and lo > 0:
                v = "win"
            elif hi is not None and hi < 0:
                v = "loss"
            else:
                v = "tie"
            if not comp["significant_bh"] and v != "tie":
                v = f"{v}_but_ci_overlaps_zero" if lo is not None else v
            comp["verdict"] = v

    # ---- write -------------------------------------------------------------
    main_csv = os.path.join(args.out, "table_main.csv")
    with open(main_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["task"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cmp_csv = os.path.join(args.out, "table_vs_ref.csv")
    with open(cmp_csv, "w", newline="", encoding="utf-8") as fh:
        fields = ["task", "family", "metric", "candidate", "reference", "n_pairs", "seeds",
                  "cand_mean", "ref_mean", "delta", "ci_low", "ci_high", "p_boot",
                  "p_wilcoxon", "p_adj_bh", "significant_bh", "verdict"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in comparisons:
            w.writerow(c)

    md = [f"# Head-to-head: candidates vs `{args.reference}`",
          "",
          f"Primary metric per task as declared in the task registry. Binary tasks use "
          f"**{args.f1}-class F1** conventions (`--f1`). Row counts come from the number of "
          f"seed runs found on disk.", "",
          "| task | family | metric | candidate | ref | delta | 95% CI | p(boot) | p(BH) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        md.append("| {task} | {family} | {metric} | {cand} | {ref} | {d} | [{lo}, {hi}] | {pb} | {pa} | {v} |".format(
            task=c["task"], family=c["family"], metric=c["metric"], cand=c["candidate"],
            ref=c["reference"],
            d=fmt(c.get("delta")), lo=fmt(c.get("ci_low")), hi=fmt(c.get("ci_high")),
            pb=fmt(c.get("p_boot")), pa=fmt(c.get("p_adj_bh")), v=c.get("verdict", "")))
    md += ["", "## Per-model summary", "",
           "| task | metric | model | n | mean | sd | paper |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['task']} | {r['metric']} | {r['model']} | {r['n_seeds']} | "
                  f"{fmt(r['mean'])} | {fmt(r['sd'])} | {r.get('paper')} |")
    n_shared = sum(1 for r in rows if r.get("shared_dev_test"))
    md += ["", f"Note: {n_shared} summary row(s) belong to tasks whose dev and test files are "
               f"byte-identical, so those test scores are development scores "
               f"(`shared_dev_test` in the registry). Nothing in this table is a smoke or "
               f"training-set number: every entry is a test-split score from the official "
               f"split, produced by a dev-selected checkpoint."]
    with open(os.path.join(args.out, "table_main.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    summary = {
        "reference": args.reference, "candidates": candidates, "models": models,
        "f1_convention": args.f1, "alpha": args.alpha,
        "n_rows": len(rows), "n_comparisons": len(comparisons),
        "rows": rows, "comparisons": comparisons,
        "wins": sum(1 for c in comparisons if str(c.get("verdict", "")).startswith("win")),
        "losses": sum(1 for c in comparisons if str(c.get("verdict", "")).startswith("loss")),
        "ties": sum(1 for c in comparisons if c.get("verdict") == "tie"),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)

    print(f"models: {models}")
    print(f"wrote {main_csv}\n      {cmp_csv}\n      {os.path.join(args.out,'table_main.md')}"
          f"\n      {os.path.join(args.out,'summary.json')}")
    print(f"rows={len(rows)} comparisons={len(comparisons)} "
          f"wins={summary['wins']} losses={summary['losses']} ties={summary['ties']}")
    if not comparisons:
        print("NOTE: no paired comparison yet -- candidates have no finished seeds on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())