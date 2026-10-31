"""Aggregate official ten-fold TE/EL results into fold-level statistics (working grade).

Reads only local, QA-passed result.json files copied from the cluster. Produces:
  data/rinalmo_te_el/TE_EL_TENFOLD_AGGREGATE.json

Scope note: per-fold p-values are paired training-seed contrasts; the ten-fold
test is a signed-rank statistic across official folds. It is a descriptive,
working-grade summary, not a population-level claim.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261030_12")
DATA = ROOT / "data/rinalmo_te_el"
OUT = DATA / "TE_EL_TENFOLD_AGGREGATE.json"

CELLS = ("Muscle", "HEK", "pc3")
TASKS = ("te", "el")
FOLDS = tuple(range(10))


def load_result(task: str, cell: str, fold: int) -> dict | None:
    path = DATA / f"rinalmo_{task}_{cell}_fold{fold}_round33_full_v4/result.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise RuntimeError(f"non-PASS result: {path}")
    protocol = data.get("protocol", {})
    if protocol.get("task") != task or protocol.get("cell_line") != cell or int(protocol.get("fold")) != fold:
        raise RuntimeError(f"protocol mismatch: {path}")
    if protocol.get("device") != "cuda":
        raise RuntimeError(f"non-GPU run: {path}")
    comparison = data["comparison"]
    return {
        "fold": fold,
        "delta": float(comparison["primary_delta_mean"]),
        "delta_sd": float(comparison["primary_delta_sample_sd"]),
        "fold_p": float(comparison["primary_one_sided_p_value"]),
        "official_spearman": float(data["heads"]["official_head"]["spearman_mean"]),
        "jev_spearman": float(data["heads"]["jev_head"]["spearman_mean"]),
        "significant_positive": bool(comparison.get("significant_positive", False)),
    }


def sign_rank(rows: list[dict]) -> dict:
    deltas = [row["delta"] for row in rows]
    positives = sum(value > 0 for value in deltas)
    try:
        from scipy.stats import wilcoxon
        p_value = float(wilcoxon(deltas, alternative="greater", method="exact").pvalue)
        method = "scipy.wilcoxon exact one-sided greater"
    except Exception as error:  # pragmatic fallback, still deterministic
        from math import comb
        p_value = float(sum(comb(len(deltas), k) for k in range(positives, len(deltas) + 1)) / 2 ** len(deltas))
        method = f"exact sign test fallback ({type(error).__name__})"
    return {
        "n_folds": len(deltas),
        "mean_delta": float(sum(deltas) / len(deltas)),
        "min_delta": float(min(deltas)),
        "max_delta": float(max(deltas)),
        "positive_folds": positives,
        "test": method,
        "one_sided_p": p_value,
    }


def holm(pairs: list[tuple[str, float]]) -> dict:
    ordered = sorted(pairs, key=lambda item: item[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (name, p_value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p_value))
        adjusted[name] = running
    return adjusted


def main() -> int:
    series, missing = {}, []
    for cell in CELLS:
        for task in TASKS:
            rows = [load_result(task, cell, fold) for fold in FOLDS]
            present = [row for row in rows if row is not None]
            key = f"{task.upper()}_{cell}"
            series[key] = {"cell": cell, "task": task, "rows": present}
            absent = [fold for fold, row in zip(FOLDS, rows) if row is None]
            if absent:
                missing.append({"series": key, "folds": absent})

    candidates = [name for name, entry in series.items() if entry["rows"]]
    adjusted = holm([(name, sign_rank(series[name]["rows"])["one_sided_p"]) for name in candidates])

    summary = {}
    for name in candidates:
        stats = sign_rank(series[name]["rows"])
        stats["holm_adjusted_p_across_present_series"] = adjusted[name]
        stats["complete_ten_folds"] = len(series[name]["rows"]) == 10
        summary[name] = stats

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ("fold-level summary of official ten-fold TE/EL protocols on frozen RiNALMo; "
                  "per-fold p is a paired training-seed contrast; cross-fold signed-rank is descriptive"),
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "missing": missing,
        "series": {
            name: {"cell": entry["cell"], "task": entry["task"], "rows": entry["rows"],
                   **summary[name]} for name, entry in series.items() if name in summary
        },
        "holm_family": candidates,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for name in candidates:
        stats = summary[name]
        print(f"{name}: n={stats['n_folds']} meanΔ={stats['mean_delta']:+.4f} "
              f"pos={stats['positive_folds']}/{stats['n_folds']} p={stats['one_sided_p']:.4g} "
              f"holm={stats['holm_adjusted_p_across_present_series']:.4g}")
    if missing:
        print(f"INCOMPLETE: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
