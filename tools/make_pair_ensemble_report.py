#!/usr/bin/env python3
"""Descriptive xens2/member complement analysis; no external parameter tuning."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

ATOMIC = ("plana0", "plana1", "r2d")
METRICS = ("map_micro_recall", "map_macro_recall", "map_micro_f1", "map_macro_f1",
           "map_micro_precision", "raw_micro_AUPRC", "raw_macro_AUPRC")


def load(path):
    x = json.loads(Path(path).read_text())
    return pd.DataFrame(x["rows"] if isinstance(x, dict) else x)


def analyze(vl0, external):
    frames = []
    for role, frame in (("VL0", vl0), ("external64", external)):
        x = frame.copy()
        x["dataset_role"] = role
        frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    data["model"] = data.score_variant.str.rsplit("_", n=1).str[0]
    data["prior"] = data.score_variant.str.rsplit("_", n=1).str[1]
    rows = []
    for (role, dataset, prior), group in data.groupby(["dataset_role", "dataset_id", "prior"]):
        by_model = {r["model"]: r for r in group.to_dict("records")}
        if "xens2" not in by_model or not all(m in by_model for m in ATOMIC):
            continue
        ensemble = by_model["xens2"]
        row = {"dataset_role": role, "dataset_id": dataset, "prior": prior}
        for metric in METRICS:
            values = {m: float(by_model[m][metric]) for m in ATOMIC}
            for m, value in values.items():
                row[f"{m}_{metric}"] = value
            if "plana_mean" in by_model:
                row[f"plana_mean_{metric}"] = float(by_model["plana_mean"][metric])
            row[f"xens2_{metric}"] = float(ensemble[metric])
            row[f"xens2_delta_{metric}_vs_best_atomic"] = float(ensemble[metric]) - max(values.values())
            row[f"xens2_delta_{metric}_vs_r2d"] = float(ensemble[metric]) - values["r2d"]
            if "plana_mean" in by_model:
                row[f"xens2_delta_{metric}_vs_plana_mean"] = float(ensemble[metric]) - row[f"plana_mean_{metric}"]
        row["micro_recall_below_all_atomic"] = row["xens2_map_micro_recall"] < min(row[f"{m}_map_micro_recall"] for m in ATOMIC)
        row["macro_recall_below_all_atomic"] = row["xens2_map_macro_recall"] < min(row[f"{m}_map_macro_recall"] for m in ATOMIC)
        rows.append(row)
    details = pd.DataFrame(rows)
    summary = {"schema_version": 1, "n_rows": len(details),
               "n_external_micro_recall_below_all_atomic": int(details.query("dataset_role == 'external64'").micro_recall_below_all_atomic.sum()) if len(details) else 0,
               "n_external_macro_recall_below_all_atomic": int(details.query("dataset_role == 'external64'").macro_recall_below_all_atomic.sum()) if len(details) else 0,
               "interpretation": "Descriptive member-complement diagnostic only; no threshold or route decision is tuned on external panels."}
    return details, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl0-route", type=Path, required=True)
    ap.add_argument("--external-route", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    details, summary = analyze(load(args.vl0_route), load(args.external_route))
    details.to_csv(args.out / "ensemble_member_complement.csv", index=False)
    (args.out / "ensemble_member_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
