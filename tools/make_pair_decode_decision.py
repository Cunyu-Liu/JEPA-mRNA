#!/usr/bin/env python3
"""Apply the frozen route rule to VL0 and once-frozen external decode panels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

CANDIDATES = ("mea", "centroid", "raw_independent", "posterior_independent")


def load_routes(path):
    payload = json.loads(Path(path).read_text())
    return pd.DataFrame(payload["rows"] if isinstance(payload, dict) else payload)


def decide(vl0, external):
    rows, details = [], []
    for variant in sorted(set(vl0.score_variant) & set(external.score_variant)):
        dev = vl0[vl0.score_variant == variant].iloc[0]
        ext = external[external.score_variant == variant].copy()
        for decoder in CANDIDATES:
            prefix = f"{decoder}_"
            if f"{prefix}delta_micro_f1_vs_map" not in dev or pd.isna(dev[f"{prefix}delta_micro_f1_vs_map"]):
                continue
            ext["delta_macro_f1"] = ext[f"{prefix}macro_f1"] - ext.map_macro_f1
            ext["micro_precision_drop"] = ext.map_micro_precision - ext[f"{prefix}micro_precision"]
            ext["both_f1_improve"] = (ext[f"{prefix}delta_micro_f1_vs_map"] > 0) & (ext.delta_macro_f1 > 0)
            ext["micro_precision_ok"] = ext.micro_precision_drop <= .05 + 1e-12
            vl0_delta = float(dev[f"{prefix}delta_micro_f1_vs_map"])
            n_f1 = int(ext.both_f1_improve.sum())
            n_precision = int(ext.micro_precision_ok.sum())
            passed = vl0_delta >= .02 and len(ext) == 8 and n_f1 >= 6 and n_precision >= 6
            rows.append({"score_variant": variant, "decoder": decoder,
                         "vl0_map_micro_f1": float(dev.map_micro_f1),
                         "vl0_micro_f1": float(dev[f"{prefix}micro_f1"]),
                         "vl0_delta_micro_f1": vl0_delta,
                         "n_external": len(ext), "n_external_micro_f1_improve": int((ext[f"{prefix}delta_micro_f1_vs_map"] > 0).sum()),
                         "n_external_macro_f1_improve": int((ext.delta_macro_f1 > 0).sum()),
                         "n_external_both_f1_improve": n_f1,
                         "n_external_micro_precision_drop_le_0.05": n_precision,
                         "median_external_delta_micro_f1": float(ext[f"{prefix}delta_micro_f1_vs_map"].median()),
                         "median_external_delta_macro_f1": float(ext.delta_macro_f1.median()),
                         "median_external_delta_micro_recall": float((ext[f"{prefix}micro_recall"] - ext.map_micro_recall).median()),
                         "median_external_micro_precision_drop": float(ext.micro_precision_drop.median()),
                         "decode_main_effect_evidence": bool(passed)})
            for row in ext.to_dict("records"):
                details.append({"dataset_id": row["dataset_id"], "score_variant": variant, "decoder": decoder,
                                "map_micro_f1": row["map_micro_f1"], "candidate_micro_f1": row[f"{prefix}micro_f1"],
                                "delta_micro_f1": row[f"{prefix}delta_micro_f1_vs_map"],
                                "map_macro_f1": row["map_macro_f1"], "candidate_macro_f1": row[f"{prefix}macro_f1"],
                                "delta_macro_f1": row["delta_macro_f1"],
                                "map_micro_precision": row["map_micro_precision"],
                                "candidate_micro_precision": row[f"{prefix}micro_precision"],
                                "micro_precision_drop": row["micro_precision_drop"],
                                "map_micro_recall": row["map_micro_recall"],
                                "candidate_micro_recall": row[f"{prefix}micro_recall"],
                                "delta_micro_recall": row[f"{prefix}micro_recall"] - row["map_micro_recall"],
                                "both_f1_improve": bool(row["both_f1_improve"]),
                                "micro_precision_ok": bool(row["micro_precision_ok"])})
    return pd.DataFrame(rows), pd.DataFrame(details)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl0-route", type=Path, required=True)
    ap.add_argument("--external-route", type=Path, required=True)
    ap.add_argument("--coverage-findings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-label", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    decision, details = decide(load_routes(args.vl0_route), load_routes(args.external_route))
    coverage = json.loads(args.coverage_findings.read_text())
    coverage_rows = coverage["rows"] if isinstance(coverage, dict) and "rows" in coverage else coverage
    if isinstance(coverage_rows, dict):
        coverage_rows = list(coverage_rows.values())
    mask_factors = [r for r in coverage_rows if float(r["mask_theoretical_max_pair_recall"]) < .99]
    frequent_fn = [r for r in coverage_rows if float(r["coarse_local_seen_100plus_fn_rate"]) >= .20]
    any_decode = bool(decision.decode_main_effect_evidence.any()) if len(decision) else False
    external_complete = bool(len(decision)) and bool((decision.n_external == 8).all())
    if not external_complete:
        route = "INSUFFICIENT_EVIDENCE"
    elif any_decode:
        route = "DECODE_MAIN_EFFECT_EVIDENCE_PRESENT"
    elif len(frequent_fn) >= 6:
        route = "PRIORITIZE_TRAINING_OBJECTIVE_OPTIMIZATION_REPRESENTATION"
    else:
        route = "INSUFFICIENT_EVIDENCE"
    summary = {"schema_version": 1, "model_label": args.model_label,
               "frozen_rule": "VL0 delta micro-F1>=0.02; >=6/8 external both micro+macro F1 improve; >=6/8 micro precision drop<=0.05",
               "decode_main_effect_evidence": any_decode,
               "external_panels_complete": external_complete,
               "passing_candidates": decision[decision.decode_main_effect_evidence].to_dict("records") if len(decision) else [],
               "mask_task_definition_factor_splits": [r["dataset_id"] for r in mask_factors],
               "frequent_seen_environment_fn_ge_20_splits": [r["dataset_id"] for r in frequent_fn],
               "route_decision": route,
               "scope": "external 64-per-split sensitivity panels; not full-split SOTA"}
    decision.to_csv(args.out / "decode_candidate_decision.csv", index=False)
    details.to_csv(args.out / "decode_external_candidate_details.csv", index=False)
    (args.out / "final_route_decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
