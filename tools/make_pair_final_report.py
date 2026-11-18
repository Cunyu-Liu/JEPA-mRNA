#!/usr/bin/env python3
"""Combine accepted coverage/raw/DP evidence into one route report and slide figure."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False})

MODEL_LABEL = {"plana0": "P0", "plana1": "P1", "plana_mean": "P-mean", "r2d": "R2D", "xens2": "xens2"}
DATASET_LABEL = {"archiveii_embok_clean": "ArchiveII", "bprna_new": "bpRNA-new", "bprna_ts0": "TS0", "chanrg_test": "CHANRG Test", "chanrg_gena": "GenA", "chanrg_genc": "GenC", "chanrg_genf": "GenF", "rfam_fam": "Rfam-family"}


def read_json(path):
    return json.loads(Path(path).read_text())


def require_pass(path, key="status"):
    data = read_json(path)
    if data.get(key) != "PASS":
        raise RuntimeError(f"verification not passed: {path}")
    return data


def variant_label(variant):
    model, prior = variant.rsplit("_", 1)
    return f"{MODEL_LABEL.get(model, model)}-{prior}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.artifact_root, args.out
    out.mkdir(parents=True, exist_ok=True)
    r2d = read_json(root / "decode_r2d_decision/final_route_decision.json")
    xens = read_json(root / "decode_xens2_decision/final_route_decision.json")
    r2d_candidates = pd.read_csv(root / "decode_r2d_decision/decode_candidate_decision.csv")
    xens_candidates = pd.read_csv(root / "decode_xens2_decision/decode_candidate_decision.csv")
    candidates = pd.concat([r2d_candidates, xens_candidates], ignore_index=True).drop_duplicates(["score_variant", "decoder"], keep="last")
    structured = candidates[candidates.decoder.isin(["mea", "centroid"])].copy()
    best = structured.sort_values(["score_variant", "n_external_both_f1_improve", "median_external_delta_micro_f1"], ascending=[True, False, False]).groupby("score_variant", as_index=False).first()
    best["variant_label"] = best.score_variant.map(variant_label)
    best = best.sort_values("score_variant", key=lambda s: s.map(lambda v: (v.rsplit("_", 1)[0], v.rsplit("_", 1)[1])))
    ensemble = pd.read_csv(root / "decode_xens2_ensemble/ensemble_member_complement.csv")
    ensemble_summary = read_json(root / "decode_xens2_ensemble/ensemble_member_summary.json")
    verification = {
        "mask_coverage": require_pass(root / "independent_verification_mask_coverage.json"),
        "r2d_vl0_decode": require_pass(root / "decode_r2d_vl0/independent_decode_verification.json"),
        "r2d_external_decode": require_pass(root / "decode_r2d_external/independent_external_raw_dp_verification.json"),
        "r2d_route_decision": require_pass(root / "decode_r2d_decision/independent_route_decision_verification.json"),
        "xens2_vl0_decode": require_pass(root / "decode_xens2_vl0/independent_decode_verification.json"),
        "xens2_external_decode": require_pass(root / "decode_xens2_external/independent_external_raw_dp_verification.json"),
        "xens2_route_decision": require_pass(root / "decode_xens2_decision/independent_route_decision_verification.json"),
        "xens2_raw_fusion": require_pass(root / "raw_scores_chunk64/raw_xens2_full_verification.json"),
    }
    if not r2d["external_panels_complete"] or not xens["external_panels_complete"]:
        raise RuntimeError("external panels are incomplete")
    any_decode = bool(r2d["decode_main_effect_evidence"] or xens["decode_main_effect_evidence"])
    route = "DECODE_MAIN_EFFECT_EVIDENCE_PRESENT" if any_decode else xens["route_decision"]
    passing_candidates = [{"model_label": "r2d", **row} for row in r2d["passing_candidates"]] + [{"model_label": "five_model", **row} for row in xens["passing_candidates"]]
    turner_on_passing = [row for row in passing_candidates if row["score_variant"].endswith("_on")]
    report = {
        "schema_version": 1, "status": "PASS", "route_decision": route,
        "model_route_decisions": {"r2d": r2d["route_decision"], "full_five_model_panel": xens["route_decision"]},
        "decode_main_effect_evidence": any_decode,
        "n_decode_candidates": int(len(candidates)),
        "n_structured_decode_candidates": int(len(structured)),
        "n_structured_decode_candidates_passing": int(structured.decode_main_effect_evidence.sum()),
        "passing_candidates": passing_candidates,
        "turner_on_passing_candidates": turner_on_passing,
        "no_training_started": True,
        "mask_task_definition_factor_splits": xens["mask_task_definition_factor_splits"],
        "frequent_seen_environment_fn_ge_20_splits": xens["frequent_seen_environment_fn_ge_20_splits"],
        "ensemble_member_summary": ensemble_summary,
        "verification": {k: {"status": v["status"], "n_failed": v.get("n_failed", len(v.get("failures", [])))} for k, v in verification.items()},
        "scope": "VL0 196 for development; eight external 64-per-split sensitivity panels with one frozen application; not full-split SOTA.",
        "interpretation": [
            "No candidate meets the preregistered decode-main-effect rule." if not any_decode else ("Decode-main-effect evidence is restricted to the listed Turner-off variants; no Turner-on candidate passes." if not turner_on_passing else "At least one candidate meets the preregistered decode-main-effect rule; inspect score-variant scope before choosing a route."),
            "Errors remain frequent in pair environments seen at least 100 times in TR1c; simple lack of local pair-environment coverage is not the main explanation.",
            "CHANRG Test/GenA/GenC/GenF additionally have task-definition/mask ceilings and must be reported separately from legal-GT recall.",
            "Prioritize training objective, optimization and representation diagnostics; do not add ensemble members or claim a decoder fix as the main route." if not any_decode else "Decoder correction is a validated contributor for the listed Turner-off variants, not a completed Turner-on production repair; residual errors still require label, objective, optimization and representation analysis."
        ],
        "limits": "External panels are sensitivity evidence. CPU fallbacks are retained in raw metadata; smoke CPU/GPU agreement is not a claim of whole-panel bitwise identity."
    }
    (out / "final_pair_diagnostic_decision.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    best.to_csv(out / "final_structured_decoder_summary.csv", index=False)

    fig = plt.figure(figsize=(15.84, 4.95), dpi=200, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.3], wspace=.40, left=.09, right=.985, top=.82, bottom=.28)
    ax = fig.add_subplot(gs[0, 0])
    order = list(reversed(range(len(best))))
    colors = ["#247BA0" if v.endswith("_off") else "#70C1B3" for v in best.score_variant]
    ax.barh(order, best.n_external_both_f1_improve, color=colors, height=.68)
    ax.axvline(6, color="#D1495B", lw=1.8, ls="--")
    for y, value in zip(order, best.n_external_both_f1_improve):
        ax.text(value + .10, y, str(int(value)), va="center", fontsize=11, color="#243447")
    ax.set_yticks(order, best.variant_label, fontsize=11)
    ax.set_xlim(0, 8.6); ax.set_xticks(range(0, 9, 2)); ax.set_xlabel("External splits with micro+macro F1 gain", fontsize=11.5)
    ax.set_title("A  External micro+macro F1 gains", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#D9E2EC", lw=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    focus = candidates[(candidates.score_variant.isin(["r2d_off", "r2d_on", "xens2_off", "xens2_on"])) & (candidates.decoder == "mea")].copy()
    focus["variant_label"] = focus.score_variant.map(variant_label)
    metrics = [("vl0_delta_micro_f1", "VL0 Δmicro-F1"), ("median_external_delta_micro_f1", "External Δmicro-F1"), ("median_external_delta_macro_f1", "External Δmacro-F1"), ("median_external_delta_micro_recall", "External Δrecall")]
    x = np.arange(len(metrics)); width = .19
    for i, row in enumerate(focus.to_dict("records")):
        values = [100 * float(row[k]) for k, _ in metrics]
        ax.bar(x + (i - 1.5) * width, values, width, label=row["variant_label"], color=["#9BC1BC", "#247BA0", "#F2C14E", "#D1495B"][i])
    ax.axhline(0, color="#52606D", lw=.9)
    ax.set_xticks(x, [label for _, label in metrics], fontsize=10, rotation=18, ha="right")
    ax.set_ylabel("Change vs MAP (percentage points)", fontsize=11.5)
    ax.set_title("B  MEA: Turner-off vs Turner-on", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9.5, ncol=2, frameon=False, loc="upper left")
    ax.grid(axis="y", color="#D9E2EC", lw=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ext = ensemble[ensemble.dataset_role == "external64"].copy()
    datasets = list(DATASET_LABEL)
    x = np.arange(len(datasets)); width = .34
    for i, prior in enumerate(["off", "on"]):
        subset = ext[ext.prior == prior].set_index("dataset_id").reindex(datasets)
        values = 100 * subset.xens2_delta_map_micro_recall_vs_best_atomic
        ax.bar(x + (i - .5) * width, values, width, label=f"Turner {prior}", color=["#247BA0", "#F2C14E"][i])
    ax.axhline(0, color="#52606D", lw=.9)
    ax.set_xticks(x, [DATASET_LABEL[d] for d in datasets], fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Δ MAP recall (percentage points)", fontsize=11.5)
    ax.set_title("C  xens2 recall vs best member", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax.grid(axis="y", color="#D9E2EC", lw=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.09, .04, "External: eight frozen 64-sequence panels; parameters selected on VL0 (n=196). Dashed line: 6/8 threshold. Passing candidates are Turner-off only.", fontsize=10.5, color="#52606D")
    fig.savefig(out / "final_pair_diagnostic_slide.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
