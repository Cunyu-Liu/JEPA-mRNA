#!/usr/bin/env python3
"""Apply the frozen three-seed terminal matrix criteria without external-panel access."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.prior_alignment import sha256_file

ARMS = ("C0", "C1", "C2", "C3")
METRICS = (
    "micro_pair_precision", "micro_pair_recall", "micro_pair_f1",
    "macro_pair_precision", "macro_pair_recall", "macro_pair_f1",
    "micro_stem_precision", "micro_stem_recall", "micro_stem_f1",
    "macro_stem_precision", "macro_stem_recall", "macro_stem_f1",
)


def _mean(values):
    return statistics.fmean(values)


def evaluate_seed_metrics(seed_rows, contract):
    """Evaluate already verified per-seed summaries; no file or external access."""
    gate = contract["full_matrix_gate"]
    primary = gate["primary_endpoint"]
    secondary = gate["secondary_endpoint"]
    means = {arm: {metric: _mean([row[arm][metric] for row in seed_rows]) for metric in METRICS}
             for arm in ARMS}
    c2_minus_c1 = {metric: means["C2"][metric] - means["C1"][metric] for metric in METRICS}
    c2_minus_c0 = {metric: means["C2"][metric] - means["C0"][metric] for metric in METRICS}
    c2_minus_c3 = {metric: means["C2"][metric] - means["C3"][metric] for metric in METRICS}
    seed_macro_deltas = [row["C2"]["macro_pair_f1"] - row["C1"]["macro_pair_f1"] for row in seed_rows]
    checks = {
        "macro_pair_f1_mean_delta": c2_minus_c1["macro_pair_f1"] >= primary["macro_pair_f1_mean_delta_min"],
        "micro_pair_f1_mean_delta": c2_minus_c1["micro_pair_f1"] >= primary["micro_pair_f1_mean_delta_min"],
        "micro_pair_precision_margin": c2_minus_c1["micro_pair_precision"] >= primary["micro_pair_precision_mean_delta_min"],
        "macro_pair_precision_margin": c2_minus_c1["macro_pair_precision"] >= primary["macro_pair_precision_mean_delta_min"],
        "three_seed_macro_direction": all(delta > 0 for delta in seed_macro_deltas),
        "micro_stem_noninferiority": c2_minus_c1["micro_stem_f1"] >= secondary["micro_stem_f1_mean_delta_min"],
        "macro_stem_noninferiority": c2_minus_c1["macro_stem_f1"] >= secondary["macro_stem_f1_mean_delta_min"],
        "C2_vs_C0_micro": c2_minus_c0["micro_pair_f1"] >= 0,
        "C2_vs_C0_macro": c2_minus_c0["macro_pair_f1"] > 0,
    }
    primary_pass = all(checks[name] for name in (
        "macro_pair_f1_mean_delta", "micro_pair_f1_mean_delta",
        "micro_pair_precision_margin", "macro_pair_precision_margin",
        "three_seed_macro_direction"))
    stem_pass = checks["micro_stem_noninferiority"] and checks["macro_stem_noninferiority"]
    c0_stronger = c2_minus_c0["micro_pair_f1"] < 0 and c2_minus_c0["macro_pair_f1"] < 0
    c0_baseline_pass = checks["C2_vs_C0_micro"] and checks["C2_vs_C0_macro"]
    candidate = primary_pass and stem_pass and c0_baseline_pass and not c0_stronger
    if c0_stronger:
        decision = "STOP_TRAIN_OFF_MAINLINE_C0_FINAL_STRONGER"
    elif not checks["three_seed_macro_direction"]:
        decision = "NO_ROUTE_TRAINING_STABILITY"
    elif primary_pass and not stem_pass:
        decision = "PAIR_STEM_OBJECTIVE_MISMATCH"
    elif candidate:
        decision = "SELECT_ALIGNED_OFF_CANDIDATE_FOR_ONE_FROZEN_EXTERNAL_EVALUATION"
    else:
        decision = "NO_ROUTE_FROZEN_ENDPOINT_NOT_MET"
    mixed_evidence = any(
        (c2_minus_c1[micro] > 0) != (c2_minus_c1[macro] > 0)
        for micro, macro in (("micro_pair_f1", "macro_pair_f1"),
                             ("micro_stem_f1", "macro_stem_f1")))
    return {
        "decision": decision,
        "candidate_selected": candidate,
        "external_evaluation_allowed": candidate,
        "production_repair": False,
        "sota_claim": False,
        "primary_pass": primary_pass,
        "stem_noninferiority_pass": stem_pass,
        "C0_baseline_pass": c0_baseline_pass,
        "C0_final_stronger": c0_stronger,
        "mixed_evidence": mixed_evidence,
        "checks": checks,
        "means": means,
        "C2_minus_C1": c2_minus_c1,
        "C2_minus_C0": c2_minus_c0,
        "C2_minus_C3": c2_minus_c3,
        "per_seed_macro_pair_f1_C2_minus_C1": seed_macro_deltas,
        "C3_repeatedly_weaker_descriptive": all(
            row["C3"]["macro_pair_f1"] < row["C2"]["macro_pair_f1"] for row in seed_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-verification", type=Path,
                        help="independent 24-checkpoint tensor verification; required for final audited verdict")
    args = parser.parse_args()
    contract_path = REPO / "configs/prior_alignment_2x2_v1.json"
    contract = json.loads(contract_path.read_text())
    contract_sha256 = sha256_file(contract_path)
    seed_rows, provenance = [], []
    for seed in contract["full_matrix_gate"]["seeds"]:
        stage = args.artifact_root / f"full_seed{seed}"
        manifest = json.loads((stage / "stage_manifest.json").read_text())
        engineering = json.loads((stage / "stage_verification.json").read_text())
        science = json.loads((stage / "scientific_verification.json").read_text())
        summary = json.loads((stage / "decode_vl0/vl0_selected_summary.json").read_text())
        assert manifest["status"] == "COMPLETED" and manifest["contract_sha256"] == contract_sha256
        assert not manifest["old_external_panels_used"]
        assert engineering["status"] == "PASS" and science["status"] == "PASS"
        assert not science["route_winner_selected"] and science["mode"] == "full" and science["seed"] == seed
        assert summary["terminal_step"] == 20000 and summary["n_sequences"] == 196
        assert not summary["old_external_panels_used"] and set(summary["arms"]) == set(ARMS)
        training_completion = []
        for arm in ("on", "off"):
            run_meta = json.loads((stage / f"train_{arm}/run_meta.json").read_text())
            log_lines = (stage / f"train_{arm}/train_log.jsonl").read_text().splitlines()
            last_log = json.loads(log_lines[-1])
            assert run_meta["status"] == "completed" and run_meta["early_stopped"] is False
            assert run_meta["schedule"]["steps"] == 20000 and last_log["step"] == 20000
            receipt = engineering["checkpoint_trajectory"][f"{arm}_20000"]
            assert receipt["actual_unfrozen_blocks"] == last_log["unfrozen"] == 33
            training_completion.append({
                "trajectory": f"train_{arm}", "status": "completed",
                "stopping_reason": "FIXED_20000_STEP_BUDGET_REACHED_NO_EARLY_STOP",
                "terminal_step": last_log["step"], "actual_unfrozen_blocks": 33,
                "checkpoint_sha256": engineering["checkpoint_sha256"][arm],
                "backbone_update_sha256": receipt["backbone_update_sha256"],
            })
        seed_rows.append(summary["arms"])
        provenance.append({
            "seed": seed, "stage": str(stage),
            "engineering_checks": engineering["n_checks"], "science_checks": science["n_checks"],
            "git_head": manifest["git_head"], "contract_sha256": contract_sha256,
            "training_completion": training_completion,
        })
    checkpoint_evidence = None
    if args.checkpoint_verification:
        verified = json.loads(args.checkpoint_verification.read_text())
        assert verified["status"] == "PASS" and verified["n_checkpoints"] == 24
        assert verified["contract_sha256"] == contract_sha256 and not verified["scientific_route_selected"]
        by_identity = {(record["seed"], record["trajectory"], record["step"]): record
                       for record in verified["records"]}
        for provenance_row in provenance:
            seed = provenance_row["seed"]
            engineering = json.loads((Path(provenance_row["stage"]) / "stage_verification.json").read_text())
            for trajectory in ("train_on", "train_off"):
                short_arm = trajectory.removeprefix("train_")
                for step in (3200, 9600, 16800, 20000):
                    record = by_identity[(seed, trajectory, step)]
                    receipt = engineering["checkpoint_trajectory"][f"{short_arm}_{step}"]
                    assert record["status"] == "PASS"
                    assert record["checkpoint_sha256"] == receipt["checkpoint_sha256"]
                    assert record["backbone_update_sha256"] == receipt["backbone_update_sha256"]
        checkpoint_evidence = {
            "path": str(args.checkpoint_verification),
            "sha256": sha256_file(args.checkpoint_verification),
            "n_checkpoints": verified["n_checkpoints"], "n_checks": verified["n_checks"],
        }
    result = evaluate_seed_metrics(seed_rows, contract)
    result.update({
        "status": "PASS", "scope": "FROZEN_THREE_SEED_TERMINAL_VL0_MATRIX",
        "contract_sha256": contract_sha256, "seeds": [0, 1, 2],
        "terminal_step": 20000, "old_external_panels_used": False,
        "provenance": provenance, "checkpoint_tensor_verification": checkpoint_evidence,
    })
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "candidate_selected", "external_evaluation_allowed",
        "primary_pass", "stem_noninferiority_pass", "C0_baseline_pass", "mixed_evidence")}, indent=2))


if __name__ == "__main__":
    main()
