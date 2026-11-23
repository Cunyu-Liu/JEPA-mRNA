#!/usr/bin/env python
"""Aggregate independently verified per-seed gates without selecting a champion seed."""
import argparse
import json
from pathlib import Path

from rnajepa.prior_alignment import sha256_file
from rnajepa.repr_head_matrix import CONTROL_ARM, active_repr_head_arms, load_repr_head_contract

METRICS = (
    "micro_auprc", "micro_recall_at_precision_0.8", "micro_pair_f1", "macro_pair_f1",
    "micro_pair_precision", "macro_pair_precision", "micro_pair_recall", "macro_pair_recall",
    "micro_stem_f1", "macro_stem_f1",
)


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed0-stage", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite three-seed verdict: {args.out}")
    contract = load_repr_head_contract(args.contract)
    contract_sha256 = sha256_file(args.contract)
    expected_variants = tuple(contract["three_seed_gate"]["promoted_arms"])
    seeds = {}
    for seed in (0, 1, 2):
        stage = args.seed0_stage if seed == 0 and args.seed0_stage else args.artifact_root / f"seed{seed}"
        verdict = read_json(stage / "evaluation/pilot_verdict.json")
        verification = read_json(stage / "verification.json")
        independent = read_json(stage / "final_independent_check.json")
        arms = active_repr_head_arms(contract, seed)
        expected_contract_sha256 = verdict["contract_sha256"] if seed == 0 else contract_sha256
        if (verification["contract_sha256"] != expected_contract_sha256
                or independent["contract_sha256"] != expected_contract_sha256
                or (seed > 0 and verdict["contract_sha256"] != contract_sha256)):
            raise ValueError(f"contract identity mismatch for seed {seed}")
        if (verification["status"] != "PASS" or independent["overall"] != "PASS"
                or tuple(verification["arms"]) != arms):
            raise ValueError(f"per-seed independent verification did not pass for seed {seed}")
        if set(verdict["arms"]) != set(arms):
            raise ValueError(f"verdict arms differ from frozen evaluation arms for seed {seed}")
        seed_result = {"verification_sha256": sha256_file(stage / "verification.json"),
                       "independent_check_sha256": sha256_file(stage / "final_independent_check.json"),
                       "verdict_sha256": sha256_file(stage / "evaluation/pilot_verdict.json"),
                       "arms": {}}
        for arm in expected_variants:
            comparison = verdict["comparisons"][arm]
            checks = comparison["checks"]
            comparison["three_seed_gate_passed"] = bool(
                comparison["direction_passed"] and all(checks.values())
                and comparison["delta_vs_control_matched"]["macro_pair_f1"] > 0)
            seed_result["arms"][arm] = comparison
        increments = {}
        for metric in METRICS:
            increments[metric] = (verdict["arms"]["repr_head"][metric]
                                  - verdict["arms"]["head_deep"][metric])
        seed_result["repr_head_minus_head_deep"] = increments
        seeds[str(seed)] = seed_result

    mean_deltas = {}
    for arm in expected_variants:
        mean_deltas[arm] = {
            metric: sum(seeds[str(seed)]["arms"][arm]["delta_vs_control_matched"][metric]
                        for seed in (0, 1, 2)) / 3
            for metric in seeds["0"]["arms"][arm]["delta_vs_control_matched"]
        }
    increment_stable = {
        metric: all(seeds[str(seed)]["repr_head_minus_head_deep"][metric] > 0 for seed in (0, 1, 2))
        for metric in METRICS
    }
    all_gates_pass = all(
        seeds[str(seed)]["arms"][arm]["three_seed_gate_passed"]
        for seed in (0, 1, 2) for arm in expected_variants
    )
    output = {
        "schema_version": 1,
        "status": "PASS",
        "decision": ("THREE_SEED_CAPACITY_DIRECTION_CONFIRMED" if all_gates_pass
                     else "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED"),
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
        "scope": "MODEL_DEV_BPRNA_VL0_THREE_SEED",
        "seeds": seeds,
        "mean_delta_vs_control_matched": mean_deltas,
        "repr_head_minus_head_deep_positive_all_seeds": increment_stable,
        "candidate_selected": False,
        "champion_seed_selected": False,
        "external_evaluation_allowed": all_gates_pass,
        "strict_ood_allowed": False,
        "sota_claim": False,
        "decoder_temperature_or_gamma_searched": False,
        "threshold_searched": False,
        "interpretation_boundary": (
            "Structure-head capacity is a robust contributing factor, not an exclusive root cause. "
            "A positive repr_head-minus-head_deep increment is reported separately and is not a promotion gate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "decision": output["decision"],
                      "external_evaluation_allowed": all_gates_pass, "sota_claim": False}, indent=2))


if __name__ == "__main__":
    main()
