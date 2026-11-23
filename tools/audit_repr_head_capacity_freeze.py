#!/usr/bin/env python
"""Freeze the capacity-route stop decision from raw-derived per-seed metrics."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from verify_raw_score_matrix import direction_gate

ARMS = ("control_matched", "head_deep", "repr_head")
VARIANTS = ("head_deep", "repr_head")
METRICS = (
    "micro_auprc", "micro_recall_at_precision_0.8", "micro_pair_f1", "macro_pair_f1",
    "micro_pair_precision", "macro_pair_precision", "micro_pair_recall", "macro_pair_recall",
    "micro_stem_f1", "macro_stem_f1",
)


def read_json(path):
    return json.loads(path.read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparisons_from_raw_recheck(recheck, contract):
    if (recheck.get("scope") != "INDEPENDENT_REPRESENTATION_HEAD_RECHECK_FROM_RAW_SCORES"
            or recheck.get("overall") != "PASS" or recheck.get("n_sequences") != 196):
        raise ValueError("raw-score independent recheck is incomplete")
    metrics = recheck["recomputed_arms"]
    output = {}
    for arm in VARIANTS:
        delta, checks, passed = direction_gate(
            metrics[arm], metrics["control_matched"], contract["pilot"]["per_arm_direction_gate"])
        output[arm] = {
            "delta_vs_control_matched": delta,
            "checks": checks,
            "direction_passed": passed,
            "failed_gates": sorted(key for key, value in checks.items() if not value),
        }
    return output


def audit_run(stage, seed, arm, contract, check_checkpoint_hash=True):
    run_dir = stage / arm
    meta = read_json(run_dir / "run_meta.json")
    provenance = read_json(stage / "provenance.json")
    contract_sha256 = sha256_file(contract["contract_path"])
    expected_init = contract["training"][f"shared_init_sha256_seed{seed}"]
    checkpoint = run_dir / "snapshots" / f"{arm}_step9600.pt"
    checkpoint_sha256 = sha256_file(checkpoint) if check_checkpoint_hash else None
    rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
    finite = all(math.isfinite(row["loss"]) and math.isfinite(row["gnorm"]) for row in rows)
    export_dir = run_dir / "export_train_off"
    exports = sorted((export_dir / "bprna_vl0").glob("*.npz"))
    export_meta = read_json(export_dir / "export_metadata.json")
    checks = {
        "completed_at_step_cap": meta["status"] == "completed" and meta["stop_reason"] == "step_cap",
        "steps_9600": meta["steps_completed"] == 9600 and rows[-1]["step"] == 9600,
        "finite_loss_gnorm": finite and not meta["divergence_reason"],
        "raw_export_196": len(exports) == 196,
        "contract_identity": meta["prior_alignment"]["contract_sha256"] == contract_sha256,
        "shared_init_identity": meta["prior_alignment"]["shared_init_sha256"] == expected_init == provenance["INIT_SHA256"],
        "fresh_head": meta["head_init"] == "fresh",
        "control_retrained": not meta["prior_alignment"]["full_control_reused_not_retrained"],
        "seed_identity": meta["seed"] == seed == int(provenance["SEED"]),
        "code_identity": bool(provenance["RUN_GIT_HEAD"]) and bool(provenance["TRAIN_SHA256"]),
        "checkpoint_identity": checkpoint_sha256 == export_meta["checkpoint"]["sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"run audit failed for seed{seed}/{arm}: {checks}")
    return {
        "status": meta["status"], "stop_reason": meta["stop_reason"],
        "steps": meta["steps_completed"], "train_log_rows": len(rows),
        "finite_loss_gnorm": finite, "raw_export_count": len(exports),
        "run_git_head": provenance["RUN_GIT_HEAD"], "train_code_sha256": provenance["TRAIN_SHA256"],
        "contract_sha256": contract_sha256, "shared_init_sha256": expected_init,
        "checkpoint_sha256": checkpoint_sha256, "checks": checks,
    }


def build_audit(contract_path, artifact_root, seed0_stage, check_checkpoint_hash=True):
    contract = read_json(contract_path)
    contract["contract_path"] = contract_path
    seeds, runs = {}, {}
    for seed in (0, 1, 2):
        stage = seed0_stage if seed == 0 else artifact_root / f"seed{seed}"
        recheck_path = stage / "final_independent_check.json"
        recheck = read_json(recheck_path)
        comparisons = comparisons_from_raw_recheck(recheck, contract)
        official = read_json(stage / "evaluation" / "pilot_verdict.json")
        for arm in VARIANTS:
            expected = official["comparisons"][arm]
            for metric, value in comparisons[arm]["delta_vs_control_matched"].items():
                if not math.isclose(value, expected["delta_vs_control_matched"][metric], abs_tol=1e-12):
                    raise ValueError(f"delta mismatch seed{seed}/{arm}/{metric}")
            if any(comparisons[arm]["checks"][key] != value
                   for key, value in expected["checks"].items()):
                raise ValueError(f"gate mismatch seed{seed}/{arm}")
        seeds[str(seed)] = {
            "raw_recheck_sha256": sha256_file(recheck_path),
            "official_verdict_sha256": sha256_file(stage / "evaluation" / "pilot_verdict.json"),
            "arms": comparisons,
        }
        if seed in (1, 2):
            runs[str(seed)] = {arm: audit_run(stage, seed, arm, contract, check_checkpoint_hash)
                               for arm in ARMS}
    means, mean_gates = {}, {}
    for arm in VARIANTS:
        delta_metrics = seeds["0"]["arms"][arm]["delta_vs_control_matched"]
        means[arm] = {metric: sum(seeds[str(seed)]["arms"][arm]["delta_vs_control_matched"][metric]
                                  for seed in (0, 1, 2)) / 3 for metric in delta_metrics}
        candidate = dict(means[arm])
        control = {metric: 0.0 for metric in delta_metrics}
        _, checks, passed = direction_gate(candidate, control, contract["pilot"]["per_arm_direction_gate"])
        mean_gates[arm] = {"checks": checks, "direction_passed": passed,
                           "failed_gates": sorted(key for key, value in checks.items() if not value)}
    official_three_seed = read_json(artifact_root / "three_seed_verdict.json")
    checks = {
        "negative_decision": official_three_seed["decision"] == "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED",
        "candidate_not_selected": official_three_seed["candidate_selected"] is False,
        "champion_not_selected": official_three_seed["champion_seed_selected"] is False,
        "external_closed": official_three_seed["external_evaluation_allowed"] is False,
        "strict_ood_closed": official_three_seed["strict_ood_allowed"] is False,
        "no_sota": official_three_seed["sota_claim"] is False,
        "both_arms_fail_mean_direction": all(not mean_gates[arm]["direction_passed"] for arm in VARIANTS),
        "seeds_1_2_fail_every_arm": all(not seeds[str(seed)]["arms"][arm]["direction_passed"]
                                        for seed in (1, 2) for arm in VARIANTS),
    }
    if not all(checks.values()):
        raise ValueError(f"freeze checks failed: {checks}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "RAW_DERIVED_THREE_SEED_CAPACITY_FREEZE",
        "decision": "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED",
        "contract_sha256": sha256_file(contract_path),
        "seeds": seeds,
        "six_confirmation_runs": runs,
        "mean_delta_vs_control_matched": means,
        "mean_direction_gates": mean_gates,
        "candidate_selected": False,
        "external_evaluation_allowed": False,
        "strict_ood_allowed": False,
        "sota_claim": False,
        "checks": checks,
        "scientific_boundary": (
            "Simple capacity expansion is not a robust production route. Raw ranking gains did not "
            "transfer reliably to decoded pair/stem endpoints; representation, objective and optimization "
            "causes remain unseparated. This audit neither retrains nor selects a champion seed/checkpoint."
        ),
        "source_narrative_erratum": (
            "The older aggregate phrase 'capacity is a robust contributing factor' is unsupported and "
            "superseded by this freeze audit; numerical gates and artifacts are unchanged."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed0-stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite audit: {args.out}")
    result = build_audit(args.contract, args.artifact_root, args.seed0_stage,
                         check_checkpoint_hash=not args.skip_checkpoint_hash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "failed_mean_gates": {arm: result["mean_direction_gates"][arm]["failed_gates"]
                                            for arm in VARIANTS}}, indent=2))


if __name__ == "__main__":
    main()
