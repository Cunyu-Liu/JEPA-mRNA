#!/usr/bin/env python3
"""Independent numerical and convergence-contract audit for structured margin."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rnajepa.harness import valid_pair_mask
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from rnajepa.structured_margin import pair_hamming_structured_margin

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
METRIC = "length_normalized_pair_hamming_structured_margin"
STOP_REASON = "CONVERGED_VALIDATION_PAIR_HAMMING_STRUCTURED_MARGIN_AFTER_TERMINAL_UNFREEZE"
CHECKPOINT_FORMAT = "foldnar.structured_margin_checkpoint.v1"


def enumerate_legal_structures_by_subset(mask: np.ndarray, min_loop: int = 3):
    legal = [
        (i, j)
        for i in range(mask.shape[0])
        for j in range(i + min_loop + 1, mask.shape[1])
        if bool(mask[i, j])
    ]
    if len(legal) > 20:
        raise ValueError("subset oracle is restricted to at most 20 legal pairs")
    for bits in range(1 << len(legal)):
        occupied = 0
        chosen = []
        valid = True
        for index, (i, j) in enumerate(legal):
            if not bits & (1 << index):
                continue
            if occupied & ((1 << i) | (1 << j)):
                valid = False
                break
            if any(i < k < j < l or k < i < l < j for k, l in chosen):
                valid = False
                break
            occupied |= (1 << i) | (1 << j)
            chosen.append((i, j))
        if valid:
            yield tuple(chosen)


def subset_oracle(scores, mask, target, min_loop=3, normalization="length"):
    target = set(target)
    structures = list(enumerate_legal_structures_by_subset(mask, min_loop))
    values = [
        sum(float(scores[i, j]) for i, j in structure) + len(set(structure) ^ target)
        for structure in structures
    ]
    best_index = max(range(len(values)), key=values.__getitem__)
    best_value = values[best_index]
    if sum(math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-10) for value in values) != 1:
        raise ValueError("gradient oracle requires a unique cost-augmented maximizer")
    best = structures[best_index]
    value = best_value - sum(float(scores[i, j]) for i, j in target)
    gradient = np.zeros_like(scores, dtype=np.float64)
    for i, j in best:
        gradient[i, j] += 1.0
    for i, j in target:
        gradient[i, j] -= 1.0
    if normalization == "length":
        value /= scores.shape[0]
        gradient /= scores.shape[0]
    return value, gradient, len(structures), best


def run_independent_exhaustive(seed=920261126):
    rng = np.random.default_rng(seed)
    cases = 0
    max_value_error = 0.0
    max_gradient_error = 0.0
    structure_count = 0
    for length in (4, 5, 6):
        for chars in itertools.product("AUGC", repeat=length):
            sequence = "".join(chars)
            mask = valid_pair_mask(sequence)
            structures = list(enumerate_legal_structures_by_subset(mask))
            target = structures[cases % len(structures)]
            scores = np.zeros((length, length), dtype=np.float64)
            scores[np.triu_indices(length, 1)] = rng.normal(size=length * (length - 1) // 2)
            expected, expected_gradient, count, _ = subset_oracle(scores, mask, target)
            tensor = torch.tensor(scores, requires_grad=True)
            loss = pair_hamming_structured_margin(
                tensor, torch.tensor(mask), target, normalization="length", min_loop=3,
            )
            loss.backward()
            max_value_error = max(max_value_error, abs(float(loss) - expected))
            max_gradient_error = max(
                max_gradient_error, float(np.max(np.abs(tensor.grad.numpy() - expected_gradient)))
            )
            structure_count += count
            cases += 1
    if cases != 5376:
        raise AssertionError(f"expected 5376 cases, observed {cases}")
    if max_value_error > 1e-10 or max_gradient_error > 1e-12:
        raise AssertionError("independent subset enumeration disagrees with the production loss")
    return {
        "status": "PASS",
        "oracle": "independent_legal_pair_subset_enumeration_no_dp_recurrence",
        "sequences": cases,
        "lengths": [4, 5, 6],
        "seed": seed,
        "structures_scored": structure_count,
        "max_absolute_loss_error": max_value_error,
        "max_absolute_gradient_error": max_gradient_error,
        "loss_tolerance": 1e-10,
        "gradient_tolerance": 1e-12,
    }


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_converged_margin_arm(directory: str | Path):
    directory = Path(directory)
    meta = json.loads((directory / "run_meta.json").read_text())
    _require(meta["schema_version"] == "foldnar.structured_margin_training_run.v1", "wrong run schema")
    _require(meta["arm"] in ARMS and meta["status"] == "converged" and meta["converged"], "not converged")
    _require(meta["stop_reason"] == STOP_REASON, "wrong stop reason")
    _require(meta["validation_metric"] == METRIC, "wrong validation metric")
    _require(meta["objective"]["name"] == "exact_pair_hamming_cost_augmented_structured_max_margin", "wrong objective")
    _require(meta["objective"]["lambda_nll"] == 0.0, "NLL weight is not zero")
    _require(meta["run_identity"]["validation_metric"] == METRIC, "run identity metric mismatch")
    _require(meta["final_unfrozen_blocks"] == 20, "not terminal 20/33 unfreeze")
    validations = [json.loads(line) for line in (directory / "validation_log.jsonl").read_text().splitlines() if line.strip()]
    _require(validations, "empty validation log")
    _require(all(row["validation_metric"] == METRIC for row in validations), "validation metric drift")
    stopped = validations[-1]
    _require(stopped["stop"] and stopped["eligible"] and stopped["bad_evaluations"] == 5, "no patience-5 stop")
    _require(stopped["step"] == meta["training_stopped_step"] == meta["steps_completed"], "stopped step mismatch")
    for row in validations:
        if row["n_unfrozen"] < 20:
            _require(not row["eligible"] and not row["save_best"] and not row["stop"], "pre-terminal selection")
    convergence = meta["convergence"]
    for key in ("best_step", "best_validation_loss", "bad_evaluations", "eligible"):
        _require(convergence[key] == stopped[key], f"stopped convergence mismatch: {key}")
    best_state = meta.get("best_convergence")
    _require(isinstance(best_state, dict), "missing distinct restored-best convergence state")
    _require(best_state["best_step"] == meta["restored_best_step"], "best step mismatch")
    _require(best_state["bad_evaluations"] == 0, "restored best must be an improving evaluation")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    resume = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    for name, state in (("best", best), ("resume", resume)):
        _require(state["format"] == CHECKPOINT_FORMAT, f"wrong {name} checkpoint format")
        _require(state["step"] == meta["restored_best_step"], f"wrong {name} step")
        _require(state["n_unfrozen"] == 20, f"wrong {name} unfreeze state")
        _require(state["run_identity"] == meta["run_identity"], f"wrong {name} identity")
        _require(state["model_tensor_sha256"] == {
            part: tensor_state_sha256(state[part]) for part in ("encoder", "head", "residual")
        }, f"{name} tensor hash mismatch")
    _require(best["checkpoint_phase"] == "best_eligible", "wrong best phase")
    _require(resume["checkpoint_phase"] == "restored_best_eligible", "resume is not restored best")
    _require(resume["terminal_status"] == "converged", "wrong resume terminal status")
    _require(resume["convergence"] == best_state, "resume best-state metadata mismatch")
    _require(best["convergence"] == best_state, "best convergence metadata mismatch")
    _require(best["model_tensor_sha256"] == resume["model_tensor_sha256"], "best/resume tensors differ")
    _require(best["validation_loss"] == stopped["best_validation_loss"], "best validation loss mismatch")
    _require(sha256_file(directory / "best.pt") == meta["best_checkpoint_sha256"], "best SHA256 mismatch")
    _require(sha256_file(directory / "resume.pt") == meta["resume_checkpoint_sha256"], "resume SHA256 mismatch")
    return {
        "arm": meta["arm"], "status": "PASS_CONVERGED_CONTRACT",
        "best_step": meta["restored_best_step"], "stopped_step": meta["training_stopped_step"],
        "best_validation_loss": stopped["best_validation_loss"],
    }


def audit_resource_evidence(evidence):
    if evidence is None:
        return "resource:worst_case_B4_L500_terminal20_not_measured"
    required = {
        "status": "PASS_B4_L500_TERMINAL20_RESOURCE_GATE",
        "batch_size": 4, "max_sequence_length": 500, "terminal_unfrozen_blocks": 20,
    }
    for key, expected in required.items():
        _require(evidence.get(key) == expected, f"resource identity mismatch: {key}")
    peak = float(evidence["measured_peak_cuda_bytes"])
    free = float(evidence["free_cuda_bytes_at_launch"])
    total = float(evidence["total_cuda_bytes"])
    reserve = max(4 * 1024**3, 0.2 * total)
    _require(peak > 0 and free >= 1.5 * peak + reserve, "resource headroom gate failed")
    return None


def audit_full_contract(margin_root: str | Path, numerical: dict, smoke: dict,
                        resource=None, structural_evaluation=None):
    margin_root = Path(margin_root)
    arms = {}
    blockers = []
    for arm in ARMS:
        directory = margin_root / arm
        if not (directory / "run_meta.json").is_file():
            arms[arm] = {"status": "NOT_RUN"}
            blockers.append(f"margin:{arm}:no_converged_run")
            continue
        try:
            arms[arm] = audit_converged_margin_arm(directory)
        except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
            arms[arm] = {"status": "FAIL", "error": str(error)}
            blockers.append(f"margin:{arm}:{error}")
    if smoke.get("status") != "PASS_TWO_STEP_ENGINEERING_SMOKE_ONLY":
        blockers.append("smoke:not_pass")
    try:
        resource_blocker = audit_resource_evidence(resource)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        resource_blocker = f"resource:{error}"
    if resource_blocker:
        blockers.append(resource_blocker)
    if numerical.get("status") != "PASS":
        blockers.append("numerical:independent_exhaustive_failed")
    if (not isinstance(structural_evaluation, dict)
            or structural_evaluation.get("status") != "PASS_GEC_SEED0_INDEPENDENT_REEVAL"
            or structural_evaluation.get("qualification_pass") is not True
            or structural_evaluation.get("overall_pass") is not True):
        blockers.append("science:fixed_MEA_independent_structural_evaluation_not_passed")
    return {
        "status": "PASS_FULL_CONTRACT" if not blockers else "FAIL_FULL_CONTRACT",
        "margin_arms": arms,
        "historical_nll_arms": {
            arm: "REFERENCE_ONLY_CHECKPOINT_QUALIFICATION_AND_SCIENCE_GATES_FAILED" for arm in ARMS
        },
        "blockers": blockers,
        "scientific_training_authorized": False,
        "full_training_started_by_review": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_convergence_v1/generation_01"))
    parser.add_argument("--smoke-summary", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_smoke_v1/generation_01/smoke_verification_v1.json"))
    parser.add_argument("--nll-reeval", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_independent_reeval_v1/independent_reeval.json"))
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("--structural-evaluation", type=Path)
    args = parser.parse_args()
    numerical = run_independent_exhaustive()
    smoke = json.loads(args.smoke_summary.read_text())
    nll = json.loads(args.nll_reeval.read_text())
    report = {
        "schema_version": "foldnar.structured_margin_independent_contract_review.v1",
        "base_commit": "92f0bc2e6440bcbfc682d2f8ca3e502ec86a6b48",
        "numerical": numerical,
        "nll_reference": {
            "status": nll["status"], "qualification_pass": nll["qualification_pass"],
            "overall_pass": nll["overall_pass"],
            "failed_qualification_gates": nll["failed_qualification_gates"],
        },
        "full_contract": audit_full_contract(
            args.margin_root, numerical, smoke,
            resource=json.loads(args.resource_evidence.read_text()) if args.resource_evidence else None,
            structural_evaluation=(json.loads(args.structural_evaluation.read_text())
                                   if args.structural_evaluation else None),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "numerical": numerical["status"], "cases": numerical["sequences"],
        "full_contract": report["full_contract"]["status"], "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
