#!/usr/bin/env python3
"""Independently verify Phase-0 contract, checkpoint and backbone identities.

This verifier does not compute the scientific matrix and never authorizes an
external evaluation. It reconstructs the seed-0 contract from the frozen file,
then replays every terminal checkpoint receipt from the actual tensor bytes.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

SEEDS = (0, 1, 2)
ARMS = ("train_on", "train_off")
TERMINAL_STEP = 9600
EXPECTED_UNFROZEN = 20
PILOT_SEED_LINE = '    "seeds": [0, 1, 2],\n'
PILOT_SEED_ANCHOR = '    "seed": 0,\n' + PILOT_SEED_LINE + '    "steps": 9600,'
LEGACY_PILOT_ANCHOR = '    "seed": 0,\n    "steps": 9600,'
SCIENTIFIC_SECTIONS = (
    "schema_version", "contract_id", "status", "research_question", "scope",
    "shared_initialization", "training", "training_trajectories",
    "evaluation_identities", "decode_selection", "metrics", "full_matrix_gate",
    "external_policy", "artifact_root", "code_root",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def reconstruct_seed0_contract(raw: bytes) -> bytes:
    """Remove exactly the later pilot whitelist; all scientific bytes stay fixed."""
    text = raw.decode("utf-8")
    if text.count(PILOT_SEED_ANCHOR) != 1:
        raise ValueError("frozen contract does not contain exactly one pilot seed whitelist")
    return text.replace(PILOT_SEED_ANCHOR, LEGACY_PILOT_ANCHOR, 1).encode("utf-8")


def scientific_projection(contract: dict) -> dict:
    return {key: copy.deepcopy(contract[key]) for key in SCIENTIFIC_SECTIONS}


def encoder_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Reimplementation of the receipt semantics, independent of training code."""
    digest = hashlib.sha256()
    if not state:
        raise ValueError("empty encoder state")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_identities(artifact_root: Path, contract_path: Path,
                      train_script_path: Path, verify_tensors: bool = True) -> dict:
    raw_contract = contract_path.read_bytes()
    legacy_contract = reconstruct_seed0_contract(raw_contract)
    current_sha = sha256_bytes(raw_contract)
    legacy_sha = sha256_bytes(legacy_contract)
    current_contract = json.loads(raw_contract)
    seed0_contract = json.loads(legacy_contract)
    require(scientific_projection(current_contract) == scientific_projection(seed0_contract),
            "seed-0 and frozen contracts differ outside pilot.seeds")
    seed0_pilot = seed0_contract["pilot"]
    current_pilot = copy.deepcopy(current_contract["pilot"])
    removed_whitelist = current_pilot.pop("seeds")
    require(removed_whitelist == [0, 1, 2], "unexpected frozen pilot seed whitelist")
    require(seed0_pilot == current_pilot, "pilot contracts differ beyond the seed whitelist")

    train_script_sha = sha256_file(train_script_path)
    seeds = []
    all_checkpoint_hashes = []
    all_backbone_hashes = []
    all_shared_inits = []
    all_train_script_hashes = []
    common_schedule = None
    common_backbone_report = None

    for seed in SEEDS:
        stage = artifact_root / f"seed{seed}_backbone_lr_1e-6"
        expected_contract_sha = legacy_sha if seed == 0 else current_sha
        verdict = load_json(stage / "optimizer_pilot_verdict.json")
        scientific = load_json(stage / "optimizer_scientific_verification.json")
        frozen = load_json(stage / "decode_vl0/frozen_decode_config.json")
        summary = load_json(stage / "decode_vl0/vl0_selected_summary.json")
        require(verdict["seed"] == summary["seed"] == seed, f"seed identity mismatch for seed {seed}")
        require(verdict["status"] == "COMPLETED", f"pilot verdict incomplete for seed {seed}")
        require(scientific["status"] == "PASS" and scientific.get("stability_pass") is True,
                f"scientific stability did not pass for seed {seed}")
        require(all(scientific.get("checks", {}).values()),
                f"scientific verifier has a failed check for seed {seed}")
        require(summary["terminal_step"] == TERMINAL_STEP and summary["n_sequences"] == 196,
                f"terminal VL0 summary mismatch for seed {seed}")
        require(not verdict["old_external_panels_used"] and not verdict["production_repair"]
                and not verdict["sota_claim"], f"scope violation for seed {seed}")
        require(verdict["train_sha256"] == train_script_sha,
                f"training script hash differs from current frozen script for seed {seed}")
        all_train_script_hashes.append(verdict["train_sha256"])

        seed_record = {
            "seed": seed,
            "expected_contract_sha256": expected_contract_sha,
            "contract_provenance": "frozen_contract_minus_pilot.seeds" if seed == 0 else "frozen_contract",
            "scientific_verification_sha256": sha256_file(stage / "optimizer_scientific_verification.json"),
            "arms": {},
        }
        arm_shared_inits = []
        for arm in ARMS:
            run = load_json(stage / arm / "run_meta.json")
            export = load_json(stage / f"export_{arm}/export_metadata.json")
            receipt_path = stage / "snapshots" / f"{arm}_step{TERMINAL_STEP}.pt.json"
            checkpoint_path = stage / "snapshots" / f"{arm}_step{TERMINAL_STEP}.pt"
            receipt = load_json(receipt_path)
            require(run["status"] == "completed" and run["stop_reason"] == "step_cap"
                    and run["steps_completed"] == TERMINAL_STEP and not run["early_stopped"],
                    f"invalid stopping reason for seed {seed} {arm}")
            require(not run["divergence_reason"], f"divergence recorded for seed {seed} {arm}")
            require(run["seed"] == export["seed"] == seed,
                    f"run/export seed mismatch for seed {seed} {arm}")
            require(run["arm"] == export["train_trajectory"] == arm,
                    f"run/export arm mismatch for seed {seed} {arm}")
            for source_name, source_sha in (
                ("run_meta", run["prior_alignment"]["contract_sha256"]),
                ("export", export["contract_sha256"]),
                ("frozen_decode", frozen["contract_sha256"]),
                ("decode_summary", summary["contract_sha256"]),
            ):
                require(source_sha == expected_contract_sha,
                        f"{source_name} contract mismatch for seed {seed} {arm}")
            shared_init = run["prior_alignment"]["shared_init_sha256"]
            require(export["shared_init_sha256"] == summary["shared_init_sha256"]
                    == verdict["shared_init_sha256"] == shared_init,
                    f"shared-init mismatch for seed {seed} {arm}")
            arm_shared_inits.append(shared_init)
            require(run["prior_alignment"]["training_trajectory"] == arm,
                    f"training trajectory mismatch for seed {seed} {arm}")
            expected_schedule = {
                "steps": TERMINAL_STEP,
                "batch_size": current_contract["training"]["batch_size"],
                "head_lr": current_contract["training"]["head_lr"],
                "backbone_lr": current_contract["training"]["backbone_lr"],
                "warmup_head_steps": current_contract["training"]["head_warmup_steps"],
                "unfreeze_every": current_contract["training"]["unfreeze_every"],
                "unfreeze_per_step": current_contract["training"]["unfreeze_per_step"],
                "n_blocks": current_contract["training"]["n_backbone_blocks"],
            }
            for key, value in expected_schedule.items():
                require(run["schedule"][key] == value,
                        f"schedule mismatch {key} for seed {seed} {arm}")
            if common_schedule is None:
                common_schedule = run["schedule"]
                common_backbone_report = run["backbone_report"]
            else:
                require(run["schedule"] == common_schedule, f"cross-seed schedule mismatch for seed {seed}")
                require(run["backbone_report"] == common_backbone_report,
                        f"cross-seed backbone architecture mismatch for seed {seed}")

            require(receipt["step"] == TERMINAL_STEP
                    and receipt["actual_unfrozen_blocks"] == EXPECTED_UNFROZEN,
                    f"terminal receipt mismatch for seed {seed} {arm}")
            require(Path(receipt["checkpoint_path"]) == checkpoint_path,
                    f"checkpoint path substitution for seed {seed} {arm}")
            checkpoint_sha = sha256_file(checkpoint_path)
            require(checkpoint_sha == receipt["checkpoint_sha256"]
                    == export["checkpoint"]["sha256"],
                    f"checkpoint file hash mismatch for seed {seed} {arm}")
            require(Path(export["checkpoint"]["path"]) == checkpoint_path,
                    f"export checkpoint path mismatch for seed {seed} {arm}")
            require(export["checkpoint_step"] == TERMINAL_STEP
                    and export["n_selected"] == export["n_exported"] == 196,
                    f"export completeness mismatch for seed {seed} {arm}")

            arm_record = {
                "run_meta_sha256": sha256_file(stage / arm / "run_meta.json"),
                "export_metadata_sha256": sha256_file(stage / f"export_{arm}/export_metadata.json"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "receipt_sha256": sha256_file(receipt_path),
                "receipt_backbone_update_sha256": receipt["backbone_update_sha256"],
                "actual_unfrozen_blocks": receipt["actual_unfrozen_blocks"],
                "stop_reason": run["stop_reason"],
                "steps_completed": run["steps_completed"],
                "shared_init_sha256": shared_init,
            }
            if verify_tensors:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                require(payload["step"] == TERMINAL_STEP,
                        f"checkpoint payload step mismatch for seed {seed} {arm}")
                require(payload["meta"]["prior_alignment"]["contract_sha256"] == expected_contract_sha,
                        f"checkpoint payload contract mismatch for seed {seed} {arm}")
                payload_receipt = payload["prior_alignment_checkpoint"]
                require(payload_receipt["backbone_update_sha256"] == receipt["backbone_update_sha256"]
                        and payload_receipt["actual_unfrozen_blocks"] == EXPECTED_UNFROZEN,
                        f"payload/receipt mismatch for seed {seed} {arm}")
                recomputed = encoder_state_sha256(payload["model"]["encoder"])
                require(recomputed == receipt["backbone_update_sha256"],
                        f"encoder tensor hash mismatch for seed {seed} {arm}")
                arm_record["recomputed_backbone_update_sha256"] = recomputed
                arm_record["encoder_tensor_count"] = len(payload["model"]["encoder"])
                del payload
                gc.collect()
            all_checkpoint_hashes.append(checkpoint_sha)
            all_backbone_hashes.append(receipt["backbone_update_sha256"])
            seed_record["arms"][arm] = arm_record
        require(len(set(arm_shared_inits)) == 1, f"arms do not share one initialization for seed {seed}")
        require(seed_record["arms"]["train_on"]["receipt_backbone_update_sha256"]
                != seed_record["arms"]["train_off"]["receipt_backbone_update_sha256"],
                f"training arms have identical terminal backbone tensors for seed {seed}")
        seed_record["shared_init_sha256"] = arm_shared_inits[0]
        all_shared_inits.append(arm_shared_inits[0])
        seeds.append(seed_record)

    require(len(set(all_shared_inits)) == 3, "seed initializations are not distinct")
    require(len(set(all_checkpoint_hashes)) == 6, "terminal checkpoint files are not six distinct identities")
    require(len(set(all_backbone_hashes)) == 6, "terminal backbone tensors are not six distinct identities")
    require(len(set(all_train_script_hashes)) == 1, "training script identity differs across seeds")
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "PHASE_0_THREE_SEED_CONTRACT_AND_TERMINAL_CHECKPOINT_IDENTITIES",
        "contract_reconstruction": {
            "frozen_contract_path": str(contract_path),
            "frozen_contract_sha256": current_sha,
            "seed0_contract_sha256": legacy_sha,
            "removed_pilot_whitelist": [0, 1, 2],
            "scientific_sections_identical": True,
            "other_pilot_fields_identical": True,
        },
        "train_script_path": str(train_script_path),
        "train_script_sha256": train_script_sha,
        "terminal_step": TERMINAL_STEP,
        "expected_unfrozen_blocks": EXPECTED_UNFROZEN,
        "checkpoint_file_sha256_replayed": True,
        "backbone_tensor_sha256_replayed": verify_tensors,
        "all_six_checkpoint_files_distinct": True,
        "all_six_terminal_backbones_distinct": True,
        "seeds": seeds,
        "matrix_computed_here": False,
        "external_evaluation_allowed": False,
        "production_repair": False,
        "sota_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-tensor-replay", action="store_true",
                        help="metadata-only debug mode; never substitutes for the final PASS")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing verification: {args.out}")
    result = verify_identities(args.artifact_root, args.contract, args.train_script,
                               verify_tensors=not args.skip_tensor_replay)
    if args.skip_tensor_replay:
        result["status"] = "PARTIAL_METADATA_ONLY"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({key: result[key] for key in (
        "status", "terminal_step", "checkpoint_file_sha256_replayed",
        "backbone_tensor_sha256_replayed", "all_six_checkpoint_files_distinct",
        "all_six_terminal_backbones_distinct")}, indent=2))


if __name__ == "__main__":
    main()
