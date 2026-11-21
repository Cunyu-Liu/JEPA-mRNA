#!/usr/bin/env python3
"""Preflight frozen inputs and the reused full control for the RAW_SCORE matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rnajepa.prior_alignment import sha256_file
from rnajepa.raw_score_matrix import load_raw_score_contract


def _seed_path(value, seed):
    return str(value).replace("seed0_backbone_lr_1e-6", f"seed{seed}_backbone_lr_1e-6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    contract = load_raw_score_contract(args.contract)
    control = contract["arms"]["full_control"]
    data = Path(contract["training"]["data"])
    init = Path(contract["training"]["shared_init_template"].format(seed=args.seed))
    run_meta_path = Path(_seed_path(control["run_meta"], args.seed))
    checkpoint = Path(_seed_path(control["checkpoint"], args.seed))
    export = Path(_seed_path(control["export"], args.seed))
    decoder_source = Path(_seed_path(contract["evaluation"]["decoder_source"], args.seed))
    for path in (data, init, run_meta_path, checkpoint, export / "export_metadata.json", decoder_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {"data": sha256_file(data), "shared_init": sha256_file(init),
              "control_run_meta": sha256_file(run_meta_path),
              "control_checkpoint": sha256_file(checkpoint),
              "decoder_source": sha256_file(decoder_source)}
    checks = {"data_sha256": hashes["data"] == contract["training"]["data_sha256"]}
    if args.seed == 0:
        checks.update({
            "shared_init_sha256": hashes["shared_init"] == contract["training"]["shared_init_sha256_seed0"],
            "control_run_meta_sha256": hashes["control_run_meta"] == control["run_meta_sha256"],
            "decoder_source_sha256": hashes["decoder_source"] == contract["evaluation"]["decoder_source_sha256"],
        })
    run_meta = json.loads(run_meta_path.read_text())
    identity = run_meta.get("prior_alignment", {})
    checks.update({
        "control_completed": run_meta.get("status") == "completed",
        "control_step_cap": run_meta.get("stop_reason") == "step_cap" and run_meta.get("steps_completed") == 9600,
        "control_unfrozen_20_of_33": str(run_meta.get("trainable_parameter_scope", "")).startswith("20/33 "),
        "control_train_off": identity.get("train_turner") == "off",
        "control_init_path": identity.get("shared_init") == str(init),
        "control_init_recorded_sha256": identity.get("shared_init_sha256") == hashes["shared_init"],
        "control_same_objective": run_meta.get("objective", {}).get("nll_normalization") == "length",
        "control_no_auxiliary_objective": all(float(run_meta.get("objective", {}).get(key, 0.0)) == 0.0
                                               for key in ("lambda_distill", "lambda_rlcd", "lambda_cal")),
    })
    export_meta = json.loads((export / "export_metadata.json").read_text())
    checks.update({
        "control_export_complete": export_meta.get("n_selected") == 196 and export_meta.get("n_skipped_length") == 0,
        "control_export_seed": export_meta.get("seed") == args.seed,
        "control_export_step": export_meta.get("checkpoint_step") == 9600,
        "control_export_checkpoint_sha256": export_meta.get("checkpoint", {}).get("sha256") == hashes["control_checkpoint"],
        "control_export_train_off": export_meta.get("train_trajectory") == "train_off",
        "control_export_shared_init": export_meta.get("shared_init_sha256") == hashes["shared_init"],
        "control_export_external_closed": export_meta.get("old_external_panels_used") is False,
    })
    decoder = json.loads(decoder_source.read_text())["config"]["C2"]
    expected = contract["evaluation"]["decoder"]
    checks["fixed_C2_decoder"] = all(decoder[key] == expected[key] for key in ("decoder", "temperature", "gamma"))
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "seed": args.seed,
              "contract_id": contract["contract_id"], "checks": checks, "sha256": hashes,
              "paths": {"data": str(data), "shared_init": str(init), "control_run_meta": str(run_meta_path),
                        "control_checkpoint": str(checkpoint), "control_export": str(export),
                        "decoder_source": str(decoder_source)},
              "external_evaluation_allowed": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
