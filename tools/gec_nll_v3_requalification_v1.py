#!/usr/bin/env python3
"""Requalify frozen NLL v3 arms from raw ledgers without changing science gates."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.gec_seed0_independent_reeval_v1 import (
    ARMS,
    audit_training_log,
    bind_predictions,
    canonical_hash,
    checkpoint_content,
    identity,
    load_checkpoint,
    load_frozen_panel,
    read_jsonl,
    reconstruct_validation,
    sha256,
    tensor_state_sha256,
)

SCHEMA = "foldnar.nll_v3_requalification.v1"
BASE_COMMIT = "e3aa3d9062938fad6f315692ce3112fc37d0fa9c"
BRANCH = "nll-v3-requalification-20261126"
EXPECTED_DECODER = {
    "name": "MEA", "temperature": 1.0, "gamma": 1.0,
    "score_variant": "train_off/plana_off", "search_allowed": False,
}


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def check(checks: dict, name: str, passed: bool, detail: object = None) -> None:
    checks[name] = {"pass": bool(passed)}
    if not passed:
        checks[name]["detail"] = detail


def checks_pass(checks: dict) -> bool:
    return all(row["pass"] for row in checks.values())


def run_meta_defect(run_meta: dict, resume_convergence: dict, expected: dict) -> dict:
    observed = run_meta.get("convergence")
    fields = {}
    for key in sorted(set(expected) | set(observed or {})):
        if (observed or {}).get(key) != expected.get(key):
            fields[key] = {"observed": (observed or {}).get(key), "ledger_expected": expected.get(key)}
    sole = set(fields) == {"bad_evaluations"} and fields["bad_evaluations"]["observed"] == 0
    return {
        "path": "run_meta.json:convergence.bad_evaluations",
        "mechanism": "restoring best-eligible checkpoint overwrote stopped convergence state",
        "observed_bad_evaluations": (observed or {}).get("bad_evaluations"),
        "ledger_bad_evaluations": expected["bad_evaluations"],
        "field_differences": fields,
        "resume_matches_run_meta": resume_convergence == observed,
        "sole_qualification_defect": sole and resume_convergence == observed,
    }


def audit_checkpoint_pair(arm_dir: Path, arm: str, reconstruction: dict,
                          run_meta: dict, config_sha256: str, source_sha256: str) -> dict:
    checks = {}
    best_path, resume_path = arm_dir / "best.pt", arm_dir / "resume.pt"
    best, errors = load_checkpoint(best_path, f"{arm} best")
    check(checks, "best_deserializes_on_cpu", not errors, errors)
    best_hashes = {}
    run_identity = None
    if best is not None:
        best_hashes, content_errors = checkpoint_content(best, f"{arm} best")
        check(checks, "best_tensor_content_hashes", not content_errors, content_errors)
        expected_best = dict(reconstruction["final_convergence"], bad_evaluations=0)
        expected_fields = {
            "format": "foldnar.gec_checkpoint.v2", "checkpoint_phase": "best_eligible",
            "step": reconstruction["best_step"], "optimizer_steps": reconstruction["best_step"],
            "n_unfrozen": 20, "convergence": expected_best,
            "validation_loss": reconstruction["best_validation_loss"],
        }
        for key, expected in expected_fields.items():
            check(checks, f"best_{key}", best.get(key) == expected,
                  {"observed": best.get(key), "expected": expected})
        selected = best.get("model")
        selected_ok = isinstance(selected, dict) and set(selected) == {"encoder", "head", "residual"}
        if selected_ok:
            selected_ok = all(tensor_state_sha256(selected[name]) == best_hashes.get(name)
                              for name in ("encoder", "head", "residual"))
        check(checks, "best_selected_model_matches_tensors", selected_ok)
        run_identity = best.get("run_identity")
        check(checks, "best_run_identity_present", isinstance(run_identity, dict))
        if isinstance(run_identity, dict):
            check(checks, "best_arm_identity", run_identity.get("arm") == arm)
            check(checks, "best_config_identity", run_identity.get("gec_config_sha256") == config_sha256)
            check(checks, "best_source_contract_identity",
                  run_identity.get("source_contract_sha256") == source_sha256)
            check(checks, "best_not_engineering_smoke", run_identity.get("engineering_smoke") is False)
            check(checks, "best_terminal_unfreeze", run_identity.get("terminal_unfrozen_blocks") == 20)
    resume, errors = load_checkpoint(resume_path, f"{arm} resume")
    check(checks, "resume_deserializes_on_cpu", not errors, errors)
    resume_hashes = {}
    resume_convergence = None
    if resume is not None:
        resume_hashes, content_errors = checkpoint_content(resume, f"{arm} resume")
        check(checks, "resume_tensor_content_hashes", not content_errors, content_errors)
        resume_convergence = resume.get("convergence")
        expected_resume = {
            "format": "foldnar.gec_checkpoint.v2", "checkpoint_phase": "restored_best_eligible",
            "terminal_status": "converged", "step": reconstruction["best_step"],
            "optimizer_steps": reconstruction["best_step"], "n_unfrozen": 20,
        }
        for key, expected in expected_resume.items():
            check(checks, f"resume_{key}", resume.get(key) == expected,
                  {"observed": resume.get(key), "expected": expected})
        check(checks, "best_resume_model_tensor_identity", best_hashes == resume_hashes,
              {"best": best_hashes, "resume": resume_hashes})
        check(checks, "best_resume_run_identity", resume.get("run_identity") == run_identity)
        selected = resume.get("model")
        if selected is not None:
            selected_ok = isinstance(selected, dict) and set(selected) == {"encoder", "head", "residual"}
            if selected_ok:
                selected_ok = all(tensor_state_sha256(selected[name]) == resume_hashes.get(name)
                                  for name in ("encoder", "head", "residual"))
            check(checks, "resume_selected_model_matches_tensors", selected_ok)
    defect = run_meta_defect(run_meta, resume_convergence, reconstruction["final_convergence"])
    check(checks, "run_meta_sole_defect_is_restored_bad_counter", defect["sole_qualification_defect"], defect)
    expected_meta = {
        "schema_version": "foldnar.gec_training_run.v2", "arm": arm,
        "status": "converged", "converged": True, "steps_completed": reconstruction["stop_step"],
        "training_stopped_step": reconstruction["stop_step"],
        "training_stopped_optimizer_steps": reconstruction["stop_step"],
        "optimizer_steps": reconstruction["best_step"], "restored_best_step": reconstruction["best_step"],
        "final_unfrozen_blocks": 20, "stop_reason": "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE",
        "best_checkpoint_sha256": identity(best_path)["sha256"],
        "resume_checkpoint_sha256": identity(resume_path)["sha256"],
        "final_residual_tensor_sha256": best_hashes.get("residual"),
        "run_identity": run_identity,
    }
    for key, expected in expected_meta.items():
        check(checks, f"run_meta_{key}", run_meta.get(key) == expected,
              {"observed": run_meta.get(key), "expected": expected} if key != "run_identity" else "mismatch")
    result = {
        "best_checkpoint": identity(best_path), "resume_checkpoint": identity(resume_path),
        "best_tensor_sha256": best_hashes, "resume_tensor_sha256": resume_hashes,
        "run_identity": run_identity, "run_meta_defect": defect,
        "checks": checks, "pass": checks_pass(checks),
    }
    del best, resume
    return result


def audit_prediction_export(arm_dir: Path, arm: str, best_sha256: str,
                            prediction_sha256: str, record_count: int) -> dict:
    export = json.loads((arm_dir / "prediction_export.json").read_text())
    expected = {
        "schema_version": "foldnar.gec_seed0_prediction_export.v1", "arm": arm,
        "panel_id": "tr1c_cluster80_dev_v1", "record_count": record_count,
        "best_checkpoint_sha256": best_sha256, "prediction_sha256": prediction_sha256,
        "decoder": EXPECTED_DECODER, "external_test_sets": False,
    }
    checks = {}
    for key, value in expected.items():
        check(checks, key, export.get(key) == value, {"observed": export.get(key), "expected": value})
    return {"path": str(arm_dir / "prediction_export.json"), "checks": checks,
            "pass": checks_pass(checks), "export": export}


def audit_historical_code(run_identity: dict, launcher: dict) -> dict:
    checks = {}
    command = launcher.get("command", [])
    script = Path(command[1]) if len(command) > 1 else Path("missing")
    source_root = script.parent.parent
    manifest = run_identity.get("code_manifest", {})
    observed_manifest = {}
    for relative, recorded in sorted(manifest.items()):
        path = source_root / relative
        try:
            actual = identity(path)
            observed_manifest[relative] = actual
            check(checks, f"file:{relative}",
                  actual["bytes"] == recorded["bytes"] and actual["sha256"] == recorded["sha256"])
        except OSError as error:
            check(checks, f"file:{relative}", False, str(error))
    check(checks, "manifest_digest", run_identity.get("code_manifest_sha256") == canonical_hash(manifest))
    check(checks, "launcher_status", launcher.get("status") == "exported" and launcher.get("returncode") == 0)
    check(checks, "launcher_arm", "--arm" in command and command[command.index("--arm") + 1] == run_identity.get("arm"))
    check(checks, "launcher_seed_zero", "--seed" in command and command[command.index("--seed") + 1] == "0")
    check(checks, "historical_training_script", script.as_posix().endswith("tools/train_plan_a_gec_convergence_v1.py"))
    return {"source_root": str(source_root), "files": observed_manifest,
            "checks": checks, "pass": checks_pass(checks)}


def unchanged_identities(inputs: dict) -> bool:
    for recorded in inputs.values():
        path = Path(recorded["path"])
        if not path.is_file():
            return False
        actual = identity(path)
        if actual["bytes"] != recorded["bytes"] or actual["sha256"] != recorded["sha256"]:
            return False
    return True


def run(repo: Path, artifact_root: Path, output_root: Path) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=repo).returncode:
        raise ValueError("requalification HEAD does not descend from required integrated base")
    config_path = repo / "configs/gec_four_arm_convergence_v1.json"
    config = json.loads(config_path.read_text())
    config_sha256, source_sha256 = sha256(config_path), sha256(config["source_contract"])
    frozen, registry, config, inputs = load_frozen_panel(repo, config_path)
    arm_reports = {}
    for arm in ARMS:
        print(f"[requalify] {arm}: raw validation ledger", flush=True)
        arm_dir = artifact_root / arm
        validation_path = arm_dir / "validation_log.jsonl"
        reconstruction = reconstruct_validation(read_jsonl(validation_path))
        validation_checks = {}
        check(validation_checks, "raw_ledger_internally_consistent", not reconstruction["mismatches"],
              reconstruction["mismatches"])
        check(validation_checks, "stop_13000_best_10500_bad_5",
              reconstruction["stop_step"] == 13000 and reconstruction["best_step"] == 10500
              and reconstruction["bad_evaluations"] == 5 and reconstruction["stop"] is True)
        training = audit_training_log(arm_dir / "train_log.jsonl", reconstruction["stop_step"])
        check(validation_checks, "contiguous_13000_step_training_ledger", training["pass"], training["errors"])
        run_meta = json.loads((arm_dir / "run_meta.json").read_text())
        print(f"[requalify] {arm}: CPU checkpoint tensor comparison", flush=True)
        checkpoint = audit_checkpoint_pair(arm_dir, arm, reconstruction, run_meta, config_sha256, source_sha256)
        launcher = json.loads((arm_dir / "launcher_status.json").read_text())
        code = audit_historical_code(checkpoint["run_identity"], launcher)
        prediction_path = arm_dir / "dev_predictions.jsonl"
        prediction_export = audit_prediction_export(
            arm_dir, arm, checkpoint["best_checkpoint"]["sha256"], sha256(prediction_path), len(frozen),
        )
        dev_ids = (arm_dir / "dev_predictions.jsonl",)
        expected_validation_ids = set(Path(config["assets"]["validation_ids"]).read_text().splitlines())
        data_checks = {}
        check(data_checks, "run_meta_validation_ids_exact",
              set(run_meta.get("validation_record_ids", [])) == expected_validation_ids
              and len(run_meta.get("validation_record_ids", [])) == len(expected_validation_ids))
        check(data_checks, "run_meta_raw_data_hash", run_meta.get("train_data_sha256") == inputs["raw_tr1c"]["sha256"])
        check(data_checks, "run_meta_train_ids_hash", run_meta.get("train_ids_sha256") == inputs["train_ids"]["sha256"])
        check(data_checks, "run_meta_validation_ids_hash", run_meta.get("validation_ids_sha256") == inputs["dev_ids"]["sha256"])
        check(data_checks, "run_meta_registry_hash",
              run_meta.get("model_dev_registry_sha256") == inputs["registry"]["sha256"])
        arm_reports[arm] = {
            "validation": {"stop_step": reconstruction["stop_step"], "best_step": reconstruction["best_step"],
                           "best_validation_loss": reconstruction["best_validation_loss"],
                           "bad_evaluations": reconstruction["bad_evaluations"],
                           "training_log": training, "checks": validation_checks,
                           "pass": checks_pass(validation_checks)},
            "checkpoint_and_run_meta": checkpoint, "historical_code": code,
            "prediction_export": prediction_export,
            "data_identity": {"checks": data_checks, "pass": checks_pass(data_checks)},
        }
        inputs.update({
            f"{arm}:validation_log": identity(validation_path), f"{arm}:train_log": identity(arm_dir / "train_log.jsonl"),
            f"{arm}:best_checkpoint": checkpoint["best_checkpoint"], f"{arm}:resume_checkpoint": checkpoint["resume_checkpoint"],
            f"{arm}:dev_predictions": identity(prediction_path), f"{arm}:prediction_export": identity(arm_dir / "prediction_export.json"),
            f"{arm}:run_meta": identity(arm_dir / "run_meta.json"), f"{arm}:launcher_status": identity(arm_dir / "launcher_status.json"),
        })
        for relative, file_identity in code["files"].items():
            inputs[f"{arm}:historical_code:{relative}"] = file_identity
    common_identities = []
    for arm in ARMS:
        value = dict(arm_reports[arm]["checkpoint_and_run_meta"]["run_identity"])
        value.pop("arm", None)
        common_identities.append(canonical_hash(value))
    for arm in ARMS:
        row = arm_reports[arm]["historical_code"]
        check(row["checks"], "common_four_arm_run_identity", len(set(common_identities)) == 1)
        row["pass"] = checks_pass(row["checks"])
    print("[requalify] binding every prediction to frozen truth and best checkpoint", flush=True)
    prediction_paths = {arm: artifact_root / arm / "dev_predictions.jsonl" for arm in ARMS}
    checkpoint_identities = {arm: arm_reports[arm]["checkpoint_and_run_meta"] for arm in ARMS}
    bound, prediction_identities = bind_predictions(frozen, prediction_paths, checkpoint_identities)
    for arm in ARMS:
        arm_reports[arm]["prediction_binding"] = {
            "record_count": len(bound[arm]), "exact_coverage": True,
            "per_record_best_checkpoint_sha256": True, "frozen_sequence_and_truth": True, "pass": True,
        }
        arm_reports[arm]["qualification_pass"] = all(arm_reports[arm][name]["pass"] for name in (
            "validation", "checkpoint_and_run_meta", "historical_code", "prediction_export",
            "data_identity", "prediction_binding",
        ))
    shared_init = identity(config["assets"]["shared_init"])
    if shared_init["sha256"] != config["assets"]["shared_init_sha256"]:
        raise ValueError("shared initialization identity mismatch")
    inputs["shared_init"] = shared_init
    inputs["gec_shared_residual_init"] = identity(artifact_root / "gec_shared_residual_init_seed0.pt")
    qualification_pass = all(row["qualification_pass"] for row in arm_reports.values())
    source_unchanged = unchanged_identities(inputs)
    report = {
        "schema_version": SCHEMA, "qualification_pass": qualification_pass and source_unchanged,
        "source_artifact_unchanged_during_audit": source_unchanged,
        "authority": "validation_log.jsonl raw ledger; run_meta convergence is non-authoritative restored-best metadata",
        "arms": arm_reports,
        "per_arm_qualification": {arm: row["qualification_pass"] for arm, row in arm_reports.items()},
        "historical_defect": {
            "sole_qualification_defect": "run_meta.json:convergence.bad_evaluations restored as 0 instead of stopped value 5",
            "fix_commit_integrated": "6a1178eddc60f9d0675552af022770c164275d02",
            "historical_files_modified": False,
        },
        "scope": {
            "eligible_role": "qualified historical NLL four-arm reference for structure evaluation only",
            "original_scientific_fail_unchanged": True, "scientific_gate_recomputed": False,
            "margin_superiority_claim_allowed": False,
            "margin_superiority_requires": "recompute NLL and margin under the same MEA contract",
        },
        "execution": {"device": "cpu", "checkpoint_map_location": "cpu", "model_inference": False,
                      "training_started": False, "gpu_validation_required": False,
                      "python": platform.python_version(), "torch": torch.__version__},
        "code_identity": {
            "required_base_commit": BASE_COMMIT, "branch": BRANCH,
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "implementation": identity(Path(__file__).resolve()),
            "reused_independent_primitives": identity(REPO / "tools/gec_seed0_independent_reeval_v1.py"),
        },
        "panel": {"panel_id": "tr1c_cluster80_dev_v1", "n_sequences": len(frozen),
                  "n_pairs": sum(len(row.true_pairs) for row in frozen), "external_test_sets": False},
    }
    write_json(output_root / "input_manifest.json", {"schema_version": SCHEMA, "inputs": inputs})
    write_json(output_root / "requalification.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.repo.resolve(), args.artifact_root, args.output_root)
        print(json.dumps({"qualification_pass": report["qualification_pass"],
                          "per_arm": report["per_arm_qualification"]}, indent=2, sort_keys=True))
        return 0 if report["qualification_pass"] else 1
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
        print(f"NLL v3 requalification input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
