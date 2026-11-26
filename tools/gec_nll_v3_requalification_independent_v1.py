#!/usr/bin/env python3
"""Second, implementation-independent CPU recheck of NLL v3 qualification."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
SCHEMA = "foldnar.nll_v3_requalification_independent_recheck.v1"
EXPECTED_DECODER = {
    "name": "MEA", "temperature": 1.0, "gamma": 1.0,
    "score_variant": "train_off/plana_off", "search_allowed": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def ledger_recheck(path: Path) -> dict:
    rows = jsonl(path)
    errors = []
    if len(rows) != 26:
        errors.append(f"row count {len(rows)} != 26")
    eligible = []
    for index, row in enumerate(rows):
        step = (index + 1) * 500
        expected_unfrozen = min(20, max(0, ((step - 1600) // 800) * 2))
        if row.get("step") != step or row.get("n_unfrozen") != expected_unfrozen:
            errors.append(f"step/unfreeze mismatch at row {index}")
        loss = row.get("validation_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            errors.append(f"invalid loss at row {index}")
        should_be_eligible = step >= 9600 and expected_unfrozen == 20
        if row.get("eligible") is not should_be_eligible:
            errors.append(f"eligibility mismatch at {step}")
        if should_be_eligible:
            eligible.append(row)
        else:
            expected = {"improved": False, "save_best": False, "stop": False,
                        "best_step": None, "best_validation_loss": None, "bad_evaluations": 0}
            if any(row.get(key) != value for key, value in expected.items()):
                errors.append(f"preterminal state mismatch at {step}")
    running_loss = math.inf
    running_step = None
    bad = 0
    expected_states = []
    for row in eligible:
        loss, step = row["validation_loss"], row["step"]
        improved = loss < running_loss - 1e-6
        if improved:
            running_loss, running_step, bad = loss, step, 0
        else:
            bad += 1
        expected_states.append((running_step, running_loss, bad, improved, bad >= 5))
    for row, (best_step, best_loss, bad, improved, stop) in zip(eligible, expected_states):
        expected = {"best_step": best_step, "best_validation_loss": best_loss,
                    "bad_evaluations": bad, "improved": improved, "save_best": improved, "stop": stop}
        if any(row.get(key) != value for key, value in expected.items()):
            errors.append(f"eligible decision mismatch at {row['step']}")
    if [row["step"] for row in eligible] != [10000, 10500, 11000, 11500, 12000, 12500, 13000]:
        errors.append("eligible step grid mismatch")
    if expected_states[-1][0] != 10500 or expected_states[-1][2] != 5 or not expected_states[-1][4]:
        errors.append("expected best/stop state not reached")
    if any(row.get("stop") for row in rows[:-1]) or rows[-1].get("stop") is not True:
        errors.append("stop must occur only at final row")
    convergence = {
        "terminal_step": 9600, "terminal_unfrozen_blocks": 20, "patience": 5, "min_delta": 1e-6,
        "best_step": expected_states[-1][0], "best_validation_loss": expected_states[-1][1],
        "bad_evaluations": expected_states[-1][2], "eligible": True,
    }
    train_rows = 0
    last_step = 0
    with (path.parent / "train_log.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            train_rows += 1
            row = json.loads(line)
            last_step = row.get("step")
            if row.get("step") != train_rows or row.get("optimizer_steps") != train_rows or row.get("optimizer_step") is not True:
                errors.append(f"training ledger mismatch at {train_rows}")
    if train_rows != 13000 or last_step != 13000:
        errors.append(f"training length {train_rows} != 13000")
    return {"stop_step": rows[-1]["step"], "best_step": convergence["best_step"],
            "best_validation_loss": convergence["best_validation_loss"], "bad_evaluations": 5,
            "final_convergence": convergence, "train_rows": train_rows,
            "errors": errors, "pass": not errors}


def direct_tensor_recheck(best_path: Path, resume_path: Path, ledger: dict,
                          run_meta: dict, arm: str) -> dict:
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    errors = []
    tensor_checks = {}
    for component in ("encoder", "head", "residual"):
        left, right = best.get(component), resume.get(component)
        equal = isinstance(left, dict) and isinstance(right, dict) and bool(left) and set(left) == set(right)
        compared = 0
        if equal:
            for name in sorted(left):
                a, b = left[name], right[name]
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    equal = False
                    break
                compared += 1
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    equal = False
        tensor_checks[component] = {"equal": equal, "tensor_count": compared}
        if not equal:
            errors.append(f"{component} tensors differ")
        selected = best.get("model", {}).get(component)
        if not isinstance(selected, dict) or set(selected) != set(left or {}):
            errors.append(f"best selected {component} state mismatch")
        else:
            for name in left:
                if not torch.equal(left[name], selected[name]):
                    errors.append(f"best selected tensor mismatch: {component}.{name}")
    expected_best = dict(ledger["final_convergence"], bad_evaluations=0)
    for label, checkpoint, phase in (("best", best, "best_eligible"), ("resume", resume, "restored_best_eligible")):
        expected = {"format": "foldnar.gec_checkpoint.v2", "checkpoint_phase": phase,
                    "step": 10500, "optimizer_steps": 10500, "n_unfrozen": 20}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                errors.append(f"{label} {key} mismatch")
    if best.get("convergence") != expected_best:
        errors.append("best convergence mismatch")
    observed_resume = resume.get("convergence")
    differences = {
        key: {"observed": observed_resume.get(key), "expected": value}
        for key, value in ledger["final_convergence"].items() if observed_resume.get(key) != value
    }
    sole_defect = differences == {"bad_evaluations": {"observed": 0, "expected": 5}}
    if not sole_defect or run_meta.get("convergence") != observed_resume:
        errors.append("run_meta/resume defect is not solely restored bad counter")
    if best.get("run_identity") != resume.get("run_identity") or run_meta.get("run_identity") != best.get("run_identity"):
        errors.append("run identity mismatch")
    if best["run_identity"].get("arm") != arm:
        errors.append("arm identity mismatch")
    expected_meta = {"steps_completed": 13000, "training_stopped_step": 13000,
                     "training_stopped_optimizer_steps": 13000, "optimizer_steps": 10500,
                     "restored_best_step": 10500, "status": "converged", "converged": True}
    for key, value in expected_meta.items():
        if run_meta.get(key) != value:
            errors.append(f"run_meta {key} mismatch")
    if run_meta.get("best_checkpoint_sha256") != sha256(best_path) or run_meta.get("resume_checkpoint_sha256") != sha256(resume_path):
        errors.append("run_meta checkpoint file identity mismatch")
    return {"tensor_checks": tensor_checks, "all_model_tensors_equal": all(row["equal"] for row in tensor_checks.values()),
            "run_meta_field_differences_from_ledger": differences,
            "sole_defect_is_restored_bad_counter": sole_defect,
            "run_identity": best.get("run_identity"), "errors": errors, "pass": not errors}


def raw_panel(repo: Path) -> tuple[dict, set[str], set[str], dict]:
    config = json.loads((repo / "configs/gec_four_arm_convergence_v1.json").read_text())
    registry_path = repo / config["assets"]["registry"]
    registry = json.loads(registry_path.read_text())
    if sha256(registry_path) != config["assets"]["model_dev_registry_sha256"]:
        raise ValueError("registry hash mismatch")
    panel = next(row for row in registry["panels"] if row["panel_id"] == "tr1c_cluster80_dev_v1")
    dev_path, train_path = Path(panel["dev_ids"]["path"]), Path(panel["train_ids"]["path"])
    dev_ids, train_ids = set(dev_path.read_text().splitlines()), set(train_path.read_text().splitlines())
    if dev_ids & train_ids:
        raise ValueError("train/dev exact ID leakage")
    records_path = Path(registry["environment_contracts"]["panel_records"]["path"])
    clusters = {}
    with gzip.open(records_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                clusters[row["sequence_id"]] = int(row["cluster_id"])
    train_clusters = {clusters[row] for row in train_ids}
    if any(clusters[row] in train_clusters for row in dev_ids):
        raise ValueError("train/dev cluster leakage")
    raw_path = Path(config["assets"]["data"])
    if sha256(raw_path) != config["assets"]["data_sha256"]:
        raise ValueError("raw data hash mismatch")
    raw_records = {}
    for index, row in enumerate(jsonl(raw_path)):
        sequence_id = f"bprna_tr1c:row{index:06d}:" + row["name"]
        if sequence_id in dev_ids:
            sequence = row["seq"].upper().replace("T", "U")
            raw_records[sequence_id] = {
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "sequence_length": len(sequence),
                "true_pairs": sorted(tuple(pair) for pair in row["pairs"]),
            }
    if set(raw_records) != dev_ids:
        raise ValueError("raw/dev coverage mismatch")
    return raw_records, dev_ids, train_ids, config


def prediction_recheck(arm_dir: Path, arm: str, raw_records: dict, config: dict) -> dict:
    best_hash = sha256(arm_dir / "best.pt")
    prediction_path = arm_dir / "dev_predictions.jsonl"
    prediction_hash = sha256(prediction_path)
    export = json.loads((arm_dir / "prediction_export.json").read_text())
    errors = []
    expected_export = {"arm": arm, "panel_id": "tr1c_cluster80_dev_v1", "record_count": len(raw_records),
                       "best_checkpoint_sha256": best_hash, "prediction_sha256": prediction_hash,
                       "decoder": EXPECTED_DECODER, "external_test_sets": False}
    for key, value in expected_export.items():
        if export.get(key) != value:
            errors.append(f"export {key} mismatch")
    seen = set()
    for row in jsonl(prediction_path):
        sequence_id = row.get("sequence_id")
        if sequence_id in seen or sequence_id not in raw_records:
            errors.append(f"duplicate/out-of-panel ID {sequence_id}")
            continue
        seen.add(sequence_id)
        truth = raw_records[sequence_id]
        for key, value in truth.items():
            observed = sorted(tuple(pair) for pair in row.get(key, [])) if key == "true_pairs" else row.get(key)
            if observed != value:
                errors.append(f"frozen {key} mismatch: {sequence_id}")
        if row.get("checkpoint_sha256") != best_hash:
            errors.append(f"not bound to best: {sequence_id}")
        pairs = [tuple(pair) for pair in row.get("predicted_pairs", [])]
        if len(pairs) != len(set(pairs)) or any(len(pair) != 2 or not 0 <= pair[0] < pair[1] < truth["sequence_length"] for pair in pairs):
            errors.append(f"invalid predicted pairs: {sequence_id}")
    if seen != set(raw_records):
        errors.append("incomplete prediction coverage")
    return {"records": len(seen), "best_checkpoint_sha256": best_hash,
            "prediction_sha256": prediction_hash, "errors": errors, "pass": not errors}


def historical_code_recheck(run_identity: dict, launcher: dict) -> dict:
    command = launcher["command"]
    root = Path(command[1]).parent.parent
    errors = []
    for relative, recorded in run_identity["code_manifest"].items():
        path = root / relative
        if path.stat().st_size != recorded["bytes"] or sha256(path) != recorded["sha256"]:
            errors.append(f"historical code mismatch: {relative}")
    if launcher.get("status") != "exported" or launcher.get("returncode") != 0:
        errors.append("launcher did not export successfully")
    return {"source_root": str(root), "errors": errors, "pass": not errors}


def run(repo: Path, artifact_root: Path, primary_root: Path) -> dict:
    primary = json.loads((primary_root / "requalification.json").read_text())
    raw_records, dev_ids, train_ids, config = raw_panel(repo)
    arms = {}
    common_identity = set()
    for arm in ARMS:
        print(f"[second] {arm}: independent ledger and direct tensor equality", flush=True)
        arm_dir = artifact_root / arm
        ledger = ledger_recheck(arm_dir / "validation_log.jsonl")
        run_meta = json.loads((arm_dir / "run_meta.json").read_text())
        tensors = direct_tensor_recheck(arm_dir / "best.pt", arm_dir / "resume.pt", ledger, run_meta, arm)
        identity_without_arm = json.dumps({key: value for key, value in tensors["run_identity"].items() if key != "arm"},
                                          sort_keys=True, separators=(",", ":"))
        common_identity.add(identity_without_arm)
        launcher = json.loads((arm_dir / "launcher_status.json").read_text())
        code = historical_code_recheck(tensors["run_identity"], launcher)
        data_errors = []
        for key, expected in (("train_data_sha256", config["assets"]["data_sha256"]),
                              ("train_ids_sha256", sha256(Path(config["assets"]["train_ids"]))),
                              ("validation_ids_sha256", sha256(Path(config["assets"]["validation_ids"]))),
                              ("model_dev_registry_sha256", config["assets"]["model_dev_registry_sha256"])):
            if run_meta.get(key) != expected:
                data_errors.append(f"run_meta {key} mismatch")
        if set(run_meta.get("validation_record_ids", [])) != dev_ids:
            data_errors.append("run_meta validation IDs mismatch")
        data = {"train_dev_id_disjoint": not bool(dev_ids & train_ids), "errors": data_errors, "pass": not data_errors}
        print(f"[second] {arm}: independent raw-truth prediction binding", flush=True)
        predictions = prediction_recheck(arm_dir, arm, raw_records, config)
        arms[arm] = {"ledger": ledger, "direct_tensors_and_run_meta": tensors,
                     "historical_code": code, "data_identity": data, "predictions": predictions}
        arms[arm]["qualification_pass"] = all(row["pass"] for row in (
            ledger, tensors, code, data, predictions,
        ))
    if len(common_identity) != 1:
        for arm in ARMS:
            arms[arm]["qualification_pass"] = False
            arms[arm]["historical_code"]["errors"].append("four-arm common run identity mismatch")
            arms[arm]["historical_code"]["pass"] = False
    own_pass = all(row["qualification_pass"] for row in arms.values())
    agreement = {arm: primary["arms"][arm]["qualification_pass"] == row["qualification_pass"] for arm, row in arms.items()}
    report = {
        "schema_version": SCHEMA, "qualification_pass": own_pass,
        "primary_agreement": agreement, "primary_and_second_agree": all(agreement.values()),
        "independence": {"primary_module_imported": False, "legacy_reeval_module_imported": False,
                         "tensor_method": "direct per-name torch.equal plus dtype/shape",
                         "truth_method": "independent raw TR1c row reconstruction"},
        "per_arm": arms, "execution": {"device": "cpu", "model_inference": False, "training_started": False},
        "scope": {"original_scientific_fail_unchanged": True, "margin_superiority_claim_allowed": False},
    }
    report["pass"] = own_pass and report["primary_and_second_agree"]
    write_json(primary_root / "independent_recheck.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3"))
    parser.add_argument("--primary-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.repo.resolve(), args.artifact_root, args.primary_root)
        print(json.dumps({"pass": report["pass"], "qualification_pass": report["qualification_pass"],
                          "primary_and_second_agree": report["primary_and_second_agree"],
                          "per_arm": {arm: row["qualification_pass"] for arm, row in report["per_arm"].items()}},
                         indent=2, sort_keys=True))
        return 0 if report["pass"] else 1
    except (OSError, ValueError, KeyError, IndexError, StopIteration, json.JSONDecodeError) as error:
        print(f"independent NLL recheck input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
