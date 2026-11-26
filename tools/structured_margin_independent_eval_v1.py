#!/usr/bin/env python3
"""Independent structured-margin checkpoint audit, MEA export and four-arm evaluation.

Raw validation ledgers, CPU checkpoint contents and raw prediction JSONL are the
authorities. run_meta.json and historical aggregate reports are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from tools import gec_seed0_independent_reeval_v1 as independent

ARMS = independent.ARMS
SCHEMA = "foldnar.structured_margin_independent_eval.v1"
MARGIN_CONFIG = "gec_structured_margin_implementation_v1"
MARGIN_CHECKPOINT = "foldnar.structured_margin_checkpoint.v1"
MARGIN_IDENTITY = "foldnar.structured_margin_run_identity.v1"
MARGIN_METRIC = "length_normalized_pair_hamming_structured_margin"
EXPORT_SCHEMA = "foldnar.structured_margin_prediction_export.v1"
NLL_SCHEMA = "foldnar.nll_v3_requalified_artifact.v1"
DEFAULT_NLL_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_requalified_v1")


class TrainingInProgress(RuntimeError):
    def __init__(self, arms: dict):
        super().__init__("one or more arms have no terminal raw-ledger stop")
        self.arms = arms


def identity(path: str | Path) -> dict:
    return independent.identity(path)


def read_jsonl(path: str | Path) -> list[dict]:
    return independent.read_jsonl(path)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_margin_panel(repo: Path) -> tuple[list, dict, dict, dict, Path]:
    config_path = repo / "configs/gec_structured_margin_convergence_v1.json"
    frozen, registry, config, inputs = independent.load_frozen_panel(repo, config_path)
    training = config["training"]
    convergence = config["convergence"]
    if (config.get("config_id") != MARGIN_CONFIG or tuple(config["arms"]) != ARMS
            or config.get("scientific_training_authorized") is not True):
        raise ValueError("not the authorized structured-margin four-arm config")
    if (training["seed"] != 0 or training["lambda_nll"] != 0.0
            or training["lambda_structured_margin"] != 1.0
            or training["fixed_step_cap"] is not None):
        raise ValueError("margin seed/objective/unbounded schedule mismatch")
    if (convergence["validation_metric"] != MARGIN_METRIC
            or convergence["eligible_only_at_terminal_unfreeze"] is not True
            or convergence["restore_best_eligible"] is not True):
        raise ValueError("margin validation selection contract mismatch")
    decoder = config["model"]["decoder"]
    if decoder != {"name": "MEA", "temperature": 1.0, "gamma": 1.0,
                   "score_variant": "train_off/plana_off", "search_allowed": False}:
        raise ValueError("decoder differs from frozen MEA")
    inputs["candidate_contract"] = identity(repo / config["candidate_contract"])
    return frozen, registry, config, inputs, config_path


def audit_validation_arm(arm_dir: Path, arm: str, config: dict) -> dict:
    path = arm_dir / "validation_log.jsonl"
    result = {"arm": arm, "validation_log": str(path), "run_meta_read": False}
    if not path.is_file():
        return {**result, "status": "IN_PROGRESS", "reason": "validation ledger not created"}
    rows = read_jsonl(path)
    if not rows:
        return {**result, "status": "IN_PROGRESS", "reason": "validation ledger is empty"}
    convergence = config["convergence"]
    training = config["training"]
    reconstruction = independent.reconstruct_validation(
        rows, terminal_step=training["terminal_step"], patience=convergence["patience"],
        min_delta=convergence["min_delta"],
    )
    mismatches = list(reconstruction["mismatches"])
    for row in rows:
        if row.get("validation_metric") != MARGIN_METRIC:
            mismatches.append({"step": row.get("step"), "field": "validation_metric",
                               "expected": MARGIN_METRIC, "observed": row.get("validation_metric")})
    result.update(validation_log=identity(path),
                  reconstruction={key: value for key, value in reconstruction.items() if key != "rows"},
                  mismatches=mismatches)
    if not reconstruction["stop"]:
        return {**result, "status": "IN_PROGRESS", "reason": "raw ledger has no patience stop"}
    training_audit = independent.audit_training_log(arm_dir / "train_log.jsonl", reconstruction["stop_step"])
    passed = (not mismatches and training_audit["pass"]
              and reconstruction["best_step"] is not None
              and reconstruction["bad_evaluations"] == convergence["patience"])
    return {**result, "status": "STOPPED" if passed else "INVALID",
            "training_log": training_audit, "pass": passed}


def audit_validation(artifact_root: Path, config: dict) -> dict:
    reports = {arm: audit_validation_arm(artifact_root / arm, arm, config) for arm in ARMS}
    invalid = [arm for arm, row in reports.items() if row["status"] == "INVALID"]
    if invalid:
        raise ValueError(f"raw-ledger qualification failed: {invalid}")
    if any(row["status"] == "IN_PROGRESS" for row in reports.values()):
        raise TrainingInProgress(reports)
    return reports


def _checkpoint_errors(checkpoint: dict, label: str, reconstruction: dict,
                       run_identity: dict, best: bool) -> list[str]:
    expected_convergence = dict(reconstruction["final_convergence"], bad_evaluations=0)
    expected = {
        "format": MARGIN_CHECKPOINT,
        "checkpoint_phase": "best_eligible" if best else "restored_best_eligible",
        "step": reconstruction["best_step"],
        "optimizer_steps": reconstruction["best_step"],
        "n_unfrozen": 20,
        "convergence": expected_convergence,
        "run_identity": run_identity,
    }
    if not best:
        expected["terminal_status"] = "converged"
    errors = [f"{label}: {key} mismatch" for key, value in expected.items()
              if checkpoint.get(key) != value]
    if best and checkpoint.get("validation_loss") != reconstruction["best_validation_loss"]:
        errors.append(f"{label}: validation_loss mismatch")
    return errors


def audit_checkpoint_arm(arm_dir: Path, arm: str, reconstruction: dict,
                         config: dict, config_path: Path, frozen: list) -> dict:
    best_path, resume_path = arm_dir / "best.pt", arm_dir / "resume.pt"
    errors = []
    best, load_errors = independent.load_checkpoint(best_path, f"{arm} best")
    errors.extend(load_errors)
    best_hashes = {}
    run_identity = None
    if best is not None:
        best_hashes, content_errors = independent.checkpoint_content(best, f"{arm} best")
        errors.extend(content_errors)
        run_identity = best.get("run_identity")
        if not isinstance(run_identity, dict):
            errors.append(f"{arm} best: missing run identity")
        else:
            expected_identity = {
                "schema_version": MARGIN_IDENTITY, "arm": arm, "engineering_smoke": False,
                "gec_config_sha256": independent.sha256(config_path),
                "source_contract_sha256": independent.sha256(config["source_contract"]),
                "candidate_contract_sha256": independent.sha256(config_path.parent.parent / config["candidate_contract"]),
                "validation_metric": MARGIN_METRIC,
                "objective": config["objective"], "terminal_unfrozen_blocks": 20,
            }
            errors.extend(f"{arm} best run identity: {key} mismatch"
                          for key, value in expected_identity.items() if run_identity.get(key) != value)
            if set(run_identity.get("validation_record_ids", [])) != {row.sequence_id for row in frozen}:
                errors.append(f"{arm} best run identity: frozen dev coverage mismatch")
        errors.extend(_checkpoint_errors(best, f"{arm} best", reconstruction, run_identity, True))
        selected = best.get("model")
        if not isinstance(selected, dict) or set(selected) != {"encoder", "head", "residual"}:
            errors.append(f"{arm} best: invalid selected model")
        else:
            for component in ("encoder", "head", "residual"):
                if independent.tensor_state_sha256(selected[component]) != best_hashes.get(component):
                    errors.append(f"{arm} best: selected {component} tensor mismatch")
    del best
    resume, load_errors = independent.load_checkpoint(resume_path, f"{arm} resume")
    errors.extend(load_errors)
    resume_hashes = {}
    if resume is not None:
        resume_hashes, content_errors = independent.checkpoint_content(resume, f"{arm} resume")
        errors.extend(content_errors)
        errors.extend(_checkpoint_errors(resume, f"{arm} resume", reconstruction, run_identity, False))
        if resume_hashes != best_hashes:
            errors.append(f"{arm}: best/resume selected tensors differ")
    del resume
    return {
        "best_checkpoint": identity(best_path), "resume_checkpoint": identity(resume_path),
        "best_tensor_sha256": best_hashes, "resume_tensor_sha256": resume_hashes,
        "run_identity": run_identity,
        "common_run_identity_sha256": independent.canonical_hash(
            {key: value for key, value in (run_identity or {}).items() if key != "arm"}
        ),
        "errors": errors, "pass": not errors,
    }


def qualify(repo: Path, artifact_root: Path) -> dict:
    frozen, registry, config, inputs, config_path = load_margin_panel(repo)
    validation = audit_validation(artifact_root, config)
    checkpoints = {}
    for arm in ARMS:
        print(f"[margin-audit] CPU checkpoint identity: {arm}", flush=True)
        checkpoints[arm] = audit_checkpoint_arm(
            artifact_root / arm, arm, validation[arm]["reconstruction"], config, config_path, frozen,
        )
    if len({row["common_run_identity_sha256"] for row in checkpoints.values()}) != 1:
        for row in checkpoints.values():
            row["errors"].append("four-arm common run identity mismatch")
            row["pass"] = False
    failed = [arm for arm, row in checkpoints.items() if not row["pass"]]
    if failed:
        raise ValueError(f"checkpoint qualification failed: {failed}")
    for arm in ARMS:
        checkpoints[arm]["validation_log_sha256"] = validation[arm]["validation_log"]["sha256"]
    return {"frozen": frozen, "registry": registry, "config": config, "inputs": inputs,
            "config_path": config_path, "validation": validation, "checkpoints": checkpoints}


def load_nll_predictions(repo: Path, frozen: list, nll_root: Path) -> tuple[dict, dict, dict]:
    manifest_path = nll_root / "artifact_manifest.json"
    summary_path = nll_root / "qualification_summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("schema_version") != NLL_SCHEMA or summary.get("schema_version") != NLL_SCHEMA:
        raise ValueError("unexpected NLL requalification schema")
    for name, recorded in manifest["artifacts"].items():
        actual = identity(nll_root / name)
        if actual["bytes"] != recorded["bytes"] or actual["sha256"] != recorded["sha256"]:
            raise ValueError(f"NLL requalification artifact changed: {name}")
    if (not summary["qualification_pass"] or summary["status"] != "PASS_NLL_V3_REQUALIFIED"
            or not summary["reference_use"]["historical_nll_four_arm_structure_reference_eligible"]
            or summary["reference_use"]["margin_superior_to_nll_claim_allowed"]):
        raise ValueError("NLL artifact is not an eligible structure-only reference")
    if any(summary["per_arm"][arm]["qualification"] != "QUALIFIED_STRUCTURE_REFERENCE" for arm in ARMS):
        raise ValueError("NLL reference does not cover four qualified arms")
    input_manifest = json.loads((nll_root / "input_manifest.json").read_text())
    inputs = input_manifest["inputs"]
    prediction_paths = {}
    checkpoint_identities = {}
    for arm in ARMS:
        prediction = inputs[f"{arm}:dev_predictions"]
        actual = identity(prediction["path"])
        if actual["bytes"] != prediction["bytes"] or actual["sha256"] != prediction["sha256"]:
            raise ValueError(f"NLL raw predictions changed: {arm}")
        prediction_paths[arm] = Path(prediction["path"])
        checkpoint_identities[arm] = {"best_checkpoint": inputs[f"{arm}:best_checkpoint"]}
    nll_config = json.loads((repo / "configs/gec_four_arm_convergence_v1.json").read_text())
    margin_config = json.loads((repo / "configs/gec_structured_margin_convergence_v1.json").read_text())
    if nll_config["model"]["decoder"] != margin_config["model"]["decoder"]:
        raise ValueError("NLL and margin frozen MEA decoders differ")
    bound, prediction_identities = independent.bind_predictions(frozen, prediction_paths, checkpoint_identities)
    provenance = {
        "qualification_summary": identity(summary_path), "artifact_manifest": identity(manifest_path),
        "input_manifest": identity(nll_root / "input_manifest.json"),
        "prediction_identities": prediction_identities,
        "reference_role": "QUALIFIED_NLL_STRUCTURE_REFERENCE_ONLY",
        "validation_loss_comparison_allowed": False,
        "same_frozen_mea": True,
    }
    return bound, prediction_identities, provenance


def _nll_comparison(margin_distribution: dict, nll_distribution: dict) -> dict:
    return {
        "margin": margin_distribution["observed"], "nll": nll_distribution["observed"],
        "delta_margin_minus_nll": margin_distribution["observed"] - nll_distribution["observed"],
        "paired_ci95": independent.ci95(margin_distribution["sequence"] - nll_distribution["sequence"]),
        "family_cluster_sensitivity_ci95": independent.ci95(
            margin_distribution["family"] - nll_distribution["family"]
        ),
    }


def evaluate_bound(frozen: list, margin: dict, nll: dict) -> dict:
    margin_evaluation = independent.evaluate_metrics(frozen, margin)
    gates = independent.verify_gates(margin_evaluation)
    if len(gates["gates"]) != 16:
        raise AssertionError("frozen scientific contract no longer has exactly 16 gates")
    reference = margin["matched_control"]
    bootstrap = independent.PairedBootstrap(frozen)
    comparisons = {}
    for arm in ARMS:
        margin_arrays = independent.make_arm_arrays(margin[arm], reference)
        nll_arrays = independent.make_arm_arrays(nll[arm], reference)
        metrics = {}
        for metric in independent.GLOBAL_METRICS:
            metrics[metric] = _nll_comparison(
                independent.metric_distribution(margin_arrays, bootstrap, metric),
                independent.metric_distribution(nll_arrays, bootstrap, metric),
            )
        subgroups = {}
        for subgroup in independent.SUBGROUPS:
            subgroups[subgroup] = {}
            for statistic in ("precision", "recall", "f1"):
                metric = f"micro_pair_{statistic}"
                subgroups[subgroup][statistic] = _nll_comparison(
                    independent.metric_distribution(margin_arrays, bootstrap, metric, subgroup),
                    independent.metric_distribution(nll_arrays, bootstrap, metric, subgroup),
                )
        comparisons[arm] = {"metrics": metrics, "subgroups": subgroups}
    return {**margin_evaluation, "margin_gates": gates,
            "margin_vs_qualified_nll_same_arm": comparisons}


def bind_margin_predictions(frozen: list, prediction_root: Path, checkpoints: dict) -> tuple[dict, dict]:
    paths = {arm: prediction_root / arm / "dev_predictions.jsonl" for arm in ARMS}
    for arm in ARMS:
        export_path = prediction_root / arm / "prediction_export.json"
        export = json.loads(export_path.read_text())
        expected = {
            "schema_version": EXPORT_SCHEMA, "arm": arm, "panel_id": "tr1c_cluster80_dev_v1",
            "record_count": len(frozen), "best_checkpoint_sha256": checkpoints[arm]["best_checkpoint"]["sha256"],
            "selected_model_tensor_sha256": checkpoints[arm]["best_tensor_sha256"],
            "device": "cuda:0", "cpu_fallback": False,
            "prediction_sha256": independent.sha256(paths[arm]),
            "decoder": {"name": "MEA", "temperature": 1.0, "gamma": 1.0,
                        "score_variant": "train_off/plana_off", "search_allowed": False},
            "validation_log_sha256": checkpoints[arm].get("validation_log_sha256"),
            "external_test_sets": False,
        }
        for key, value in expected.items():
            if export.get(key) != value:
                raise ValueError(f"{arm}: prediction export {key} mismatch")
    return independent.bind_predictions(frozen, paths, checkpoints)


def export_arm(repo: Path, arm: str, qualification: dict, prediction_root: Path, device_name: str) -> dict:
    from pair_decode_common import decode_mea
    from rinalmo_preflight import VOCAB, load_encoder
    from rnajepa import harness
    from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
    from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
    from rnajepa.gec_context import GECResidualV1, gec_destroy_key_v1, gec_geometry_v1, require_explicit_cuda
    from rnajepa.gec_numeric_v1 import compute_gec_context_v1
    from tools.train_plan_a_gec_convergence_v1 import NEG_BIG, load_registered_records

    if device_name != "cuda:0" or os.environ.get("RN_CUDA_INSIDE") != "1":
        raise ValueError("prediction export requires explicit cuda:0 and RN_CUDA_INSIDE=1")
    config = qualification["config"]
    checkpoint_audit = qualification["checkpoints"][arm]
    best_path = Path(checkpoint_audit["best_checkpoint"]["path"])
    if identity(best_path) != checkpoint_audit["best_checkpoint"]:
        raise ValueError(f"{arm}: best checkpoint changed since qualification")
    output_dir = prediction_root / arm
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "dev_predictions.jsonl"
    device = require_explicit_cuda(device_name)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=config["model"]["d_z"], hidden=64,
                            scorer="resnet2d", chunk_size=0, resnet_blocks=4, resnet_hidden=64,
                            use_turner_prior=True)
    residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    residual.load_state_dict(checkpoint["residual"], strict=True)
    head.use_turner_prior = False
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    residual = residual.to(device).eval()
    records = load_registered_records(config, "validation", 0)
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records, 1):
            sequence = record["sequence"]
            length = len(sequence)
            framed = [VOCAB["<cls>"]] + [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence] + [VOCAB["<eos>"]]
            encoder_ids = torch.tensor([framed], dtype=torch.long, device=device)
            attention = torch.ones_like(encoder_ids)
            head_ids = torch.tensor([[BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]],
                                    dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = encoder(input_ids=encoder_ids, attention_mask=attention)
            hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
            hidden = hidden[:, 1:length + 1, :].float().to(torch.float16).to(torch.float32)
            chunks = [head.pair_repr.cross(hidden, hidden[:, start:start + 64])
                      for start in range(0, length, 64)]
            pair_representation = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
            raw_scores = head.score_pair_matrix(pair_representation, head_ids, False, None, None)
            base_scores = head.calibration(raw_scores, torch.tensor([length], device=device))
            full_mask = pair_mask_from_ids(head_ids, head.min_loop).to(device)
            base_scores = torch.where(full_mask, base_scores, torch.full_like(base_scores, NEG_BIG))
            mask = torch.as_tensor(harness.valid_pair_mask(sequence), device=device)
            first_pass = base_scores[0]
            numerics = compute_gec_context_v1(first_pass, mask, mode=arm,
                                              shuffle_key=gec_destroy_key_v1(0, 0, 0, sequence))
            geometry = gec_geometry_v1(mask, head.min_loop)
            delta = residual(pair_representation[0], numerics.context, geometry, mask)
            scores = torch.where(mask, first_pass + delta, torch.full_like(first_pass, NEG_BIG))
            mask_numpy = mask.detach().cpu().numpy()
            _, posterior = harness.inside_outside(scores.detach().cpu().numpy().astype(np.float64), mask_numpy)
            if not np.isfinite(posterior).all():
                raise FloatingPointError(f"non-finite posterior: {arm} {record['id']}")
            predicted = decode_mea(posterior, mask_numpy, 1.0)
            rows.append({"sequence_id": record["id"], "true_pairs": [list(pair) for pair in record["pairs"]],
                         "predicted_pairs": [list(pair) for pair in sorted(predicted)],
                         "sequence_length": length, "sequence_sha256": record["sequence_sha256"],
                         "checkpoint_sha256": checkpoint_audit["best_checkpoint"]["sha256"]})
            if index % 100 == 0 or index == len(records):
                print(f"[margin-export] {arm} {index}/{len(records)}", flush=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, output)
    export = {
        "schema_version": EXPORT_SCHEMA, "arm": arm, "panel_id": "tr1c_cluster80_dev_v1",
        "record_count": len(rows), "device": device_name, "cpu_fallback": False,
        "best_checkpoint_sha256": checkpoint_audit["best_checkpoint"]["sha256"],
        "selected_model_tensor_sha256": checkpoint_audit["best_tensor_sha256"],
        "validation_log_sha256": qualification["validation"][arm]["validation_log"]["sha256"],
        "prediction_sha256": independent.sha256(output), "decoder": config["model"]["decoder"],
        "external_test_sets": False,
    }
    write_json(output_dir / "prediction_export.json", export)
    del encoder, head, residual, checkpoint
    torch.cuda.empty_cache()
    return export


def evaluate_run(repo: Path, artifact_root: Path, prediction_root: Path,
                 nll_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        unexpected = [path.name for path in output_root.iterdir()
                      if path.name != "predictions" or path.resolve() != prediction_root.resolve()]
        if unexpected:
            raise ValueError(f"output directory contains unexpected entries: {unexpected}")
    output_root.mkdir(parents=True, exist_ok=True)
    qualification = qualify(repo, artifact_root)
    frozen = qualification["frozen"]
    margin, margin_identities = bind_margin_predictions(frozen, prediction_root, qualification["checkpoints"])
    nll, nll_identities, nll_provenance = load_nll_predictions(repo, frozen, nll_root)
    print("[margin-eval] recomputing 16 gates, paired CI and sensitivity from raw pairs", flush=True)
    science = evaluate_bound(frozen, margin, nll)
    passed = science["margin_gates"]["pass"]
    report = {
        "schema_version": SCHEMA,
        "status": "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if passed else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL",
        "overall_pass": passed, "final_verdict_allowed": True,
        "independence": {"run_meta_read": False, "official_evaluator_imported": False,
                         "metric_source": "raw predicted pairs and frozen truth",
                         "validation_loss_nll_comparison": "FORBIDDEN_DIFFERENT_OBJECTIVES"},
        "execution": {"metric_device": "cpu", "checkpoint_deserialization": "map_location=cpu",
                      "model_inference_during_evaluation": False, "python": platform.python_version(),
                      "numpy": np.__version__, "torch": torch.__version__},
        "code_identity": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                          "script": identity(Path(__file__).resolve())},
        "panel": {"panel_id": "tr1c_cluster80_dev_v1", "n_sequences": len(frozen),
                  "n_pairs": sum(len(row.true_pairs) for row in frozen)},
        "validation_audit": qualification["validation"], "checkpoint_audit": qualification["checkpoints"],
        "margin_prediction_identities": margin_identities, "qualified_nll_reference": nll_provenance,
        **science,
    }
    write_json(output_root / "independent_evaluation.json", report)
    summary = {"schema_version": SCHEMA, "status": report["status"], "overall_pass": passed,
               "failed_margin_gates": science["margin_gates"]["failed_gates"],
               "arm_metrics": science["arm_metrics"],
               "margin_vs_nll_global": {arm: row["metrics"] for arm, row in science["margin_vs_qualified_nll_same_arm"].items()}}
    write_json(output_root / "gate_summary.json", summary)
    write_json(output_root / "input_manifest.json", {"inputs": qualification["inputs"]})
    return summary


def export_run(repo: Path, artifact_root: Path, prediction_root: Path, device: str) -> dict:
    qualification = qualify(repo, artifact_root)
    if prediction_root.exists():
        raise ValueError(f"prediction root must not exist: {prediction_root}")
    prediction_root.mkdir(parents=True)
    exports = {}
    for arm in ARMS:
        exports[arm] = export_arm(repo, arm, qualification, prediction_root, device)
    return exports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "export", "evaluate", "all"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--nll-artifact-root", type=Path, default=DEFAULT_NLL_ROOT)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    prediction_root = args.prediction_root or args.output_root / "predictions"
    try:
        if args.command == "audit":
            prepare_output(args.output_root)
            qualification = qualify(repo, args.artifact_root)
            report = {"schema_version": SCHEMA, "status": "QUALIFIED_FOR_INDEPENDENT_EXPORT",
                      "final_verdict_allowed": False, "run_meta_read": False,
                      "validation_audit": qualification["validation"],
                      "checkpoint_audit": qualification["checkpoints"]}
            write_json(args.output_root / "qualification_audit.json", report)
            print(json.dumps({"status": report["status"]}, indent=2))
            return 0
        if args.command == "export":
            exports = export_run(repo, args.artifact_root, prediction_root, args.device)
            print(json.dumps({"status": "EXPORTED", "arms": list(exports)}, indent=2))
            return 0
        if args.command == "all":
            export_run(repo, args.artifact_root, prediction_root, args.device)
        summary = evaluate_run(repo, args.artifact_root, prediction_root,
                               args.nll_artifact_root, args.output_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["overall_pass"] else 1
    except TrainingInProgress as progress:
        report = {"schema_version": SCHEMA, "status": "IN_PROGRESS", "final_verdict_allowed": False,
                  "scientific_verdict": None, "run_meta_read": False, "arms": progress.arms}
        if not args.output_root.exists() or not any(args.output_root.iterdir()):
            write_json(args.output_root / "in_progress.json", report)
        print(json.dumps({"status": "IN_PROGRESS", "final_verdict_allowed": False,
                          "arms": {arm: row["reason"] for arm, row in progress.arms.items()
                                   if row["status"] == "IN_PROGRESS"}}, indent=2, sort_keys=True))
        return 3
    except (OSError, ValueError, FloatingPointError, KeyError, StopIteration, json.JSONDecodeError) as error:
        print(f"structured-margin independent evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
