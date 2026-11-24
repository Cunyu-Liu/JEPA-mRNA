#!/usr/bin/env python3
"""Resume-safe four-arm GEC seed0 production orchestration on explicit CUDA."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/miniconda3/envs/lucaone_cu128/bin/python")
ARTIFACT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v2")
CONFIG = REPO / "configs/gec_four_arm_convergence_v1.json"
LONG_RUN_AUDIT = Path(
    "/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v1/"
    "independent_live_integrity_audit_20261124T074426Z.json"
)
ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
IDENTITY_SCHEMA = "foldnar.gec_seed0_production_identity.v1"
STATUS_SCHEMA = "foldnar.gec_seed0_production_arm_status.v1"
MEMORY_SAFETY_FACTOR = 1.5
CODE_FILES = (
    "src/rnajepa/gec_numeric_v1.py",
    "src/rnajepa/gec_context.py",
    "src/rnajepa/harness.py",
    "src/rnajepa/decision_head.py",
    "src/rnajepa/encoder.py",
    "src/rnajepa/prior_alignment.py",
    "src/rnajepa/raw_score_matrix.py",
    "tools/train_plan_a_gec_convergence_v1.py",
    "tools/train_plan_a_clean_head_p1.py",
    "tools/rinalmo_preflight.py",
    "tools/init_gec_residual_v1.py",
    "tools/pair_decode_common.py",
    "tools/gec_seed0_scientific_evaluator.py",
    "tools/run_gec_seed0_production_v1.py",
    "tools/run_gec_seed0_production_v1.sh",
    "tools/verify_gec_seed0_production_v1.py",
    "configs/gec_four_arm_convergence_v1.json",
    "manifests/foldnar_v1/model_dev_registry.json",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    config = read_json(CONFIG)
    if config["training"]["seed"] != 0 or config["training"]["fixed_step_cap"] is not None:
        raise ValueError("production requires seed0 without a fixed step cap")
    if tuple(config["arms"]) != ARMS:
        raise ValueError("unexpected four-arm identities")
    if config["model"]["decoder"] != {
        "name": "MEA", "temperature": 1.0, "gamma": 1.0,
        "score_variant": "train_off/plana_off", "search_allowed": False,
    }:
        raise ValueError("decoder drift")
    return config


def registry_path(config: dict) -> Path:
    path = Path(config["assets"]["registry"])
    return path if path.is_absolute() else REPO / path


def build_arm_command(arm: str, root: Path = ARTIFACT_ROOT) -> list[str]:
    if arm not in ARMS:
        raise ValueError(arm)
    config = load_config()
    assets = config["assets"]
    return [
        str(PYTHON), str(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
        "--gec-config", str(CONFIG), "--arm", arm, "--out", str(root / arm),
        "--shared-init", assets["shared_init"],
        "--gec-init", str(root / "gec_shared_residual_init_seed0.pt"),
        "--device", "cuda:0", "--steps", "0", "--seed", "0",
        "--batch-size", str(config["training"]["batch_size"]),
        "--head-lr", str(config["training"]["head_lr"]),
        "--backbone-lr", str(config["training"]["backbone_lr"]),
        "--warmup-head-steps", str(config["training"]["warmup_head_steps"]),
        "--unfreeze-every", str(config["training"]["unfreeze_every"]),
        "--unfreeze-per-step", str(config["training"]["unfreeze_per_step"]),
        "--max-unfrozen-blocks", str(config["training"]["max_unfrozen_blocks"]),
        "--warmup-steps", str(config["training"]["optimizer_warmup_steps"]),
        "--grad-clip", str(config["training"]["grad_clip"]),
        "--validation-every", str(config["convergence"]["validation_every"]),
        "--validation-patience", str(config["convergence"]["patience"]),
        "--validation-min-delta", str(config["convergence"]["min_delta"]),
        "--save-every", "500", "--train-record-limit", "0",
        "--validation-record-limit", "0", "--zchunk", "64",
    ]


def build_evaluator_command(root: Path = ARTIFACT_ROOT) -> list[str]:
    command = [
        str(PYTHON), str(REPO / "tools/gec_seed0_scientific_evaluator.py"),
        "--registry", str(registry_path(load_config())),
        "--panel-id", "tr1c_cluster80_dev_v1", "--gec-config", str(CONFIG),
    ]
    for arm in ARMS:
        command += ["--prediction", arm + "=" + str(root / arm / "dev_predictions.jsonl")]
    for arm in ARMS:
        command += ["--run-metadata", arm + "=" + str(root / arm / "run_meta.json")]
    command += ["--output", str(root / "scientific_evaluation.json")]
    return command


def query_gpu0() -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits", "-i", "0",
    ], text=True).strip()
    index, name, total, used = [value.strip() for value in output.splitlines()[0].split(",")]
    if int(index) != 0:
        raise ValueError("explicit GPU index 0 was not returned")
    return {"index": 0, "name": name, "total_mib": int(total), "used_mib": int(used)}


def plan_parallelism(audit_path: Path = LONG_RUN_AUDIT, gpu: dict | None = None) -> dict:
    audit = read_json(audit_path)
    if (audit.get("schema_version") != "foldnar.independent_gec_seed0_live_integrity_audit_20261124.v1"
            or audit.get("artifact_root") != str(LONG_RUN_AUDIT.parent)
            or audit.get("scientific_result") is not False):
        raise ValueError("unexpected long-run memory audit")
    peaks = {}
    for arm in ARMS:
        result = audit["arms"][arm]
        peak = int(result["training"]["cuda_peak_bytes"]["max"])
        if peak <= 0 or result["completed_steps"] < 45:
            raise ValueError(f"insufficient long-run memory evidence for {arm}")
        peaks[arm] = peak
    if not any(audit["arms"][arm].get("cuda_oom_observed") for arm in ARMS):
        raise ValueError("long-run audit does not contain the required OOM evidence")
    gpu = query_gpu0() if gpu is None else gpu
    total_bytes = int(gpu["total_mib"]) * 1024 ** 2
    used_bytes = int(gpu["used_mib"]) * 1024 ** 2
    reserve_bytes = max(4 * 1024 ** 3, int(total_bytes * 0.20))
    max_peak = max(peaks.values())
    per_arm_budget = int(max_peak * MEMORY_SAFETY_FACTOR)
    safe = min(len(ARMS), max(0, (total_bytes - used_bytes - reserve_bytes) // per_arm_budget))
    if safe < 1:
        raise RuntimeError("measured long-run memory does not support even one safe CUDA arm")
    waves = [list(ARMS[index:index + safe]) for index in range(0, len(ARMS), safe)]
    return {
        "parallelism": int(safe), "waves": waves, "device": "cuda:0", "cpu_fallback": False,
        "gpu": gpu, "long_run_audit": str(audit_path),
        "long_run_audit_sha256": sha256(audit_path),
        "long_run_allocated_peak_bytes": peaks,
        "max_long_run_allocated_bytes": max_peak,
        "memory_safety_factor": MEMORY_SAFETY_FACTOR,
        "per_arm_budget_bytes": per_arm_budget, "reserved_bytes": reserve_bytes,
        "evidence_scope": "failed seed0_v1 long-run memory only; no statuses or checkpoints inherited",
    }


def code_manifest() -> dict:
    return {
        name: {"sha256": sha256(REPO / name), "bytes": (REPO / name).stat().st_size}
        for name in CODE_FILES
    }



def current_git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def require_clean_scientific_files() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *CODE_FILES], cwd=REPO, text=True,
    )
    if status.strip():
        raise ValueError(f"scientific source/config/launcher files must be clean: {status.strip()}")


def build_identity(root: Path, plan: dict) -> dict:
    config = load_config()
    assets = config["assets"]
    require_clean_scientific_files()
    head = current_git_head()
    manifest = code_manifest()
    arm_commands = {arm: build_arm_command(arm, root) for arm in ARMS}
    evaluator_command = build_evaluator_command(root)
    return {
        "schema_version": IDENTITY_SCHEMA, "artifact_root": str(root),
        "repository": str(REPO), "git_commit": head,
        "launcher_command": "bash " + str(REPO / "tools/run_gec_seed0_production_v1.sh") + " --launch-approved",
        "code_manifest": manifest,
        "code_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "gec_config": str(CONFIG), "gec_config_sha256": sha256(CONFIG),
        "source_contract": assets.get("source_contract", config["source_contract"]),
        "source_contract_sha256": sha256(config["source_contract"]),
        "model_dev_registry": str(registry_path(config)),
        "model_dev_registry_sha256": sha256(registry_path(config)),
        "data": assets["data"], "data_sha256": sha256(assets["data"]),
        "train_ids": assets["train_ids"], "train_ids_sha256": sha256(assets["train_ids"]),
        "validation_ids": assets["validation_ids"],
        "validation_ids_sha256": sha256(assets["validation_ids"]),
        "shared_init": assets["shared_init"], "shared_init_sha256": sha256(assets["shared_init"]),
        "gec_init": str(root / "gec_shared_residual_init_seed0.pt"),
        "gec_init_sha256": sha256(root / "gec_shared_residual_init_seed0.pt"),
        "panel_id": "tr1c_cluster80_dev_v1", "external_test_sets": False,
        "arm_commands": arm_commands,
        "arm_command_sha256": {arm: command_sha256(cmd) for arm, cmd in arm_commands.items()},
        "evaluator_command": evaluator_command,
        "evaluator_command_sha256": command_sha256(evaluator_command),
        "parallelism_plan": plan, "created_utc": utc_now(),
    }


def validate_identity(document: dict, root: Path = ARTIFACT_ROOT) -> None:
    if document.get("schema_version") != IDENTITY_SCHEMA or Path(document["artifact_root"]) != root:
        raise ValueError("production artifact identity or root mismatch")
    expected = build_identity(root, document["parallelism_plan"])
    comparable = {key: value for key, value in expected.items() if key != "created_utc"}
    actual = {key: value for key, value in document.items() if key != "created_utc"}
    if actual != comparable:
        differing = sorted(key for key in set(actual) | set(comparable)
                           if actual.get(key) != comparable.get(key))
        raise ValueError(f"frozen production identity mismatch: {differing}")


def acquire_lock(path: Path):
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another production process holds lock: {path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def release_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def production_environment() -> dict:
    return dict(
        os.environ, PYTHONPATH=str(REPO / "src") + ":" + str(REPO), CUDA_VISIBLE_DEVICES="0",
        RN_CUDA_INSIDE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
    )


def arm_status_path(root: Path, arm: str) -> Path:
    return root / arm / "launcher_status.json"


def write_arm_status(root: Path, arm: str, status: str, **updates) -> dict:
    path = arm_status_path(root, arm)
    previous = read_json(path) if path.exists() else {
        "schema_version": STATUS_SCHEMA, "arm": arm, "history": [],
    }
    if previous.get("status") != status:
        previous.setdefault("history", []).append({
            "from": previous.get("status"), "to": status, "utc": utc_now(),
        })
    previous.update(status=status, updated_utc=utc_now(), **updates)
    atomic_json(previous, path)
    return previous


def converged_metadata(root: Path, arm: str) -> dict:
    path = root / arm / "run_meta.json"
    metadata = read_json(path)
    if (metadata.get("status") != "converged" or metadata.get("converged") is not True
            or metadata.get("stop_reason") != "CONVERGED_VALIDATION_NLL_AFTER_TERMINAL_UNFREEZE"
            or metadata.get("engineering_smoke") is not False
            or metadata["schedule"]["steps"] != 0
            or metadata["schedule"]["train_record_limit"] != 0
            or metadata["schedule"]["validation_record_limit"] != 0
            or metadata["environment"]["resolved_device"] != "cuda:0"
            or metadata["compute_kernels"]["cpu_fallback"] is not False):
        raise ValueError(f"{arm} has no eligible converged full-CUDA run")
    return metadata


def terminate_processes(active: dict, root: Path, reason: str) -> None:
    for arm, item in active.items():
        if item["process"].poll() is None:
            item["process"].send_signal(signal.SIGTERM)
    for arm, item in active.items():
        try:
            returncode = item["process"].wait(timeout=60)
        except subprocess.TimeoutExpired:
            item["process"].kill()
            returncode = item["process"].wait()
        item["log"].close()
        release_lock(item["lock"])
        write_arm_status(root, arm, "failed", pid=item["process"].pid,
                         returncode=returncode, failure_reason=reason)


def run_training(root: Path, parallelism: int) -> None:
    pending = []
    active = {}
    for arm in ARMS:
        try:
            converged_metadata(root, arm)
            write_arm_status(root, arm, "converged", pid=None, returncode=0, command=build_arm_command(arm, root), command_sha256=command_sha256(build_arm_command(arm, root)), log=str(root / arm / "production.log"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pending.append(arm)
    try:
        while pending or active:
            while pending and len(active) < parallelism:
                arm = pending.pop(0)
                directory = root / arm
                directory.mkdir(parents=True, exist_ok=True)
                lock = acquire_lock(directory / ".production_arm.lock")
                command = build_arm_command(arm, root)
                log = (directory / "production.log").open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=REPO, env=production_environment(),
                    stdout=log, stderr=subprocess.STDOUT,
                )
                (directory / "production.pid").write_text(f"{process.pid}\n")
                active[arm] = {"process": process, "log": log, "lock": lock}
                write_arm_status(root, arm, "running", pid=process.pid, returncode=None,
                                 command=command, command_sha256=command_sha256(command),
                                 log=str(directory / "production.log"), failure_reason=None,
                                 selected_parallelism=parallelism,
                                 wave=ARMS.index(arm) // parallelism + 1)
            time.sleep(1)
            for arm, item in list(active.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["log"].close()
                release_lock(item["lock"])
                del active[arm]
                if returncode == 0:
                    try:
                        converged_metadata(root, arm)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                        returncode = 1
                        failure = f"trainer exited zero without eligible convergence: {error}"
                    else:
                        write_arm_status(root, arm, "converged", pid=item["process"].pid,
                                         returncode=0, failure_reason=None)
                        continue
                else:
                    failure = f"trainer exited with status {returncode}"
                write_arm_status(root, arm, "failed", pid=item["process"].pid,
                                 returncode=returncode, failure_reason=failure)
                terminate_processes(active, root, f"peer arm {arm} failed")
                active.clear()
                raise RuntimeError(f"production arm {arm} failed: {failure}; no evaluation invoked")
    except BaseException:
        terminate_processes(active, root, "supervisor interrupted")
        active.clear()
        raise


def export_predictions(root: Path, arm: str) -> None:
    metadata = converged_metadata(root, arm)
    directory = root / arm
    best = directory / "best.pt"
    best_sha256 = sha256(best)
    if best_sha256 != metadata["best_checkpoint_sha256"]:
        raise ValueError(f"best checkpoint identity mismatch for {arm}")
    output = directory / "dev_predictions.jsonl"
    export_meta_path = directory / "prediction_export.json"
    if output.exists() or export_meta_path.exists():
        export_meta = read_json(export_meta_path)
        if (export_meta.get("best_checkpoint_sha256") != best_sha256
                or export_meta.get("prediction_sha256") != sha256(output)
                or export_meta.get("arm") != arm
                or export_meta.get("device") != "cuda:0"
                or export_meta.get("panel_id") != "tr1c_cluster80_dev_v1"):
            raise ValueError(f"stale prediction export for {arm}")
        write_arm_status(root, arm, "exported", pid=None, returncode=0)
        return
    write_arm_status(root, arm, "exporting", pid=os.getpid(), returncode=None)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from pair_decode_common import decode_mea
    from rinalmo_preflight import VOCAB, load_encoder
    from rnajepa import harness
    from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
    from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
    from rnajepa.gec_context import (
        GECResidualV1, gec_destroy_key_v1, gec_geometry_v1, require_explicit_cuda,
    )
    from rnajepa.gec_numeric_v1 import compute_gec_context_v1
    from tools.train_plan_a_gec_convergence_v1 import NEG_BIG, load_registered_records

    if os.environ.get("RN_CUDA_INSIDE") != "1":
        raise ValueError("prediction export requires RN_CUDA_INSIDE=1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = require_explicit_cuda("cuda:0")
    config = load_config()
    records = load_registered_records(config, "validation", 0)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_phase") != "best_eligible" or checkpoint.get("n_unfrozen") != 20:
        raise ValueError(f"prediction export requires terminal best-eligible checkpoint for {arm}")
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(
        d_model=int(report["d_model"]), d_z=config["model"]["d_z"], hidden=64,
        scorer="resnet2d", chunk_size=0, resnet_blocks=4, resnet_hidden=64,
        use_turner_prior=True,
    )
    residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    residual.load_state_dict(checkpoint["residual"], strict=True)
    head.use_turner_prior = False
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    residual = residual.to(device).eval()
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records, 1):
            sequence = record["sequence"]
            length = len(sequence)
            tokens = [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence]
            framed = [VOCAB["<cls>"]] + tokens + [VOCAB["<eos>"]]
            encoder_ids = torch.tensor([framed], dtype=torch.long, device=device)
            attention = torch.ones_like(encoder_ids)
            head_ids = torch.tensor(
                [[BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]],
                dtype=torch.long, device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = encoder(input_ids=encoder_ids, attention_mask=attention)
            hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
            hidden = hidden[:, 1:length + 1, :].float().to(torch.float16).to(torch.float32)
            chunks = [
                head.pair_repr.cross(hidden, hidden[:, start:start + 64])
                for start in range(0, length, 64)
            ]
            pair_representation = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
            raw_scores = head.score_pair_matrix(pair_representation, head_ids, False, None, None)
            lengths_tensor = torch.tensor([length], device=device)
            base_scores = head.calibration(raw_scores, lengths_tensor)
            full_mask = pair_mask_from_ids(head_ids, head.min_loop).to(device)
            base_scores = torch.where(
                full_mask, base_scores, torch.full_like(base_scores, NEG_BIG),
            )
            mask = torch.as_tensor(harness.valid_pair_mask(sequence), device=device)
            first_pass = base_scores[0]
            numerics = compute_gec_context_v1(
                first_pass, mask, mode=arm,
                shuffle_key=gec_destroy_key_v1(0, 0, 0, sequence),
            )
            geometry = gec_geometry_v1(mask, head.min_loop)
            delta = residual(pair_representation[0], numerics.context, geometry, mask)
            scores = torch.where(mask, first_pass + delta, torch.full_like(first_pass, NEG_BIG))
            mask_numpy = mask.detach().cpu().numpy()
            _, posterior = harness.inside_outside(
                scores.detach().cpu().numpy().astype(np.float64) / 1.0, mask_numpy,
            )
            if not np.isfinite(posterior).all():
                raise FloatingPointError("non-finite posterior for %s %s" % (arm, record["id"]))
            predicted = decode_mea(posterior, mask_numpy, 1.0)
            rows.append({
                "sequence_id": record["id"],
                "true_pairs": [list(pair) for pair in record["pairs"]],
                "predicted_pairs": [list(pair) for pair in sorted(predicted)],
                "sequence_length": length, "sequence_sha256": record["sequence_sha256"],
                "checkpoint_sha256": best_sha256,
            })
            if index % 100 == 0 or index == len(records):
                print(f"[gec-export] arm={arm} {index}/{len(records)}", flush=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, output)
    atomic_json({
        "schema_version": "foldnar.gec_seed0_prediction_export.v1", "arm": arm,
        "panel_id": "tr1c_cluster80_dev_v1", "record_count": len(rows),
        "device": "cuda:0", "cpu_fallback": False,
        "best_checkpoint_sha256": best_sha256, "prediction_sha256": sha256(output),
        "decoder": config["model"]["decoder"], "external_test_sets": False,
    }, export_meta_path)
    write_arm_status(root, arm, "exported", pid=os.getpid(), returncode=0)
    del encoder, head, residual, checkpoint
    torch.cuda.empty_cache()


def require_all_converged(root: Path) -> None:
    for arm in ARMS:
        converged_metadata(root, arm)


def run_scientific_evaluator(root: Path) -> int:
    require_all_converged(root)
    for arm in ARMS:
        if not (root / arm / "dev_predictions.jsonl").is_file():
            raise ValueError(f"missing legal cluster-dev predictions for {arm}")
        report = root / "scientific_evaluation.json"
    final_status = root / "production_final_status.json"
    if report.exists() or final_status.exists():
        status = read_json(final_status)
        if status.get("scientific_evaluation_sha256") != sha256(report):
            raise ValueError("stale or incomplete scientific evaluation status")
        return int(status["evaluator_exit_code"])
    command = build_evaluator_command(root)
    with (root / "scientific_evaluator.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, env=production_environment(), stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if not report.is_file():
        raise RuntimeError("strict evaluator did not create its report")
    result = read_json(report)
    expected_status = "PASS_GEC_SEED0" if completed.returncode == 0 else "FAIL_GEC_SEED0"
    if completed.returncode not in (0, 1) or result.get("status") != expected_status:
        raise RuntimeError(f"strict evaluator failed with status {completed.returncode}")
    atomic_json({
        "schema_version": "foldnar.gec_seed0_production_final.v1",
        "status": result["status"], "evaluator_exit_code": completed.returncode,
        "scientific_evaluation_sha256": sha256(report),
        "evaluator_command_sha256": command_sha256(command),
        "all_arms_converged_before_evaluation": True,
        "panel_id": "tr1c_cluster80_dev_v1", "external_test_sets": False,
        "completed_utc": utc_now(),
    }, final_status)
    print(json.dumps({"status": result["status"], "report": str(report),
                      "evaluator_exit_code": completed.returncode}, indent=2))
    return completed.returncode


def dry_run() -> dict:
    config = load_config()
    for key in ("data", "train_ids", "validation_ids", "shared_init"):
        path = Path(config["assets"][key])
        expected = config["assets"].get(key + "_sha256")
        if expected is not None and sha256(path) != expected:
            raise ValueError(f"frozen asset hash mismatch: {key}")
    plan = plan_parallelism()
    return {
        "status": "DRY_RUN_NO_TRAINING", "artifact_root": str(ARTIFACT_ROOT),
        "artifact_root_created": False, "git_commit": current_git_head(),
        "parallelism_plan": plan, "arm_commands": {arm: build_arm_command(arm) for arm in ARMS},
        "evaluator_command": build_evaluator_command(),
        "production_launch_command": (
            "bash " + str(REPO / "tools/run_gec_seed0_production_v1.sh") + " --launch-approved"
        ),
        "full_cluster_train_ids": config["assets"]["train_ids"],
        "full_cluster_dev_ids": config["assets"]["validation_ids"],
        "steps": 0, "fixed_step_cap": None, "external_test_sets": False,
    }


def prepare_artifact_root() -> tuple[dict, object]:
    root = ARTIFACT_ROOT
    require_clean_scientific_files()
    head = current_git_head()
    fresh = not root.exists()
    if fresh:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json({
            "schema_version": "foldnar.gec_seed0_production_initialization.v1",
            "artifact_root": str(root), "repository": str(REPO), "git_commit": head,
        }, root / "production_initialization.json")
    lock = acquire_lock(root / ".production_supervisor.lock")
    identity_path = root / "production_identity.json"
    if identity_path.exists():
        identity = read_json(identity_path)
        validate_identity(identity, root)
        return identity, lock
    marker = read_json(root / "production_initialization.json")
    if (marker.get("schema_version") != "foldnar.gec_seed0_production_initialization.v1"
            or marker.get("artifact_root") != str(root) or marker.get("repository") != str(REPO)
            ):
        release_lock(lock)
        raise ValueError("refusing non-fresh or foreign artifact root")
    init = root / "gec_shared_residual_init_seed0.pt"
    if not init.exists():
        command = [
            str(PYTHON), str(REPO / "tools/init_gec_residual_v1.py"),
            "--shared-init", load_config()["assets"]["shared_init"],
            "--gec-config", str(CONFIG), "--out", str(init), "--seed", "0",
        ]
        with (root / "initialization.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO, env=production_environment(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    identity = build_identity(root, plan_parallelism())
    atomic_json(identity, identity_path)
    return identity, lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch-approved", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2, allow_nan=False))
        return 0
    identity, supervisor_lock = prepare_artifact_root()
    try:
        os.environ.update(production_environment())
        atomic_json({
            "schema_version": "foldnar.gec_seed0_production_launch.v1",
            "production_identity_sha256": sha256(ARTIFACT_ROOT / "production_identity.json"),
            "parallelism": identity["parallelism_plan"]["parallelism"],
            "device": "cuda:0", "cpu_fallback": False, "pid": os.getpid(),
            "started_utc": utc_now(),
        }, ARTIFACT_ROOT / "production_launch.json")
        run_training(ARTIFACT_ROOT, identity["parallelism_plan"]["parallelism"])
        require_all_converged(ARTIFACT_ROOT)
        for arm in ARMS:
            export_predictions(ARTIFACT_ROOT, arm)
        return run_scientific_evaluator(ARTIFACT_ROOT)
    finally:
        release_lock(supervisor_lock)


if __name__ == "__main__":
    raise SystemExit(main())
