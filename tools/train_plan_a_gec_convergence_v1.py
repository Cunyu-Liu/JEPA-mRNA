#!/usr/bin/env python3
"""Four-arm GEC residual training on the active Plan-A initialization and schedule."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa import harness
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.gec_context import (
    GECConvergenceStateV1,
    GECResidualV1,
    GEC_RESIDUAL_FORMAT_V1,
    gec_destroy_key_v1,
    gec_geometry_v1,
    require_explicit_cuda,
    terminal_unfreeze_step_v1,
    validate_gec_arm_v1,
)
from rnajepa.gec_numeric_v1 import compute_gec_context_v1
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from rnajepa.raw_score_matrix import capped_unfreeze_blocks
from rinalmo_preflight import VOCAB, load_encoder
from tools.train_plan_a_clean_head_p1 import (
    BatchStream,
    NEG_BIG,
    atomic_torch_save,
    divergence_reason,
    restore_batch_stream,
    restore_scheduler_state,
    restore_unfrozen_blocks,
)
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


MODEL_DEV_REGISTRY_SHA256 = "62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902"
RUN_META_SCHEMA_V2 = "foldnar.gec_training_run.v2"
RUN_IDENTITY_SCHEMA_V2 = "foldnar.gec_run_identity.v2"
SMOKE_IDENTITY_SCHEMA_V2 = "foldnar.gec_smoke_artifact_identity.v2"
SMOKE_ARTIFACT_GENERATION_V2 = "gec_smoke_v2"
VALIDATION_DESTROY_STEP_V2 = 0
BATCH_NORM_POLICY_V2 = "inherited_padded_batch_train_once_plus_frozen_singleton_eval"
CODE_FILES = (
    "src/rnajepa/gec_numeric_v1.py",
    "src/rnajepa/gec_context.py",
    "tools/train_plan_a_gec_convergence_v1.py",
    "tools/init_gec_residual_v1.py",
    "tools/run_gec_four_arm_smoke_v1.sh",
    "tools/verify_gec_smoke_v1.py",
    "configs/gec_four_arm_convergence_v1.json",
    "tests/test_gec_numeric_v1.py",
    "tests/test_gec_context.py",
    "tests/test_gec_convergence.py",
)


def build_code_manifest(repo: str | Path = REPO) -> dict:
    repo = Path(repo)
    return {
        name: {"sha256": sha256_file(repo / name), "bytes": (repo / name).stat().st_size}
        for name in CODE_FILES
    }


def code_manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _registry_path(config: dict) -> Path:
    path = Path(config["assets"]["registry"])
    return path if path.is_absolute() else REPO / path


def require_model_dev_registry(config: dict) -> tuple[Path, dict]:
    path = _registry_path(config)
    actual = sha256_file(path)
    if actual != MODEL_DEV_REGISTRY_SHA256:
        raise ValueError(f"model-development registry SHA256 mismatch: {actual}")
    if config["assets"].get("model_dev_registry_sha256") != MODEL_DEV_REGISTRY_SHA256:
        raise ValueError("frozen config does not pin the required model-development registry")
    return path, json.loads(path.read_text())


def _atomic_json(value, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(value, path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def _finite_float_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def require_finite_validation_nll(value, step: int, record: dict | None = None) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    if record is None:
        raise FloatingPointError(f"NON_FINITE_AGGREGATE_VALIDATION_NLL step={step} loss={value}")
    raise FloatingPointError(
        f"NON_FINITE_VALIDATION_NLL step={step} record_id={record['id']} "
        f"length={len(record['sequence'])} loss={value}"
    )


def validation_error_record(step: int, error: Exception, record: dict | None = None,
                            losses: list[float] | None = None) -> dict:
    return {
        "step": step, "error": str(error),
        "record_id": None if record is None else record["id"],
        "sequence_length": None if record is None else len(record["sequence"]),
        "sequence_sha256": None if record is None else record["sequence_sha256"],
        "validation_losses": losses,
    }


def build_validation_log_record(step: int, validation_loss: float, decision: dict,
                                convergence: GECConvergenceStateV1, n_unfrozen: int) -> dict:
    return {
        "step": step, "validation_loss": validation_loss,
        "n_unfrozen": n_unfrozen,
        **decision,
        "best_step": convergence.best_step,
        "best_validation_loss": _finite_float_or_none(convergence.best_validation_loss),
        "bad_evaluations": convergence.bad_evaluations,
    }


def build_terminal_convergence_metadata(stopped: dict, restored_best: dict | None = None) -> dict:
    metadata = {"convergence": stopped}
    if restored_best is not None:
        metadata["best_convergence"] = restored_best
    return metadata


def _read_previous_run_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    if not isinstance(previous, dict) or previous.get("schema_version") != RUN_META_SCHEMA_V2:
        raise ValueError("previous run_meta is missing or has an unsupported schema")
    if not isinstance(previous.get("run_identity"), dict):
        raise ValueError("previous run_meta has no complete run identity")
    if previous.get("status") not in {
        "running", "interrupted", "diverged", "engineering_smoke_completed", "converged",
    }:
        raise ValueError(f"invalid previous run status: {previous.get('status')}")
    return previous


def build_smoke_artifact_identity(
    config_path: str | Path,
    shared_init: str | Path,
    gec_init: str | Path,
    repo: str | Path = REPO,
) -> dict:
    config = load_gec_config(config_path)
    residual_initial = torch.load(gec_init, map_location="cpu", weights_only=False)
    registry_path, _ = require_model_dev_registry(config)
    assets = config["assets"]
    manifest = build_code_manifest(repo)
    return {
        "schema_version": SMOKE_IDENTITY_SCHEMA_V2,
        "artifact_generation": SMOKE_ARTIFACT_GENERATION_V2,
        "code_manifest": manifest,
        "code_manifest_sha256": code_manifest_sha256(manifest),
        "gec_config": str(Path(config_path).resolve()),
        "gec_config_sha256": sha256_file(config_path),
        "source_contract": str(Path(config["source_contract"]).resolve()),
        "source_contract_sha256": sha256_file(config["source_contract"]),
        "data": str(Path(assets["data"]).resolve()),
        "data_sha256": sha256_file(assets["data"]),
        "model_dev_registry": str(registry_path.resolve()),
        "model_dev_registry_sha256": MODEL_DEV_REGISTRY_SHA256,
        "train_ids": str(Path(assets["train_ids"]).resolve()),
        "train_ids_sha256": sha256_file(assets["train_ids"]),
        "validation_ids": str(Path(assets["validation_ids"]).resolve()),
        "validation_ids_sha256": sha256_file(assets["validation_ids"]),
        "shared_init": str(Path(shared_init).resolve()),
        "shared_init_sha256": sha256_file(shared_init),
        "gec_init": str(Path(gec_init).resolve()),
        "gec_init_sha256": sha256_file(gec_init),
        "initial_residual_tensor_sha256": residual_initial["residual_tensor_sha256"],
        "smoke": config["smoke"],
    }


def validate_smoke_artifact_identity(document: dict, expected: dict) -> None:
    if document != expected:
        differing = sorted(key for key in set(document) | set(expected)
                           if document.get(key) != expected.get(key))
        raise ValueError(f"stale or invalid smoke artifact identity: {differing}")


def build_run_identity(
    args: argparse.Namespace,
    config: dict,
    train_records: list[dict],
    validation_records: list[dict],
    residual_initial: dict,
) -> dict:
    registry_path, _ = require_model_dev_registry(config)
    assets = config["assets"]
    manifest = build_code_manifest()
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA_V2,
        "arm": args.arm,
        "engineering_smoke": bool(args.engineering_smoke),
        "code_manifest": manifest,
        "code_manifest_sha256": code_manifest_sha256(manifest),
        "gec_config_sha256": sha256_file(args.gec_config),
        "source_contract_sha256": sha256_file(config["source_contract"]),
        "shared_init_sha256": sha256_file(args.shared_init),
        "gec_init_sha256": sha256_file(args.gec_init),
        "initial_residual_tensor_sha256": residual_initial["residual_tensor_sha256"],
        "model_dev_registry": str(registry_path.resolve()),
        "model_dev_registry_sha256": MODEL_DEV_REGISTRY_SHA256,
        "train_data_sha256": assets["data_sha256"],
        "train_ids_sha256": sha256_file(assets["train_ids"]),
        "validation_ids_sha256": sha256_file(assets["validation_ids"]),
        "train_record_ids": [row["id"] for row in train_records],
        "validation_record_ids": [row["id"] for row in validation_records],
        "validation_destroy_step": VALIDATION_DESTROY_STEP_V2,
        "batch_norm_policy": BATCH_NORM_POLICY_V2,
        "determinism": "torch_and_cudnn_deterministic_tf32_off",
        "terminal_unfrozen_blocks": config["training"]["max_unfrozen_blocks"],
    }
    if args.engineering_smoke:
        identity["smoke_artifact_identity_sha256"] = sha256_file(
            Path(config["smoke"]["artifact_root"]) / "smoke_artifact_identity.json"
        )
    return identity


def validate_run_identity(previous: dict, current_identity: dict) -> None:
    previous_identity = previous.get("run_identity")
    if previous_identity != current_identity:
        differing = sorted(
            key for key in set(previous_identity or {}) | set(current_identity)
            if (previous_identity or {}).get(key) != current_identity.get(key)
        )
        raise ValueError(f"resume identity mismatch: {differing}")
    for key in ("arm", "gec_config_sha256", "shared_init_sha256", "gec_init_sha256",
                "initial_residual_tensor_sha256", "train_record_ids", "validation_record_ids"):
        if previous.get(key) != current_identity.get(key):
            raise ValueError(f"resume metadata identity mismatch: {key}")


def validate_checkpoint_identity(state: dict, current_identity: dict) -> None:
    if state.get("format") != "foldnar.gec_checkpoint.v2":
        raise ValueError("unsupported GEC checkpoint format")
    if state.get("run_identity") != current_identity:
        raise ValueError("checkpoint run identity mismatch")


def model_tensor_hashes(encoder, head, residual) -> dict[str, str]:
    return {
        "encoder": tensor_state_sha256(encoder.state_dict()),
        "head": tensor_state_sha256(head.state_dict()),
        "residual": tensor_state_sha256(residual.state_dict()),
    }


def validate_converged_reentry(previous: dict, state: dict, best: dict, current_identity: dict) -> None:
    validate_checkpoint_identity(state, current_identity)
    validate_checkpoint_identity(best, current_identity)
    best_step = previous.get("restored_best_step")
    if (not previous.get("converged") or best_step is None
            or best_step != previous.get("convergence", {}).get("best_step")):
        raise ValueError("converged run metadata has no restored best eligible state")
    if (state.get("checkpoint_phase") != "restored_best_eligible"
            or state.get("terminal_status") != "converged"
            or int(state.get("step", -1)) != best_step
            or best.get("checkpoint_phase") != "best_eligible"
            or int(best.get("step", -1)) != best_step):
        raise ValueError("resume.pt does not represent the restored best eligible state")
    if state.get("model_tensor_sha256") != best.get("model_tensor_sha256"):
        raise ValueError("restored resume and best model tensor identities differ")
    if int(state.get("n_unfrozen", -1)) != current_identity.get("terminal_unfrozen_blocks", 20):
        raise ValueError("restored best checkpoint is not at terminal unfreeze")


def _batch_norm_running_state(head) -> list[tuple[torch.nn.BatchNorm2d, torch.Tensor, torch.Tensor, torch.Tensor]]:
    modules = [] if head.resnet2d is None else [
        module for module in head.resnet2d.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    return [(module, module.running_mean.clone(), module.running_var.clone(),
             module.num_batches_tracked.clone()) for module in modules]


def _restore_batch_norm_running_state(saved) -> None:
    with torch.no_grad():
        for module, running_mean, running_var, num_batches_tracked in saved:
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(num_batches_tracked)


def load_gec_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    require_model_dev_registry(config)
    if config.get("config_id") != "gec_four_arm_implementation_v1":
        raise ValueError("unexpected GEC implementation config")
    if tuple(config.get("arms", ())) != (
        "matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"
    ):
        raise ValueError("GEC arm identities or order changed")
    source = json.loads(Path(config["source_contract"]).read_text())
    if source.get("contract_id") != "gec_four_arm_convergence_contract_v1":
        raise ValueError("unexpected frozen GEC source contract")
    training = config["training"]
    terminal = terminal_unfreeze_step_v1(
        training["warmup_head_steps"], training["unfreeze_every"],
        training["unfreeze_per_step"], training["n_blocks"], training["max_unfrozen_blocks"],
    )
    if terminal != training["terminal_step"] or terminal != 9600:
        raise ValueError("GEC terminal unfreeze identity mismatch")
    return config


def _panel_asset(registry: dict, panel_id: str, key: str) -> dict:
    panel = next((row for row in registry["panels"] if row["panel_id"] == panel_id), None)
    if panel is None:
        raise ValueError(f"missing registered model-development panel: {panel_id}")
    return panel[key]


def _read_ids(path: str | Path, expected_sha256: str) -> set[str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"model-development ID hash mismatch: {path}")
    ids = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"empty model-development ID file: {path}")
    return ids


def load_registered_records(
    config: dict,
    split: str,
    record_limit: int = 0,
    verify_data_hash: bool = True,
) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    assets = config["assets"]
    data_path = Path(assets["data"])
    if verify_data_hash and sha256_file(data_path) != assets["data_sha256"]:
        raise ValueError("TR1c source data SHA256 mismatch")
    registry_path, registry = require_model_dev_registry(config)
    panel_id = "tr1c_cluster80_dev_v1"
    asset = _panel_asset(registry, panel_id, "train_ids" if split == "train" else "dev_ids")
    ids_path = assets["train_ids" if split == "train" else "validation_ids"]
    if Path(ids_path).resolve() != Path(asset["path"]).resolve():
        raise ValueError("configured ID path differs from the frozen registry")
    selected_ids = _read_ids(ids_path, asset["sha256"])
    records = []
    with data_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            record_id = f"bprna_tr1c:row{row_index:06d}:{raw['name']}"
            if record_id not in selected_ids:
                continue
            sequence = str(raw["seq"]).upper().replace("T", "U")
            records.append({
                "id": record_id,
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "pairs": sorted(tuple(pair) for pair in raw.get("pairs", [])),
            })
    if {row["id"] for row in records} != selected_ids:
        raise ValueError(f"registered {split} IDs do not exactly match TR1c rows")
    records.sort(key=lambda row: (len(row["sequence"]), row["id"]))
    if record_limit:
        eligible = [row for row in records if row["pairs"] and len(row["sequence"]) >= 8]
        if len(eligible) < record_limit:
            raise ValueError("insufficient legal registered records for the smoke fixture")
        records = eligible[:record_limit]
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gec-config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shared-init", required=True)
    parser.add_argument("--gec-init", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-head-steps", type=int, default=1600)
    parser.add_argument("--unfreeze-every", type=int, default=800)
    parser.add_argument("--unfreeze-per-step", type=int, default=2)
    parser.add_argument("--max-unfrozen-blocks", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--validation-min-delta", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--train-record-limit", type=int, default=0)
    parser.add_argument("--validation-record-limit", type=int, default=0)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--smoke-interrupt-after-step", type=int, default=0)
    parser.add_argument("--zchunk", type=int, default=64)
    parser.add_argument("--abort-loss-threshold", type=float, default=0.0)
    parser.add_argument("--abort-gnorm-threshold", type=float, default=0.0)
    return parser


def validate_training_args(args: argparse.Namespace, config: dict) -> None:
    validate_gec_arm_v1(args.arm)
    if os.environ.get("RN_CUDA_INSIDE") != "1":
        raise ValueError("GEC requires RN_CUDA_INSIDE=1; CPU fallback is forbidden")
    training = config["training"]
    exact = {
        "seed": training["seed"], "batch_size": training["batch_size"],
        "head_lr": training["head_lr"], "backbone_lr": training["backbone_lr"],
        "warmup_head_steps": training["warmup_head_steps"],
        "unfreeze_every": training["unfreeze_every"],
        "unfreeze_per_step": training["unfreeze_per_step"],
        "max_unfrozen_blocks": training["max_unfrozen_blocks"],
        "warmup_steps": training["optimizer_warmup_steps"],
        "grad_clip": training["grad_clip"],
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(f"GEC frozen training mismatch for {name}")
    if Path(args.shared_init).resolve() != Path(config["assets"]["shared_init"]).resolve():
        raise ValueError("base shared initialization differs from the frozen config")
    if sha256_file(args.shared_init) != config["assets"]["shared_init_sha256"]:
        raise ValueError("base shared initialization SHA256 mismatch")
    convergence = config["convergence"]
    if (args.validation_patience != convergence["patience"]
            or args.validation_min_delta != convergence["min_delta"]):
        raise ValueError("GEC convergence patience or min-delta mismatch")
    if args.engineering_smoke:
        smoke = config["smoke"]
        expected = {
            "steps": smoke["steps_per_arm"],
            "train_record_limit": smoke["train_record_limit"],
            "validation_record_limit": smoke["validation_record_limit"],
            "validation_every": smoke["validation_every"],
        }
        for name, target in expected.items():
            if getattr(args, name) != target:
                raise ValueError(f"GEC engineering-smoke mismatch for {name}")
        root = Path(smoke["artifact_root"]).resolve()
        if root not in Path(args.out).resolve().parents:
            raise ValueError("smoke output must be under the registered GEC smoke root")
        if args.save_every != 1:
            raise ValueError("smoke requires a resume checkpoint after every step")
        if args.smoke_interrupt_after_step not in (0, smoke["controlled_interrupt_after_step"]):
            raise ValueError("invalid controlled smoke interruption step")
        if smoke["artifact_generation"] != SMOKE_ARTIFACT_GENERATION_V2:
            raise ValueError("smoke artifact generation is not gec_smoke_v2")
        if smoke["validation_destroy_step"] != VALIDATION_DESTROY_STEP_V2:
            raise ValueError("validation destroy identity must be frozen at step zero")
        if smoke["batch_norm_policy"] != BATCH_NORM_POLICY_V2:
            raise ValueError("unexpected BatchNorm train/eval policy")
    elif args.smoke_interrupt_after_step:
        raise ValueError("controlled interruption is restricted to engineering smoke")
    else:
        if args.steps != 0:
            raise ValueError("non-smoke GEC training has no fixed step cap; use --steps 0")
        if args.validation_every != convergence["validation_every"]:
            raise ValueError("non-smoke validation cadence differs from the contract")
        if args.train_record_limit or args.validation_record_limit:
            raise ValueError("record limits are restricted to engineering smoke")


def _locate_blocks(encoder):
    blocks = getattr(encoder, "encoder", None)
    if blocks is None:
        blocks = getattr(encoder, "transformer", None)
    if blocks is not None and hasattr(blocks, "layer"):
        blocks = blocks.layer
    if blocks is None or not hasattr(blocks, "__getitem__"):
        raise RuntimeError(f"cannot locate transformer blocks on {type(encoder)}")
    return blocks


def _load_initialization(args, config, encoder, head, residual):
    initial = torch.load(args.shared_init, map_location="cpu", weights_only=False)
    if initial.get("format") != "prior_alignment_shared_init_v1" or initial.get("seed") != args.seed:
        raise ValueError("base shared initialization format or seed mismatch")
    encoder.load_state_dict(initial["model"]["encoder"], strict=True)
    head.load_state_dict(initial["model"]["head"], strict=True)
    head.use_turner_prior = False
    residual_initial = torch.load(args.gec_init, map_location="cpu", weights_only=False)
    if residual_initial.get("format") != GEC_RESIDUAL_FORMAT_V1:
        raise ValueError("unexpected GEC residual initialization format")
    if residual_initial.get("seed") != args.seed:
        raise ValueError("GEC residual initialization seed mismatch")
    if residual_initial.get("gec_config_sha256") != sha256_file(args.gec_config):
        raise ValueError("GEC residual initialization config hash mismatch")
    if residual_initial.get("base_shared_init_sha256") != sha256_file(args.shared_init):
        raise ValueError("GEC residual/base initialization identity mismatch")
    residual.load_state_dict(residual_initial["model"], strict=True)
    actual = tensor_state_sha256(residual.state_dict())
    if actual != residual_initial["residual_tensor_sha256"]:
        raise ValueError("GEC residual tensor identity mismatch")
    return residual_initial


def _random_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
    }


def _restore_random_state(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


def main() -> None:
    args = build_parser().parse_args()
    config = load_gec_config(args.gec_config)
    validate_training_args(args, config)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = require_explicit_cuda(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "resume.pt"
    meta_path = output / "run_meta.json"
    previous = _read_previous_run_meta(meta_path)
    if resume_path.exists() and previous is None:
        raise ValueError("refusing to resume without a valid previous run_meta.json")
    if previous is not None and not resume_path.exists():
        raise ValueError("refusing to overwrite previous run_meta without resume.pt")
    if not resume_path.exists():
        existing = [name for name in ("train_log.jsonl", "batch_order.jsonl", "validation_log.jsonl")
                    if (output / name).exists() and (output / name).stat().st_size > 0]
        if existing:
            raise ValueError(f"refusing to mix a fresh run with existing ledgers: {existing}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    train_records = load_registered_records(config, "train", args.train_record_limit)
    validation_records = load_registered_records(
        config, "validation", args.validation_record_limit, verify_data_hash=False,
    )
    train_ids = {row["id"] for row in train_records}
    validation_ids = {row["id"] for row in validation_records}
    if train_ids & validation_ids:
        raise ValueError("train/development ID overlap")
    if {row["sequence_sha256"] for row in train_records} & {
        row["sequence_sha256"] for row in validation_records
    }:
        raise ValueError("train/development exact-sequence overlap")
    if not train_records or (args.steps == 0 and not validation_records):
        raise ValueError("training records and unbounded-run validation records are required")

    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(
        d_model=int(report["d_model"]), d_z=config["model"]["d_z"], hidden=64,
        scorer="resnet2d", chunk_size=0, resnet_blocks=4, resnet_hidden=64,
        use_turner_prior=True,
    )
    residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    residual_initial = _load_initialization(args, config, encoder, head, residual)
    encoder = encoder.to(device)
    head = head.to(device)
    residual = residual.to(device)
    encoder.train()
    head.train()
    residual.train()
    for module in encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    blocks = _locate_blocks(encoder)
    if len(blocks) != config["training"]["n_blocks"]:
        raise ValueError("encoder block count differs from the frozen 33-block contract")

    head_parameters = list(head.parameters()) + list(residual.parameters())
    encoder_parameters = list(encoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.head_lr},
        {"params": encoder_parameters, "lr": args.backbone_lr},
    ], weight_decay=0.0, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)),
    )
    stream_records = [(row["sequence"], row["pairs"]) for row in train_records]
    stream = BatchStream(stream_records, args.batch_size, args.seed)
    convergence = GECConvergenceStateV1(
        terminal_step=config["training"]["terminal_step"],
        terminal_unfrozen_blocks=args.max_unfrozen_blocks,
        patience=args.validation_patience,
        min_delta=args.validation_min_delta,
    )
    initial_residual_hash = tensor_state_sha256(residual.state_dict())
    if args.engineering_smoke:
        artifact_path = Path(config["smoke"]["artifact_root"]) / "smoke_artifact_identity.json"
        if not artifact_path.is_file():
            raise ValueError("missing current gec_smoke_v2 artifact identity document")
        artifact = json.loads(artifact_path.read_text())
        expected_artifact = build_smoke_artifact_identity(
            args.gec_config, args.shared_init, args.gec_init,
        )
        validate_smoke_artifact_identity(artifact, expected_artifact)
    run_identity = build_run_identity(
        args, config, train_records, validation_records, residual_initial,
    )
    n_unfrozen = 0
    start_step = 0
    optimizer_steps = 0
    meta = {
        "schema_version": RUN_META_SCHEMA_V2,
        "status": "running",
        "run_identity": run_identity,
        "arm": args.arm,
        "engineering_smoke": args.engineering_smoke,
        "scientific_training_authorized": False,
        "gec_config": str(Path(args.gec_config).resolve()),
        "gec_config_sha256": sha256_file(args.gec_config),
        "source_contract": config["source_contract"],
        "source_contract_sha256": sha256_file(config["source_contract"]),
        "model_dev_registry": run_identity["model_dev_registry"],
        "model_dev_registry_sha256": MODEL_DEV_REGISTRY_SHA256,
        "train_data_sha256": run_identity["train_data_sha256"],
        "train_ids_sha256": run_identity["train_ids_sha256"],
        "validation_ids_sha256": run_identity["validation_ids_sha256"],
        "shared_init": args.shared_init,
        "shared_init_sha256": sha256_file(args.shared_init),
        "gec_init": args.gec_init,
        "gec_init_sha256": sha256_file(args.gec_init),
        "initial_residual_tensor_sha256": initial_residual_hash,
        "residual_parameter_count": sum(p.numel() for p in residual.parameters()),
        "trainable_head_and_residual_parameter_count": sum(p.numel() for p in head_parameters),
        "train_record_ids": [row["id"] for row in train_records],
        "validation_record_ids": [row["id"] for row in validation_records],
        "objective": {
            "name": "exact_structured_NLL_only", "lambda_nll": 1.0,
            "normalization": "length", "all_auxiliary_weights": 0.0,
            "first_pass_auxiliary_loss": False,
        },
        "schedule": vars(args),
        "convergence": convergence.state_dict(),
        "environment": {
            "requested_device": str(device), "resolved_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "decoder": config["model"]["decoder"],
        "validation_destroy_step": VALIDATION_DESTROY_STEP_V2,
        "batch_norm_policy": BATCH_NORM_POLICY_V2,
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "tf32": False,
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "compute_kernels": {
            "inside_outside": "triton_cuda_float64",
            "cpu_fallback": False,
            "RN_CUDA_INSIDE": os.environ["RN_CUDA_INSIDE"],
        },
        "resume_history": [],
        "controlled_interruption_steps": [],
        "backbone_report": report,
        "steps_completed": 0,
        "optimizer_steps": 0,
        "converged": False,
    }
    if previous is not None:
        validate_run_identity(previous, run_identity)
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(state, run_identity)
        if previous.get("resume_checkpoint_sha256") is not None:
            if sha256_file(resume_path) != previous["resume_checkpoint_sha256"]:
                raise ValueError("resume checkpoint SHA256 differs from previous run_meta")
        if previous["status"] == "converged":
            best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            validate_converged_reentry(previous, state, best, run_identity)
            print(f"[gec] already converged arm={args.arm}; restored_best_step={state['step']}; no further training", flush=True)
            return
        if previous["status"] == "engineering_smoke_completed":
            if int(state["step"]) != args.steps or int(state["optimizer_steps"]) != args.steps:
                raise ValueError("completed smoke checkpoint step mismatch")
            print(f"[gec] engineering smoke already complete arm={args.arm}; no further training", flush=True)
            return
        if state.get("checkpoint_phase") != "latest":
            raise ValueError("non-terminal resume must use a latest-state checkpoint")
        meta["resume_history"] = list(previous.get("resume_history", [])) + [{
            "from_step": int(state["step"]),
            "previous_status": previous["status"],
            "resume_sha256": sha256_file(resume_path),
        }]
        meta["controlled_interruption_steps"] = list(previous.get("controlled_interruption_steps", []))
    else:
        _atomic_json(meta, meta_path)

    if previous is not None:
        encoder.load_state_dict(state["encoder"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        residual.load_state_dict(state["residual"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_scheduler_state(scheduler, state["scheduler"], state["optimizer"])
        n_unfrozen = int(state["n_unfrozen"])
        restore_unfrozen_blocks(blocks, n_unfrozen)
        start_step = int(state["step"])
        optimizer_steps = int(state["optimizer_steps"])
        restore_batch_stream(stream, state["stream_state"], args.batch_size)
        convergence.load_state_dict(state["convergence"])
        _restore_random_state(state["random_state"])
        for name in ("train_log.jsonl", "batch_order.jsonl", "compute_ledger.jsonl", "validation_log.jsonl"):
            path = output / name
            if path.exists():
                retained = [line for line in path.read_text().splitlines(keepends=True)
                            if line.strip() and json.loads(line)["step"] <= start_step]
                path.write_text("".join(retained))
        print(f"[gec] resumed arm={args.arm} step={start_step} unfrozen={n_unfrozen}/33", flush=True)

    def score_batch(batch_records, step: int, epoch: int):
        sequences = [row["sequence"] for row in batch_records]
        lengths = [len(sequence) for sequence in sequences]
        max_length = max(lengths)
        batch_size = len(batch_records)
        encoder_ids = torch.zeros(batch_size, max_length + 2, dtype=torch.long)
        attention = torch.zeros_like(encoder_ids)
        head_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            tokens = [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence]
            framed = [VOCAB["<cls>"]] + tokens + [VOCAB["<eos>"]]
            encoder_ids[index, :len(framed)] = torch.tensor(framed)
            attention[index, :len(framed)] = 1
            head_ids[index, :len(sequence)] = torch.tensor(
                [BASE_TO_INDEX.get(char, UNK_INDEX) for char in sequence]
            )
        encoder_ids = encoder_ids.to(device)
        attention = attention.to(device)
        head_ids = head_ids.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = encoder(input_ids=encoder_ids, attention_mask=attention)
        hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        hidden = hidden[:, 1:max_length + 1, :].float().to(torch.float16).to(torch.float32)
        for index, length in enumerate(lengths):
            if length < max_length:
                hidden[index, length:, :] = 0.0
        zchunk = args.zchunk if args.zchunk > 0 else max_length
        chunks = [
            gradient_checkpoint(head.pair_repr.cross, hidden, hidden[:, start:start + zchunk],
                                use_reentrant=False)
            for start in range(0, max_length, zchunk)
        ]
        pair_representation = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
        batch_norm_state = _batch_norm_running_state(head)
        raw_scores = gradient_checkpoint(
            head.score_pair_matrix, pair_representation, head_ids, False, None, None,
            use_reentrant=False,
        )
        lengths_tensor = torch.tensor(lengths, device=device)
        base_scores = head.calibration(raw_scores, lengths_tensor)
        full_mask = pair_mask_from_ids(head_ids, head.min_loop).to(device)
        base_scores = torch.where(
            full_mask, base_scores, torch.full_like(base_scores, NEG_BIG),
        )
        losses = []
        final_scores = []
        context_log_z = []
        for index, row in enumerate(batch_records):
            length = lengths[index]
            mask = torch.as_tensor(harness.valid_pair_mask(row["sequence"]), device=device)
            first_pass = base_scores[index, :length, :length]
            numerics = compute_gec_context_v1(
                first_pass, mask, mode=args.arm,
                shuffle_key=gec_destroy_key_v1(args.seed, epoch, step, row["sequence"]),
            )
            geometry = gec_geometry_v1(mask, head.min_loop)
            delta = residual(
                pair_representation[index, :length, :length],
                numerics.context, geometry, mask,
            )
            scores = torch.where(
                mask, first_pass + delta, torch.full_like(first_pass, NEG_BIG),
            )
            loss = harness.negative_log_likelihood(
                scores, mask, row["pairs"], normalization="length",
            )
            if not isinstance(loss, torch.Tensor) or loss.device != device:
                raise RuntimeError("CUDA structured NLL left the requested device")
            losses.append(loss)
            final_scores.append(scores)
            context_log_z.append(float(numerics.log_z.detach()))
        return torch.stack(losses).mean(), final_scores, context_log_z, batch_norm_state

    def evaluate(step: int):
        encoder.eval()
        head.eval()
        residual.eval()
        losses = []
        try:
            with torch.no_grad():
                for row in validation_records:
                    loss, _, _, _ = score_batch([row], VALIDATION_DESTROY_STEP_V2, 0)
                    try:
                        losses.append(require_finite_validation_nll(loss.detach(), step, row))
                    except FloatingPointError as error:
                        _append_jsonl(validation_error_record(step, error, row),
                                      output / "validation_error.jsonl")
                        raise
            aggregate = float(np.mean(losses))
            try:
                return require_finite_validation_nll(aggregate, step)
            except FloatingPointError as error:
                _append_jsonl(validation_error_record(step, error, losses=losses),
                              output / "validation_error.jsonl")
                raise
        finally:
            encoder.train()
            head.train()
            residual.train()

    def training_checkpoint(step: int, phase: str, terminal_status: str = "running"):
        return {
            "format": "foldnar.gec_checkpoint.v2",
            "checkpoint_phase": phase,
            "terminal_status": terminal_status,
            "run_identity": run_identity,
            "step": int(step), "encoder": encoder.state_dict(), "head": head.state_dict(),
            "residual": residual.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "n_unfrozen": n_unfrozen,
            "optimizer_steps": optimizer_steps, "stream_state": stream.state,
            "convergence": convergence.state_dict(), "random_state": _random_state(),
            "model_tensor_sha256": model_tensor_hashes(encoder, head, residual),
            "validation_destroy_step": VALIDATION_DESTROY_STEP_V2,
            "batch_norm_policy": BATCH_NORM_POLICY_V2,
        }

    def save_resume(step: int, phase: str = "latest", terminal_status: str = "running"):
        atomic_torch_save(training_checkpoint(step, phase, terminal_status), resume_path)

    step = start_step
    divergence = ""
    early_stopped = False
    iterator = range(start_step + 1, args.steps + 1) if args.steps else itertools.count(start_step + 1)
    for step in iterator:
        due = capped_unfreeze_blocks(
            step, args.warmup_head_steps, args.unfreeze_every, args.unfreeze_per_step,
            len(blocks), args.max_unfrozen_blocks,
        )
        while n_unfrozen < due:
            for parameter in blocks[len(blocks) - 1 - n_unfrozen].parameters():
                parameter.requires_grad_(True)
            n_unfrozen += 1
        epoch = stream.state["epoch"]
        bucket = stream.next()
        batch_records = [train_records[index] for index in bucket]
        _append_jsonl({
            "step": step, "epoch": epoch, "record_ids": [row["id"] for row in batch_records],
            "sequence_sha256": [row["sequence_sha256"] for row in batch_records],
        }, output / "batch_order.jsonl")
        loss, _, log_z, batch_norm_state = score_batch(batch_records, step, epoch)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
        finally:
            _restore_batch_norm_running_state(batch_norm_state)
        trainable = [parameter for parameter in head_parameters + encoder_parameters
                     if parameter.requires_grad]
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in trainable):
            divergence = "NON_FINITE_LOSS_OR_GRADIENT_NORM"
            gradient_norm = float("nan")
        else:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        residual_gradient_norm = math.sqrt(sum(
            float(parameter.grad.detach().square().sum()) for parameter in residual.parameters()
            if parameter.grad is not None
        ))
        if not divergence:
            divergence = divergence_reason(
                float(loss.detach()), gradient_norm,
                args.abort_loss_threshold, args.abort_gnorm_threshold,
            )
        if not divergence:
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
        residual_hash = tensor_state_sha256(residual.state_dict())
        _append_jsonl({
            "step": step, "arm": args.arm, "loss": _finite_float_or_none(loss.detach()),
            "context_log_z": [_finite_float_or_none(value) for value in log_z],
            "gradient_norm": _finite_float_or_none(gradient_norm),
            "residual_gradient_norm": _finite_float_or_none(residual_gradient_norm),
            "residual_tensor_sha256": residual_hash,
            "residual_changed_from_initial": residual_hash != initial_residual_hash,
            "unfrozen_blocks": n_unfrozen, "optimizer_step": not divergence,
            "optimizer_steps": optimizer_steps, "divergence_reason": divergence,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "elapsed_seconds": time.time() - started,
        }, output / "train_log.jsonl")
        _append_jsonl({
            "step": step, "lengths": [len(row["sequence"]) for row in batch_records],
            "record_ids": [row["id"] for row in batch_records], "unfrozen": n_unfrozen,
            "optimizer_step": not divergence, "trainable_params": sum(p.numel() for p in trainable),
        }, output / "compute_ledger.jsonl")
        print(f"[gec] arm={args.arm} step={step}/{args.steps or 'unbounded'} "
              f"loss={float(loss.detach()):.8f} gnorm={gradient_norm:.6g} "
              f"unfrozen={n_unfrozen}/33", flush=True)
        if divergence:
            break
        if args.save_every > 0 and step % args.save_every == 0:
            save_resume(step)
        if validation_records and args.validation_every > 0 and step % args.validation_every == 0:
            validation_loss = evaluate(step)
            decision = convergence.observe(step, validation_loss, n_unfrozen)
            record = build_validation_log_record(step, validation_loss, decision, convergence, n_unfrozen)
            _append_jsonl(record, output / "validation_log.jsonl")
            if decision["save_best"]:
                best_payload = training_checkpoint(step, "best_eligible")
                best_payload["model"] = {
                    "encoder": encoder.state_dict(), "head": head.state_dict(),
                    "residual": residual.state_dict(),
                }
                best_payload["validation_loss"] = validation_loss
                atomic_torch_save(best_payload, output / "best.pt")
            if decision["stop"]:
                early_stopped = True
        if args.smoke_interrupt_after_step == step:
            meta.update({
                "status": "interrupted", "steps_completed": int(step),
                "optimizer_steps": optimizer_steps, "stop_reason": "ENGINEERING_SMOKE_CONTROLLED_INTERRUPT",
                "convergence": convergence.state_dict(),
                "final_residual_tensor_sha256": tensor_state_sha256(residual.state_dict()),
                "residual_changed_from_initial": tensor_state_sha256(residual.state_dict()) != initial_residual_hash,
                "final_unfrozen_blocks": n_unfrozen,
                "controlled_interruption_steps": meta["controlled_interruption_steps"] + [int(step)],
                "resume_checkpoint_sha256": sha256_file(resume_path),
                "elapsed_seconds": time.time() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            })
            _atomic_json(meta, meta_path)
            print(f"[gec] CONTROLLED_INTERRUPT arm={args.arm} step={step}", flush=True)
            return
        if early_stopped:
            break

    training_stopped_step = int(step)
    training_stopped_optimizer_steps = optimizer_steps
    training_stopped_convergence = convergence.state_dict()
    best_convergence = None
    if early_stopped:
        best_path = output / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(best, run_identity)
        if best.get("checkpoint_phase") != "best_eligible" or int(best["n_unfrozen"]) != args.max_unfrozen_blocks:
            raise ValueError("refusing to restore a non-best or pre-terminal checkpoint")
        encoder.load_state_dict(best["model"]["encoder"], strict=True)
        head.load_state_dict(best["model"]["head"], strict=True)
        residual.load_state_dict(best["model"]["residual"], strict=True)
        optimizer.load_state_dict(best["optimizer"])
        restore_scheduler_state(scheduler, best["scheduler"], best["optimizer"])
        n_unfrozen = int(best["n_unfrozen"])
        restore_unfrozen_blocks(blocks, n_unfrozen)
        optimizer_steps = int(best["optimizer_steps"])
        restore_batch_stream(stream, best["stream_state"], args.batch_size)
        convergence.load_state_dict(best["convergence"])
        best_convergence = convergence.state_dict()
        _restore_random_state(best["random_state"])
        step = int(best["step"])
        save_resume(step, "restored_best_eligible", "converged")
        meta["best_checkpoint_sha256"] = sha256_file(best_path)
        meta["restored_best_step"] = step
        meta["training_stopped_step"] = training_stopped_step
        meta["training_stopped_optimizer_steps"] = training_stopped_optimizer_steps
    elif not resume_path.exists() or training_stopped_step % max(1, args.save_every) != 0:
        save_resume(step)
    final_hash = tensor_state_sha256(residual.state_dict())
    meta.update({
        "status": "diverged" if divergence else (
            "converged" if early_stopped else "engineering_smoke_completed"
        ),
        "steps_completed": training_stopped_step, "optimizer_steps": optimizer_steps,
        "stop_reason": divergence or (
            config["convergence"]["stop_reason"] if early_stopped else "ENGINEERING_SMOKE_TWO_STEP_CAP"
        ),
        "divergence_reason": divergence, "converged": early_stopped,
        **build_terminal_convergence_metadata(training_stopped_convergence, best_convergence),
        "final_residual_tensor_sha256": final_hash,
        "residual_changed_from_initial": final_hash != initial_residual_hash,
        "final_unfrozen_blocks": n_unfrozen,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    })
    if resume_path.exists():
        meta["resume_checkpoint_sha256"] = sha256_file(resume_path)
    _atomic_json(meta, meta_path)
    print(f"[gec] DONE arm={args.arm} status={meta['status']} step={step} "
          f"optimizer_steps={optimizer_steps} residual_changed={final_hash != initial_residual_hash}", flush=True)


if __name__ == "__main__":
    main()
