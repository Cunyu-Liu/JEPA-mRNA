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


def load_gec_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
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
    registry_path = Path(assets["registry"])
    if not registry_path.is_absolute():
        registry_path = REPO / registry_path
    registry = json.loads(registry_path.read_text())
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
    parser.add_argument("--zchunk", type=int, default=64)
    parser.add_argument("--abort-loss-threshold", type=float, default=0.0)
    parser.add_argument("--abort-gnorm-threshold", type=float, default=0.0)
    return parser


def validate_training_args(args: argparse.Namespace, config: dict) -> None:
    validate_gec_arm_v1(args.arm)
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
    device = require_explicit_cuda(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "resume.pt"
    meta_path = output / "run_meta.json"
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
    n_unfrozen = 0
    start_step = 0
    optimizer_steps = 0
    meta = {
        "schema_version": "foldnar.gec_training_run.v1",
        "status": "running",
        "arm": args.arm,
        "engineering_smoke": args.engineering_smoke,
        "scientific_training_authorized": False,
        "gec_config": str(Path(args.gec_config).resolve()),
        "gec_config_sha256": sha256_file(args.gec_config),
        "source_contract": config["source_contract"],
        "source_contract_sha256": sha256_file(config["source_contract"]),
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
        "backbone_report": report,
        "steps_completed": 0,
        "optimizer_steps": 0,
        "converged": False,
    }
    _atomic_json(meta, meta_path)

    if resume_path.exists():
        previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        identity_keys = (
            "arm", "gec_config_sha256", "shared_init_sha256", "gec_init_sha256",
            "initial_residual_tensor_sha256", "train_record_ids", "validation_record_ids",
        )
        for key in identity_keys:
            if previous.get(key) != meta[key]:
                raise ValueError(f"resume identity mismatch: {key}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
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
        return torch.stack(losses).mean(), final_scores, context_log_z

    def evaluate(step: int):
        encoder.eval()
        head.eval()
        residual.eval()
        losses = []
        with torch.no_grad():
            for row in validation_records:
                loss, _, _ = score_batch([row], step, 0)
                losses.append(float(loss.detach()))
        encoder.train()
        head.train()
        residual.train()
        return float(np.mean(losses))

    def save_resume(step: int):
        atomic_torch_save({
            "step": step, "encoder": encoder.state_dict(), "head": head.state_dict(),
            "residual": residual.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "n_unfrozen": n_unfrozen,
            "optimizer_steps": optimizer_steps, "stream_state": stream.state,
            "convergence": convergence.state_dict(), "random_state": _random_state(),
        }, resume_path)

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
        loss, _, log_z = score_batch(batch_records, step, epoch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        if validation_records and args.validation_every > 0 and step % args.validation_every == 0:
            validation_loss = evaluate(step)
            decision = convergence.observe(step, validation_loss, n_unfrozen)
            record = {
                "step": step, "validation_loss": validation_loss,
                "n_unfrozen": n_unfrozen, **decision,
                "best_step": convergence.best_step,
                "best_validation_loss": convergence.best_validation_loss,
                "bad_evaluations": convergence.bad_evaluations,
            }
            _append_jsonl(record, output / "validation_log.jsonl")
            if decision["save_best"]:
                atomic_torch_save({
                    "step": step, "model": {
                        "encoder": encoder.state_dict(), "head": head.state_dict(),
                        "residual": residual.state_dict(),
                    },
                    "validation_loss": validation_loss, "n_unfrozen": n_unfrozen,
                    "gec_config_sha256": meta["gec_config_sha256"],
                    "arm": args.arm, "convergence": convergence.state_dict(),
                }, output / "best.pt")
            if decision["stop"]:
                early_stopped = True
        if args.save_every > 0 and step % args.save_every == 0:
            save_resume(step)
        if early_stopped:
            break

    if early_stopped:
        best_path = output / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if int(best["n_unfrozen"]) != args.max_unfrozen_blocks:
            raise ValueError("refusing to restore a pre-terminal checkpoint")
        encoder.load_state_dict(best["model"]["encoder"], strict=True)
        head.load_state_dict(best["model"]["head"], strict=True)
        residual.load_state_dict(best["model"]["residual"], strict=True)
        meta["best_checkpoint_sha256"] = sha256_file(best_path)
    if not resume_path.exists() or (early_stopped and args.save_every <= 0):
        save_resume(step)
    final_hash = tensor_state_sha256(residual.state_dict())
    meta.update({
        "status": "diverged" if divergence else (
            "converged" if early_stopped else "engineering_smoke_completed"
        ),
        "steps_completed": int(step), "optimizer_steps": optimizer_steps,
        "stop_reason": divergence or (
            config["convergence"]["stop_reason"] if early_stopped else "ENGINEERING_SMOKE_TWO_STEP_CAP"
        ),
        "divergence_reason": divergence, "converged": early_stopped,
        "convergence": convergence.state_dict(),
        "final_residual_tensor_sha256": final_hash,
        "residual_changed_from_initial": final_hash != initial_residual_hash,
        "final_unfrozen_blocks": n_unfrozen,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    })
    _atomic_json(meta, meta_path)
    print(f"[gec] DONE arm={args.arm} status={meta['status']} step={step} "
          f"optimizer_steps={optimizer_steps} residual_changed={final_hash != initial_residual_hash}", flush=True)


if __name__ == "__main__":
    main()
