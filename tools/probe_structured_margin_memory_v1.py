#!/usr/bin/env python3
"""One bounded B=4/L=500/terminal-20 structured-margin resource probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa import harness
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.gec_context import (
    GECResidualV1,
    gec_destroy_key_v1,
    gec_geometry_v1,
    require_explicit_cuda,
)
from rnajepa.gec_numeric_v1 import compute_gec_context_v1
from rnajepa.prior_alignment import sha256_file
from rnajepa.structured_margin import pair_hamming_structured_margin
from rinalmo_preflight import VOCAB, load_encoder
from tools.train_plan_a_clean_head_p1 import NEG_BIG, restore_unfrozen_blocks
from tools.train_plan_a_gec_convergence_v1 import (
    _batch_norm_running_state,
    _load_initialization,
    _locate_blocks,
    _restore_batch_norm_running_state,
    load_gec_config,
    load_registered_records,
)
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

GIB = 1024 ** 3
MIB = 1024 ** 2
BATCH_SIZE = 4
LENGTH = 500
TERMINAL_UNFROZEN_BLOCKS = 20
DEFAULT_MIN_FREE_GIB = 24.0
DEFAULT_OUT = Path(
    "/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_memory_probe_v1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def gpu_snapshot(device_index: int) -> dict:
    fields = "timestamp,index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu"
    output = subprocess.check_output([
        "nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
    ], text=True)
    rows = [line.split(", ") for line in output.splitlines() if line.strip()]
    row = next((values for values in rows if int(values[1]) == device_index), None)
    if row is None:
        raise ValueError(f"nvidia-smi has no GPU index {device_index}")
    process_output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, used = line.split(", ")
        if uuid == row[3]:
            processes.append({
                "pid": int(pid), "process_name": name, "used_memory_mib": int(used),
            })
    return {
        "timestamp": row[0], "index": int(row[1]), "name": row[2], "uuid": row[3],
        "memory_total_mib": int(row[4]), "memory_used_mib": int(row[5]),
        "memory_free_mib": int(row[6]), "utilization_gpu_percent": int(row[7]),
        "compute_processes": processes,
    }


def probe_gate(free_mib: int, min_free_gib: float) -> dict:
    required_mib = math.ceil(min_free_gib * 1024)
    sufficient = free_mib >= required_mib
    return {
        "status": "RUN_ALLOWED" if sufficient else "QUEUED_INSUFFICIENT_FREE_MEMORY",
        "free_mib": free_mib,
        "required_free_mib": required_mib,
        "sufficient": sufficient,
        "policy": "run one bounded probe only when the preflight free-memory gate passes; never kill or preempt",
    }


def select_worst_case_batch(records: list[dict]) -> list[dict]:
    eligible = [row for row in records if len(row["sequence"]) == LENGTH]
    if len(eligible) < BATCH_SIZE:
        raise ValueError(f"need four real length-{LENGTH} training records; found {len(eligible)}")
    return sorted(eligible, key=lambda row: (-len(row["pairs"]), row["id"]))[:BATCH_SIZE]


def mask_sha256(mask: torch.Tensor) -> str:
    packed = torch.where(mask, 1, 0).to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(packed).hexdigest()


def batch_mask_metadata(batch_records: list[dict]) -> list[dict]:
    metadata = []
    for row in batch_records:
        mask = torch.as_tensor(harness.valid_pair_mask(row["sequence"]), dtype=torch.bool)
        legal_pairs = {(int(i), int(j)) for i, j in zip(*torch.where(mask))}
        target_pairs = {(int(i), int(j)) for i, j in row["pairs"]}
        if not target_pairs <= legal_pairs:
            raise ValueError(f"non-canonical or illegal target pair in {row['id']}")
        metadata.append({
            "record_id": row["id"], "sequence_length": len(row["sequence"]),
            "sequence_sha256": row["sequence_sha256"],
            "canonical_pair_count": len(legal_pairs),
            "target_pair_count": len(target_pairs),
            "canonical_mask_sha256": mask_sha256(mask),
            "target_pairs_sha256": hashlib.sha256(
                json.dumps(sorted(target_pairs), separators=(",", ":")).encode()
            ).hexdigest(),
        })
    return metadata


def release_thresholds(peak_bytes: int, gpu_total_bytes: int) -> dict:
    if peak_bytes <= 0 or gpu_total_bytes <= 0:
        raise ValueError("peak and GPU memory must be positive")
    reserve_bytes = math.ceil(max(4 * GIB, 0.20 * gpu_total_bytes))
    single = math.ceil(1.5 * peak_bytes + reserve_bytes)
    dual = math.ceil(1.5 * (2 * peak_bytes) + reserve_bytes)
    return {
        "formula": "ceil(1.5 * arms * measured_peak_bytes + max(4 GiB, 0.20 * GPU_total_bytes))",
        "measured_peak_basis": "max(torch.cuda.max_memory_allocated, torch.cuda.max_memory_reserved)",
        "measured_peak_bytes": peak_bytes,
        "gpu_total_bytes": gpu_total_bytes,
        "reserve_bytes": reserve_bytes,
        "single_arm": {
            "arms": 1, "required_free_bytes": single,
            "required_free_gib": single / GIB,
        },
        "dual_arm": {
            "arms": 2, "required_free_bytes": dual,
            "required_free_gib": dual / GIB,
        },
    }


def structured_margin_forward(batch_records, *, step: int, epoch: int, arm: str,
                              seed: int, zchunk: int, device, config,
                              encoder, head, residual):
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
    zchunk = zchunk if zchunk > 0 else max_length
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
    context_log_z = []
    for index, row in enumerate(batch_records):
        length = lengths[index]
        mask = torch.as_tensor(harness.valid_pair_mask(row["sequence"]), device=device)
        first_pass = base_scores[index, :length, :length]
        numerics = compute_gec_context_v1(
            first_pass, mask, mode=arm,
            shuffle_key=gec_destroy_key_v1(seed, epoch, step, row["sequence"]),
        )
        geometry = gec_geometry_v1(mask, head.min_loop)
        delta = residual(
            pair_representation[index, :length, :length],
            numerics.context, geometry, mask,
        )
        scores = torch.where(
            mask, first_pass + delta, torch.full_like(first_pass, NEG_BIG),
        )
        loss = pair_hamming_structured_margin(
            scores, mask, row["pairs"], normalization="length",
        )
        if loss.device != device or loss.dtype != torch.float64:
            raise RuntimeError("structured margin is not float64 on the requested CUDA device")
        losses.append(loss)
        context_log_z.append(float(numerics.log_z.detach()))
    return torch.stack(losses).mean(), context_log_z, batch_norm_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gec-config", default=str(REPO / "configs/gec_structured_margin_convergence_v1.json"))
    parser.add_argument("--arm", default="ensemble_context", choices=(
        "matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context",
    ))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--zchunk", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device != "cuda:0":
        raise ValueError("this resource probe requires explicit --device cuda:0")
    if args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "resource_probe_v1.json"
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if previous.get("status") == "COMPLETED_FORWARD_BACKWARD_RESOURCE_PROBE_ONLY":
            raise ValueError("completed probe already exists; use a new --out for a reproducibility rerun")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    preflight = gpu_snapshot(0)
    gate = probe_gate(preflight["memory_free_mib"], args.min_free_gib)
    atomic_json(preflight, args.out / "initial_gpu_snapshot_v1.json")
    (args.out / "command_v1.txt").write_text(command + "\n")
    if not gate["sufficient"]:
        atomic_json({
            "schema_version": "foldnar.structured_margin_memory_probe.v1",
            "status": gate["status"], "created_at": utc_now(), "command": command,
            "code_commit": repo_commit(),
            "full_training_started": False, "optimizer_steps": 0,
            "preflight": preflight, "gate": gate,
            "action": "leave queued and rerun the same command after memory is available",
        }, output_path)
        print(json.dumps({"status": gate["status"], "gate": gate}, indent=2))
        return

    config = load_gec_config(args.gec_config)
    if config["training"]["batch_size"] != BATCH_SIZE or config["training"]["max_unfrozen_blocks"] != TERMINAL_UNFROZEN_BLOCKS:
        raise ValueError("resource probe B=4/terminal-20 contract mismatch")
    if config["training"]["seed"] != 0 or config["model"]["decoder"]["search_allowed"]:
        raise ValueError("resource probe is restricted to seed0 with no decoder search")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = require_explicit_cuda(args.device)
    torch.manual_seed(0)
    started = time.time()
    records = load_registered_records(config, "train")
    batch_records = select_worst_case_batch(records)
    metadata = batch_mask_metadata(batch_records)

    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(
        d_model=int(report["d_model"]), d_z=config["model"]["d_z"], hidden=64,
        scorer="resnet2d", chunk_size=0, resnet_blocks=4, resnet_hidden=64,
        use_turner_prior=True,
    )
    residual = GECResidualV1(config["model"]["d_z"], config["model"]["residual_hidden"])
    init_args = argparse.Namespace(
        shared_init=config["assets"]["shared_init"],
        gec_init="/teams/HoulabData_1788502323/liucunyu/foldnar/gec_smoke_v2/generation_02/gec_shared_residual_init_seed0.pt",
        seed=0, gec_config=args.gec_config,
    )
    _load_initialization(init_args, config, encoder, head, residual)
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
    restore_unfrozen_blocks(blocks, TERMINAL_UNFROZEN_BLOCKS)
    unfrozen_indices = [
        index for index, block in enumerate(blocks)
        if any(parameter.requires_grad for parameter in block.parameters())
    ]
    if unfrozen_indices != list(range(len(blocks) - TERMINAL_UNFROZEN_BLOCKS, len(blocks))):
        raise RuntimeError("terminal unfreeze did not enable exactly the last 20 blocks")

    properties = torch.cuda.get_device_properties(device)
    device_identity = {
        "requested": args.device, "resolved": str(device), "index": device.index,
        "name": properties.name, "uuid": str(properties.uuid),
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    wall_start = time.perf_counter()
    loss, context_log_z, batch_norm_state = structured_margin_forward(
        batch_records, step=config["training"]["terminal_step"], epoch=0,
        arm=args.arm, seed=0, zchunk=args.zchunk, device=device, config=config,
        encoder=encoder, head=head, residual=residual,
    )
    forward_end.record()
    torch.cuda.synchronize(device)
    forward_wall_seconds = time.perf_counter() - wall_start
    forward_peak_allocated = torch.cuda.max_memory_allocated(device)
    forward_peak_reserved = torch.cuda.max_memory_reserved(device)
    backward_start.record()
    backward_wall_start = time.perf_counter()
    try:
        loss.backward()
    finally:
        _restore_batch_norm_running_state(batch_norm_state)
    backward_end.record()
    torch.cuda.synchronize(device)
    backward_wall_seconds = time.perf_counter() - backward_wall_start
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    forward_cuda_ms = forward_start.elapsed_time(forward_end)
    backward_cuda_ms = backward_start.elapsed_time(backward_end)
    if not math.isfinite(float(loss.detach())) or not all(math.isfinite(x) for x in context_log_z):
        raise FloatingPointError("non-finite structured-margin resource probe")
    trainable = [p for p in list(encoder.parameters()) + list(head.parameters()) + list(residual.parameters()) if p.requires_grad]
    if any(p.grad is None for p in residual.parameters()):
        raise RuntimeError("a residual parameter has no backward gradient")
    if any(p.grad is None for block in blocks[-TERMINAL_UNFROZEN_BLOCKS:] for p in block.parameters()):
        raise RuntimeError("a terminal-20 encoder parameter has no backward gradient")
    gradients = [p.grad for p in trainable if p.grad is not None]
    if any(not torch.isfinite(grad).all() for grad in gradients):
        raise FloatingPointError("non-finite gradient in structured-margin resource probe")

    peak_basis = max(peak_allocated, peak_reserved)
    result = {
        "schema_version": "foldnar.structured_margin_memory_probe.v1",
        "status": "COMPLETED_FORWARD_BACKWARD_RESOURCE_PROBE_ONLY",
        "created_at": utc_now(), "command": command, "code_commit": repo_commit(),
        "scientific_training_authorized": False, "full_training_started": False,
        "optimizer_steps": 0, "decoder_search": False, "seed": 0, "arm": args.arm,
        "batch_size": BATCH_SIZE, "max_sequence_length": LENGTH,
        "terminal_step": config["training"]["terminal_step"],
        "unfrozen_blocks": TERMINAL_UNFROZEN_BLOCKS, "unfrozen_block_indices": unfrozen_indices,
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "objective": config["objective"], "margin_loss": float(loss.detach()),
        "margin_internal_dtype": str(loss.dtype), "gec_context_log_z": context_log_z,
        "batch": metadata,
        "timing": {
            "forward_wall_seconds": forward_wall_seconds,
            "backward_wall_seconds": backward_wall_seconds,
            "forward_backward_wall_seconds": forward_wall_seconds + backward_wall_seconds,
            "forward_cuda_event_seconds": forward_cuda_ms / 1000,
            "backward_cuda_event_seconds": backward_cuda_ms / 1000,
            "forward_backward_cuda_event_seconds": (forward_cuda_ms + backward_cuda_ms) / 1000,
            "total_script_wall_seconds": time.time() - started,
        },
        "memory": {
            "allocated_before_probe_bytes": allocated_before,
            "reserved_before_probe_bytes": reserved_before,
            "forward_peak_allocated_bytes": forward_peak_allocated,
            "forward_peak_reserved_bytes": forward_peak_reserved,
            "forward_backward_peak_allocated_bytes": peak_allocated,
            "forward_backward_peak_reserved_bytes": peak_reserved,
            "release_peak_basis_bytes": peak_basis,
            "release_peak_basis_gib": peak_basis / GIB,
        },
        "device": device_identity,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "RN_CUDA_INSIDE": os.environ.get("RN_CUDA_INSIDE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "preflight": preflight, "gate": gate,
        "postflight": gpu_snapshot(0),
        "release_thresholds": release_thresholds(peak_basis, properties.total_memory),
        "hashes": {
            name: sha256_file(path) for name, path in {
                "probe": Path(__file__).resolve(),
                "trainer": REPO / "tools/train_plan_a_gec_convergence_v1.py",
                "structured_margin": REPO / "src/rnajepa/structured_margin.py",
                "gec_numeric": REPO / "src/rnajepa/gec_numeric_v1.py",
                "gec_context": REPO / "src/rnajepa/gec_context.py",
                "config": Path(args.gec_config),
                "candidate_contract": REPO / config["candidate_contract"],
                "source_contract": Path(config["source_contract"]),
                "train_data": Path(config["assets"]["data"]),
                "train_ids": Path(config["assets"]["train_ids"]),
                "shared_init": Path(config["assets"]["shared_init"]),
                "gec_init": Path(init_args.gec_init),
            }.items()
        },
    }
    atomic_json(result, output_path)
    print(json.dumps({
        "status": result["status"], "memory": result["memory"],
        "timing": result["timing"], "release_thresholds": result["release_thresholds"],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
