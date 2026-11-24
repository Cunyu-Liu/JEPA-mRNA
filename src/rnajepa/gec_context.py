"""Trainable second-pass residual and convergence guards for GEC v1."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch import nn

from rnajepa.gec_numeric_v1 import GEC_ARMS_V1, GEC_FEATURE_NAMES_V1

GEC_GEOMETRY_NAMES_V1 = (
    "log_span",
    "reciprocal_span",
    "short_bucket",
    "medium_bucket",
    "long_bucket",
    "very_long_bucket",
    "hairpin_boundary_distance",
    "nearest_interval_bucket_boundary_distance",
)
GEC_INPUT_NAMES_V1 = ("pair_representation",) + GEC_FEATURE_NAMES_V1 + GEC_GEOMETRY_NAMES_V1
GEC_RESIDUAL_FORMAT_V1 = "gec_shared_residual_init_v1"


def require_explicit_cuda(device: str | torch.device = "cuda:0") -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"GEC requires explicit CUDA; requested {resolved}")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    resolved = torch.device("cuda", index)
    torch.cuda.set_device(resolved)
    probe = torch.tensor([1.25, -2.0, 3.5], dtype=torch.float64, device=resolved)
    result = probe * 2.0 - 1.0
    torch.cuda.synchronize(resolved)
    if result.device != resolved or not torch.isfinite(result).all():
        raise RuntimeError("explicit CUDA arithmetic failed; CPU fallback is forbidden")
    return resolved


def gec_destroy_key_v1(seed: int, epoch: int, step: int, sequence: str) -> str:
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    return f"gec_destroy_v1|{seed}|{epoch}|{step}|{sequence_sha256}"


def gec_geometry_v1(legal_mask: torch.Tensor, min_loop: int = 3) -> torch.Tensor:
    if legal_mask.ndim != 2 or legal_mask.shape[0] != legal_mask.shape[1]:
        raise ValueError("legal_mask must be square")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must be bool")
    length = legal_mask.shape[0]
    positions = torch.arange(length, device=legal_mask.device)
    spans = (positions[None, :] - positions[:, None]).abs()
    upper = torch.triu(legal_mask, diagonal=1)
    spans64 = spans.to(torch.float64)
    boundaries = torch.tensor([4, 11, 31, 101], device=legal_mask.device)
    nearest = boundaries[(spans[..., None] - boundaries).abs().argmin(dim=-1)]
    features = torch.stack(
        (
            torch.log1p(spans64),
            spans64.reciprocal().nan_to_num(0.0),
            ((spans >= 4) & (spans <= 10)).to(torch.float64),
            ((spans >= 11) & (spans <= 30)).to(torch.float64),
            ((spans >= 31) & (spans <= 100)).to(torch.float64),
            (spans >= 101).to(torch.float64),
            spans64 - float(min_loop + 1),
            spans64 - nearest.to(torch.float64),
        ),
        dim=-1,
    )
    features = torch.where(upper.unsqueeze(-1), features, torch.zeros_like(features))
    return (features + features.transpose(0, 1)).detach()


class GECResidualV1(nn.Module):
    def __init__(self, d_z: int = 128, hidden: int = 64) -> None:
        super().__init__()
        if d_z <= 0 or hidden <= 0:
            raise ValueError("GEC dimensions must be positive")
        self.d_z = int(d_z)
        self.hidden = int(hidden)
        self.context_channels = len(GEC_FEATURE_NAMES_V1)
        self.geometry_channels = len(GEC_GEOMETRY_NAMES_V1)
        self.net = nn.Sequential(
            nn.Linear(d_z + self.context_channels + self.geometry_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        pair_representation: torch.Tensor,
        context: torch.Tensor,
        geometry: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pair_representation.ndim != 3 or pair_representation.shape[-1] != self.d_z:
            raise ValueError("pair_representation must be L x L x d_z")
        length = pair_representation.shape[0]
        expected_context = (length, length, self.context_channels)
        expected_geometry = (length, length, self.geometry_channels)
        if context.shape != expected_context or geometry.shape != expected_geometry:
            raise ValueError("context or geometry shape mismatch")
        if legal_mask.shape != (length, length) or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask must be an L x L bool matrix")
        if context.requires_grad or geometry.requires_grad:
            raise ValueError("GEC context and geometry must be stop-gradient tensors")
        dtype = pair_representation.dtype
        inputs = torch.cat((pair_representation, context.to(dtype), geometry.to(dtype)), dim=-1)
        delta = self.net(inputs).squeeze(-1)
        delta = 0.5 * (delta + delta.transpose(0, 1))
        return torch.where(legal_mask, delta, torch.zeros_like(delta))


def terminal_unfreeze_step_v1(
    warmup_head_steps: int = 1600,
    unfreeze_every: int = 800,
    unfreeze_per_step: int = 2,
    n_blocks: int = 33,
    max_unfrozen_blocks: int = 20,
) -> int:
    if not 0 <= max_unfrozen_blocks <= n_blocks:
        raise ValueError("invalid terminal unfreeze cap")
    if min(warmup_head_steps, unfreeze_per_step, max_unfrozen_blocks) < 0 or unfreeze_every <= 0:
        raise ValueError("invalid unfreeze schedule")
    if max_unfrozen_blocks == 0:
        return warmup_head_steps
    events = math.ceil(max_unfrozen_blocks / unfreeze_per_step)
    return warmup_head_steps + events * unfreeze_every


@dataclass
class GECConvergenceStateV1:
    terminal_step: int = 9600
    terminal_unfrozen_blocks: int = 20
    patience: int = 5
    min_delta: float = 1e-6
    best_step: Optional[int] = None
    best_validation_loss: float = float("inf")
    bad_evaluations: int = 0
    eligible: bool = False

    def observe(self, step: int, loss: float, n_unfrozen: int) -> dict[str, Any]:
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite convergence-development NLL")
        if n_unfrozen > self.terminal_unfrozen_blocks:
            raise ValueError("checkpoint exceeds the terminal 20/33 unfreeze cap")
        if n_unfrozen < self.terminal_unfrozen_blocks or step < self.terminal_step:
            return {"eligible": False, "improved": False, "save_best": False, "stop": False}
        if n_unfrozen != self.terminal_unfrozen_blocks:
            raise ValueError("eligible checkpoints must have exactly 20/33 unfrozen blocks")
        if not self.eligible:
            self.eligible = True
            self.best_step = None
            self.best_validation_loss = float("inf")
            self.bad_evaluations = 0
        improved = loss < self.best_validation_loss - self.min_delta
        if improved:
            self.best_step = int(step)
            self.best_validation_loss = float(loss)
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return {
            "eligible": True,
            "improved": improved,
            "save_best": improved,
            "stop": self.bad_evaluations >= self.patience,
        }

    def state_dict(self) -> dict[str, Any]:
        state = asdict(self)
        if not math.isfinite(self.best_validation_loss):
            state["best_validation_loss"] = None
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("terminal_step", "terminal_unfrozen_blocks", "patience", "min_delta"):
            if getattr(self, name) != state[name]:
                raise ValueError(f"resume convergence identity mismatch: {name}")
        self.best_step = state["best_step"]
        self.best_validation_loss = (
            float("inf") if state["best_validation_loss"] is None
            else float(state["best_validation_loss"])
        )
        self.bad_evaluations = int(state["bad_evaluations"])
        self.eligible = bool(state["eligible"])


def validate_gec_arm_v1(arm: str) -> str:
    if arm not in GEC_ARMS_V1:
        raise ValueError(f"unknown GEC arm: {arm}")
    return arm
