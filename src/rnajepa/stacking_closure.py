"""Parameter-matched stacking-closure and negative-control pair scorers."""
from __future__ import annotations

import torch
from torch import nn

MODES = ("local_control", "stacking_closure", "destroy_offset", "destroy_channel_shuffle")


def shift_pair(z, di, dj):
    """Shift a (B,L,L,C) pair tensor with zero boundary padding."""
    B, L, L2, C = z.shape
    if L != L2:
        raise ValueError("pair tensor must be square")
    out = torch.zeros_like(z)
    dst_i = slice(max(di, 0), min(L + di, L))
    dst_j = slice(max(dj, 0), min(L + dj, L))
    src_i = slice(max(-di, 0), min(L - di, L))
    src_j = slice(max(-dj, 0), min(L - dj, L))
    out[:, dst_i, dst_j] = z[:, src_i, src_j]
    return out


class Closure2DScorer(nn.Module):
    """One residual pair-representation cell with explicit neighbor topology."""

    def __init__(self, d_z, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown closure mode: {mode}")
        self.mode = mode
        self.d_z = int(d_z)
        self.per_block_checkpoint = False
        self.input = nn.Linear(5 * d_z, d_z)
        self.update = nn.Linear(d_z, d_z)
        self.head = nn.Linear(d_z, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        generator = torch.Generator().manual_seed(20261124)
        self.register_buffer("channel_permutation", torch.randperm(d_z, generator=generator))

    def neighbors(self, z):
        if self.mode == "stacking_closure" or self.mode == "destroy_channel_shuffle":
            first, second = shift_pair(z, 1, -1), shift_pair(z, -1, 1)
        elif self.mode == "local_control":
            first, second = shift_pair(z, 1, 0), shift_pair(z, -1, 0)
        else:
            first, second = shift_pair(z, 1, 1), shift_pair(z, -1, -1)
        if self.mode == "destroy_channel_shuffle":
            first = first[..., self.channel_permutation]
            second = second[..., self.channel_permutation]
        return first, second

    def forward(self, z):
        if z.ndim != 4 or z.shape[1] != z.shape[2] or z.shape[-1] != self.d_z:
            raise ValueError("expected (B,L,L,d_z)")
        first, second = self.neighbors(z)
        features = torch.cat((z, first, second, z * first, z * second), dim=-1)
        update = self.update(torch.nn.functional.gelu(self.input(features)))
        updated = 0.5 * ((z + update) + (z + update).transpose(1, 2))
        scores = self.head(torch.nn.functional.gelu(updated)).squeeze(-1)
        scores = 0.5 * (scores + scores.transpose(1, 2))
        length = z.shape[1]
        positions = torch.arange(length, device=z.device)
        valid = (positions[:, None] - positions[None, :]).abs() > 3
        return torch.where(valid.unsqueeze(0), scores, torch.zeros_like(scores))
