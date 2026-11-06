from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RipeConfig:
    n_heads: int = 12
    mode: str = "r3"
    random_period_seed: int = 0
    random_complement_seed: int = 0


class RipePositionalBias(nn.Module):
    MODES = {"log", "r1", "r2", "r3", "r5", "r6"}

    def __init__(self, config: RipeConfig):
        super().__init__()
        if config.mode not in self.MODES:
            raise ValueError(config.mode)
        self.config = config
        self.n_heads = config.n_heads
        self.distance_raw = nn.Parameter(torch.zeros(self.n_heads, 8))
        self.helix_logits = nn.Parameter(torch.zeros(self.n_heads, 2))
        self.helix_phase = nn.Parameter(torch.zeros(self.n_heads, 2))
        self.nu_raw = nn.Parameter(torch.full((self.n_heads,), 0.5))
        self.kappa_raw = nn.Parameter(torch.full((self.n_heads,), 0.1))
        centers = torch.exp(torch.linspace(torch.log(torch.tensor(4.0)),
                                           torch.log(torch.tensor(128.0)), 8))
        focal = torch.exp(torch.linspace(torch.log(torch.tensor(8.0)),
                                         torch.log(torch.tensor(256.0)), self.n_heads))
        self.register_buffer("centers", centers)
        self.register_buffer("focal", focal)
        self.register_buffer("periods", torch.tensor([11.0, 22.0]))
        complement = torch.zeros(8, 8)
        for first, second in ((0, 3), (3, 0), (1, 2), (2, 1)):
            complement[first, second] = 1.0
        for first, second in ((2, 3), (3, 2)):
            complement[first, second] = 0.5
        self.register_buffer("complement", complement)
        if config.mode == "r5":
            generator = torch.Generator().manual_seed(config.random_period_seed)
            self.register_buffer("random_periods", 5.0 + 25.0 * torch.rand(2, generator=generator))
        if config.mode == "r6":
            generator = torch.Generator().manual_seed(config.random_complement_seed)
            permutation = torch.randperm(4, generator=generator)
            while permutation.equal(torch.tensor([1, 0, 2, 3])):
                permutation = torch.randperm(4, generator=generator)
            random_complement = complement.clone()
            random_complement[:4, :4] = complement[permutation][:, permutation]
            self.register_buffer("random_complement", random_complement)

    def _distance_basis(self, delta):
        value = torch.log(delta.clamp_min(4.0))
        left = torch.searchsorted(self.centers, value, right=True) - 1
        left = left.clamp(0, len(self.centers) - 2)
        right = left + 1
        center_left = self.centers[left]
        center_right = self.centers[right]
        fraction = (value - center_left.log()) / (center_right.log() - center_left.log())
        basis = torch.zeros(*delta.shape, 8, dtype=torch.float32, device=delta.device)
        basis.scatter_(-1, left.unsqueeze(-1), 1.0 - fraction.unsqueeze(-1))
        basis.scatter_(-1, right.unsqueeze(-1), fraction.unsqueeze(-1))
        basis = torch.where((delta <= 4.0).unsqueeze(-1),
                            torch.nn.functional.one_hot(torch.zeros_like(delta, dtype=torch.long),
                                                        8).float(), basis)
        basis = torch.where((delta >= 128.0).unsqueeze(-1),
                            torch.nn.functional.one_hot(torch.full_like(delta, 7, dtype=torch.long),
                                                        8).float(), basis)
        basis = torch.where((delta == 0).unsqueeze(-1), torch.zeros_like(basis), basis)
        return basis

    def forward(self, seq_ids):
        if seq_ids.ndim != 2:
            raise ValueError("seq_ids must be [batch, length]")
        length = seq_ids.shape[1]
        delta = torch.arange(length, device=seq_ids.device)
        delta = (delta[:, None] - delta[None, :]).abs().float()
        bias = torch.zeros(self.n_heads, length, length,
                           dtype=torch.float32, device=seq_ids.device)
        if self.config.mode != "log":
            weights = torch.tanh(self.distance_raw)
            distance = torch.einsum("ijk,hk->hij", self._distance_basis(delta), weights)
            reference = weights[:, 0].reshape(self.n_heads, 1, 1)
            distance = torch.where(delta.unsqueeze(0) > 0, distance - reference,
                                   torch.zeros_like(distance))
            bias = bias + distance
        far = torch.nn.functional.softplus(self.nu_raw) + 1.0001
        far = far.reshape(self.n_heads, 1, 1) * torch.log1p(
            delta.unsqueeze(0) / self.focal.reshape(self.n_heads, 1, 1))
        bias = bias - far
        if self.config.mode in {"r2", "r3", "r5", "r6"}:
            periods = self.random_periods if self.config.mode == "r5" else self.periods
            weights = torch.softmax(self.helix_logits, dim=-1)
            phase = delta.unsqueeze(0).unsqueeze(-1) * (2 * torch.pi / periods)
            phase = phase + self.helix_phase.reshape(self.n_heads, 1, 1, 2)
            helix = -(torch.cos(phase) * weights.reshape(self.n_heads, 1, 1, 2)).sum(dim=-1)
            bias = bias + torch.where(delta.unsqueeze(0) > 0, helix, torch.zeros_like(helix))
        if self.config.mode in {"r3", "r6"}:
            table = self.random_complement if self.config.mode == "r6" else self.complement
            valid = (seq_ids >= 0) & (seq_ids < 4)
            content = table[seq_ids.unsqueeze(2), seq_ids.unsqueeze(1)]
            content = torch.where(valid.unsqueeze(2) & valid.unsqueeze(1),
                                  content, torch.zeros_like(content))
            content = content * (delta.unsqueeze(0) >= 4.0).float()
            kappa = 2.0 * torch.sigmoid(self.kappa_raw).reshape(self.n_heads, 1, 1)
            bias = bias.unsqueeze(0) + kappa.unsqueeze(0) * content.unsqueeze(1)
        else:
            bias = bias.unsqueeze(0)
        return bias


def nucleotide_ids(sequence, device=None):
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    return torch.tensor([mapping.get(char, 7) for char in sequence], device=device)
