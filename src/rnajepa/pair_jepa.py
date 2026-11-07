import copy

import torch
from torch import nn


@torch.no_grad()
def latent_effective_rank(features, max_samples=512):
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a nonempty [sample, latent] matrix")
    if features.shape[0] > max_samples:
        indices = torch.linspace(0, features.shape[0] - 1, max_samples,
                                 device=features.device).round().long()
        features = features[indices]
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / features.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if not torch.isfinite(total) or total <= 0:
        return features.new_zeros((), dtype=torch.float32)
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


class SymmetricPairMLP(nn.Module):
    def __init__(self, d_z=128, latent_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_z)
        self.input_projection = nn.Linear(d_z, latent_dim)
        self.hidden = nn.Sequential(nn.GELU(), nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.logit = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(self, z):
        z = 0.5 * (z + z.transpose(1, 2))
        return self.hidden(self.input_projection(self.norm(z)))

    def score(self, z):
        return self.logit(self.forward(z)).squeeze(-1)


class PairJEPA(nn.Module):
    def __init__(self, d_z=128, latent_dim=64, use_ema=True, random_target=False, seed=0,
                 use_teacher=True):
        super().__init__()
        if use_ema and not use_teacher:
            raise ValueError("EMA requires a target encoder")
        self.scorer = SymmetricPairMLP(d_z, latent_dim)
        self.use_ema = use_ema
        self.random_target = random_target
        self.seed = seed
        self.teacher = copy.deepcopy(self.scorer) if use_teacher else None
        if self.teacher is not None:
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
            self.teacher.eval()

    @torch.no_grad()
    def ema_update(self, momentum=0.999):
        if not self.use_ema:
            return
        for target, source in zip(self.teacher.parameters(), self.scorer.parameters()):
            target.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)
        for target, source in zip(self.teacher.buffers(), self.scorer.buffers()):
            target.copy_(source)

    def masked_latent_loss(self, z, mask, rank_samples=0):
        if self.teacher is None:
            raise ValueError("latent target requested without a target encoder")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be bool")
        prediction = self.scorer(z)
        with torch.no_grad():
            target = self.teacher(z)
            if self.random_target:
                generator = torch.Generator(device=z.device).manual_seed(self.seed)
                shape = target.shape
                flat = target.reshape(-1, shape[-1])
                permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
                target = flat[permutation].reshape(shape)
        if not mask.any():
            raise ValueError("empty masked-pair target")
        loss = torch.nn.functional.mse_loss(prediction[mask], target[mask])
        diagnostics = {
            "target_variance": target[mask].var(unbiased=False).detach(),
            "prediction_variance": prediction[mask].var(unbiased=False).detach(),
            "target_mean_norm": target[mask].norm(dim=-1).mean().detach(),
            "prediction_mean_norm": prediction[mask].norm(dim=-1).mean().detach(),
        }
        if rank_samples:
            diagnostics["target_effective_rank"] = latent_effective_rank(
                target[mask], max_samples=rank_samples)
            diagnostics["prediction_effective_rank"] = latent_effective_rank(
                prediction[mask].detach(), max_samples=rank_samples)
        return loss, diagnostics

    def forward(self, z):
        return self.scorer.score(z)
