import copy

import torch
from torch import nn

from .context_mixer import LocalContextMixer
from .pair_diagnostics import masked_latent_correlations


@torch.no_grad()
def latent_effective_rank(features, max_samples=512):
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a nonempty [sample, latent] matrix")
    if features.shape[0] > max_samples:
        indices = torch.linspace(0, features.shape[0] - 1, max_samples,
                                 device=features.device).round().long()
        features = features[indices]
    centered = features.double() - features.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / features.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if not torch.isfinite(total) or total <= 0:
        return features.new_zeros((), dtype=torch.float32)
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


class LatentPairMLP(nn.Module):
    def __init__(self, d_z=128, latent_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_z)
        self.input_projection = nn.Linear(d_z, latent_dim)
        self.hidden = nn.Sequential(nn.GELU(), nn.Linear(latent_dim, latent_dim), nn.GELU())

    def forward(self, z, presymmetrized=False):
        if not presymmetrized:
            z = 0.5 * (z + z.transpose(1, 2))
        return self.hidden(self.input_projection(self.norm(z)))


class SymmetricPairMLP(LatentPairMLP):
    def __init__(self, d_z=128, latent_dim=64):
        super().__init__(d_z, latent_dim)
        self.logit = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def score(self, z, presymmetrized=False):
        return self.logit(self.forward(z, presymmetrized)).squeeze(-1)


def masked_neighbor_context(z, mask, valid_mask=None, kernel_size=3):
    if z.ndim != 4 or mask.shape != z.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("mask must be a bool [batch, length, length] tensor")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("context kernel must be odd and at least 3")
    if valid_mask is None:
        valid_mask = torch.ones_like(mask)
    if valid_mask.shape != mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have the same bool shape as mask")
    known = valid_mask & ~mask
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=z.dtype, device=z.device)
    kernel[0, 0, kernel_size // 2, kernel_size // 2] = 0
    padding = kernel_size // 2
    known_map = known.unsqueeze(1).to(z.dtype)
    numerator = torch.nn.functional.conv2d(
        (z * known.unsqueeze(-1)).permute(0, 3, 1, 2),
        kernel.expand(z.shape[-1], -1, -1, -1), padding=padding, groups=z.shape[-1])
    denominator = torch.nn.functional.conv2d(known_map, kernel, padding=padding)
    context = (numerator / denominator.clamp_min(1)).permute(0, 2, 3, 1)
    return context, denominator.squeeze(1)


def anti_collapse_terms(prediction, mask, std_floor=1.0):
    selected = prediction[mask]
    if selected.ndim != 2 or selected.shape[0] == 0:
        raise ValueError("masked prediction must be a nonempty [sample, latent] matrix")

    def terms(features):
        zero = features.sum() * 0.0
        if features.shape[0] < 2:
            return zero, zero
        variance = features.var(dim=0, unbiased=False)
        variance_loss = torch.nn.functional.relu(std_floor - (variance + 1e-4).sqrt()).mean()
        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / features.shape[0]
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        covariance_loss = (off_diagonal.square().sum()
                           / (features.shape[1] * (features.shape[1] - 1)))
        return variance_loss, covariance_loss

    position_variance, position_covariance = terms(selected)
    sequence_means = torch.stack([prediction[index, mask[index]].mean(dim=0)
                                  for index in range(prediction.shape[0])])
    sequence_variance, sequence_covariance = terms(sequence_means)
    return {"position_variance": position_variance, "position_covariance": position_covariance,
            "sequence_variance": sequence_variance, "sequence_covariance": sequence_covariance}


class PairJEPA(nn.Module):
    def __init__(self, d_z=128, latent_dim=64, use_ema=True, random_target=False, seed=0,
                 use_teacher=True, block_context=False):
        super().__init__()
        if use_ema and not use_teacher:
            raise ValueError("EMA requires a target encoder")
        self.scorer = SymmetricPairMLP(d_z, latent_dim)
        self.use_ema = use_ema
        self.random_target = random_target
        self.seed = seed
        self.teacher = copy.deepcopy(self.scorer) if use_teacher else None
        self.context_predictor = LatentPairMLP(d_z, latent_dim) if use_teacher else None
        self.context_mixer = LocalContextMixer(d_z) if block_context else None
        if self.teacher is not None:
            latent_state = {key: value for key, value in self.scorer.state_dict().items()
                            if not key.startswith("logit.")}
            self.context_predictor.load_state_dict(latent_state, strict=True)
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

    def masked_latent_loss(self, z, mask, rank_samples=0, valid_mask=None,
                         variance_weight=0.0, covariance_weight=0.0, std_floor=1.0):
        if self.teacher is None:
            raise ValueError("latent target requested without a target encoder")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be bool")
        mask = mask | mask.transpose(1, 2)
        if not mask.any():
            raise ValueError("empty masked-pair target")
        context, context_counts = masked_neighbor_context(z, mask, valid_mask)
        if self.context_mixer is not None:
            context = self.context_mixer(context)
        prediction = self.context_predictor(context)
        with torch.no_grad():
            target = self.teacher(z)
            if self.random_target:
                generator = torch.Generator(device=z.device).manual_seed(self.seed)
                shape = target.shape
                flat = target.reshape(-1, shape[-1])
                permutation = torch.randperm(flat.shape[0], generator=generator, device=flat.device)
                target = flat[permutation].reshape(shape)
        mse = torch.nn.functional.mse_loss(prediction[mask], target[mask])
        loss = mse
        anti_collapse = None
        if variance_weight or covariance_weight:
            anti_collapse = anti_collapse_terms(prediction, mask, std_floor)
            loss = loss + variance_weight * (anti_collapse["position_variance"]
                                             + anti_collapse["sequence_variance"])
            loss = loss + covariance_weight * (anti_collapse["position_covariance"]
                                               + anti_collapse["sequence_covariance"])
        diagnostics = {
            "target_variance": target[mask].var(unbiased=False).detach(),
            "prediction_variance": prediction[mask].var(unbiased=False).detach(),
            "target_mean_norm": target[mask].norm(dim=-1).mean().detach(),
            "prediction_mean_norm": prediction[mask].norm(dim=-1).mean().detach(),
            "contextless_pairs": (context_counts[mask] == 0).sum().detach(),
        }
        if rank_samples:
            diagnostics["target_effective_rank"] = latent_effective_rank(
                target[mask], max_samples=rank_samples)
            diagnostics["prediction_effective_rank"] = latent_effective_rank(
                prediction[mask].detach(), max_samples=rank_samples)
            diagnostics["teacher_student_mse"] = mse.detach()
            if anti_collapse is not None:
                diagnostics.update({name: value.detach() for name, value in anti_collapse.items()})
                diagnostics["jepa_mse"] = mse.detach()
                diagnostics["anti_collapse_loss"] = (loss - mse).detach()
            diagnostics["teacher_student_cosine"] = torch.nn.functional.cosine_similarity(
                prediction[mask].detach(), target[mask], dim=-1).mean().detach()
            for prefix, features in (("target", target[mask]),
                                     ("prediction", prediction[mask].detach())):
                diagnostics[f"{prefix}_channel_variance"] = features.var(
                    dim=0, unbiased=False).cpu().tolist()
                position, sequence = masked_latent_correlations(
                    target if prefix == "target" else prediction.detach(), mask,
                    max_samples=rank_samples)
                diagnostics[f"{prefix}_cross_position_cosine"] = position
                diagnostics[f"{prefix}_cross_sequence_cosine"] = sequence
        return loss, diagnostics

    def forward(self, z, presymmetrized=False):
        return self.scorer.score(z, presymmetrized)
