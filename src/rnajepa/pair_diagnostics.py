import torch


@torch.no_grad()
def off_diagonal_cosine(features, max_samples=512):
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a nonempty [sample, latent] matrix")
    if features.shape[0] < 2:
        return features.new_zeros((), dtype=torch.float32)
    if features.shape[0] > max_samples:
        indices = torch.linspace(0, features.shape[0] - 1, max_samples,
                                 device=features.device).round().long()
        features = features[indices]
    features = features.float()
    normalized = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = normalized @ normalized.T
    n = similarity.shape[0]
    return (similarity.sum() - similarity.diagonal().sum()) / (n * (n - 1))


@torch.no_grad()
def masked_latent_correlations(features, mask, max_samples=512):
    if features.ndim != 4 or mask.shape != features.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("features and mask must have matching [batch, length, length] shapes")
    position_values = []
    sequence_means = []
    for batch in range(features.shape[0]):
        selected = features[batch][mask[batch]]
        if selected.shape[0] == 0:
            continue
        position_values.append(off_diagonal_cosine(selected, max_samples=max_samples))
        sequence_means.append(selected.float().mean(dim=0))
    if not position_values:
        raise ValueError("at least one masked feature is required")
    position = torch.stack(position_values).mean()
    if len(sequence_means) > 1:
        sequence = off_diagonal_cosine(torch.stack(sequence_means), max_samples=max_samples)
    else:
        sequence = features.new_zeros((), dtype=torch.float32)
    return position, sequence
