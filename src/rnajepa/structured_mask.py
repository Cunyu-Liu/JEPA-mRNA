import torch


def structured_pair_mask(valid, probability, seed, block_size=8, random_pair_probability=0.01):
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[-1] != valid.shape[-2]:
        raise ValueError("valid must be a bool [batch, length, length] tensor")
    if not 0 < probability <= 1 or not 0 <= random_pair_probability <= 1:
        raise ValueError("mask probabilities must be in (0, 1] and [0, 1]")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if not valid.any():
        raise ValueError("no legal pairs available for structured Pair-JEPA target")
    length = valid.shape[-1]
    n_blocks = (length + block_size - 1) // block_size
    generator = torch.Generator(device=valid.device).manual_seed(seed)
    block_scores = torch.rand((valid.shape[0], n_blocks, n_blocks),
                              device=valid.device, generator=generator)
    coordinates = torch.arange(length, device=valid.device) // block_size
    anti_diagonal = (coordinates[:, None] + coordinates[None, :]) == n_blocks - 1
    selected_blocks = block_scores < probability
    block_mask = selected_blocks[:, coordinates[:, None], coordinates[None, :]]
    structured = block_mask & anti_diagonal.unsqueeze(0) & valid
    random_scores = torch.rand(valid.shape, device=valid.device, generator=generator)
    random_pairs = (random_scores < random_pair_probability) & valid
    mask = structured | random_pairs
    if not mask.any():
        indices = valid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        selected = indices[torch.randint(indices.numel(), (1,), device=valid.device,
                                        generator=generator)]
        mask.reshape(-1)[selected] = True
    return mask | mask.transpose(1, 2)
