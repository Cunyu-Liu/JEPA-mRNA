import numpy as np
import torch

from rnajepa.harness import valid_pair_mask


def sample_pair_indices(sequence, gt_pairs, n_samples, seed):
    if n_samples <= 0 or seed not in (42, 2026, 3407):
        raise ValueError("invalid pair sample size or seed")
    valid = np.asarray(valid_pair_mask(sequence), dtype=bool)
    gt = {(int(i), int(j)) for i, j in gt_pairs if valid[int(i), int(j)]}
    positive = np.asarray(sorted(gt), dtype=np.int64).reshape(-1, 2)
    negative = np.argwhere(valid)
    if len(negative):
        keep = np.asarray([tuple(pair) not in gt for pair in negative])
        negative = negative[keep]
    if not len(positive) or not len(negative):
        raise ValueError("both positive and negative legal pairs are required")
    rng = np.random.default_rng(seed)
    n_positive = min(len(positive), n_samples // 2)
    n_negative = min(len(negative), n_samples - n_positive)
    if n_positive + n_negative < n_samples:
        n_positive = min(len(positive), n_samples - n_negative)
    if n_positive + n_negative < n_samples:
        n_negative = min(len(negative), n_samples - n_positive)
    positive = positive[rng.choice(len(positive), n_positive, replace=False)]
    negative = negative[rng.choice(len(negative), n_negative, replace=False)]
    pairs = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate([np.ones(n_positive), np.zeros(n_negative)]).astype(np.float32)
    order = rng.permutation(len(pairs))
    return pairs[order], labels[order]


def symmetric_pair_features(hidden, pairs):
    left = hidden[pairs[:, 0]]
    right = hidden[pairs[:, 1]]
    return np.concatenate([left + right, np.abs(left - right)], axis=1).astype(np.float32)


def fit_linear_probe(features, labels, steps=200, learning_rate=0.01,
                     weight_decay=1e-4, seed=42, batch_size=1024, log_every=50):
    if not torch.cuda.is_available() or features.device.type != "cuda" or labels.device.type != "cuda":
        raise RuntimeError("frozen-representation probe fitting requires CUDA")
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("expected [sample, feature] and [sample] tensors")
    if steps <= 0 or batch_size <= 0 or seed not in (42, 2026, 3407):
        raise ValueError("invalid probe optimizer configuration")
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], 1).to(features.device)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("both classes are required")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        order = torch.randperm(len(features), device=features.device, generator=generator)
        total = 0.0
        for start in range(0, len(features), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features[indices]).squeeze(1), labels[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        if step == 1 or step == steps or step % log_every == 0:
            history.append({"step": step, "weighted_nll": total / len(features)})
    return model, history


def predict_logits(model, features, batch_size=4096):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs.append(model(features[start:start + batch_size]).squeeze(1))
    return torch.cat(outputs)
