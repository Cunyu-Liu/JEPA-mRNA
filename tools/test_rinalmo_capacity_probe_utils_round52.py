import numpy as np
import pytest
import torch

from rnajepa.harness import valid_pair_mask
from rinalmo_capacity_probe_utils import (
    fit_linear_probe, predict_logits, sample_pair_indices, symmetric_pair_features)


def test_pair_sampling_is_legal_reproducible_and_symmetric_features_are_exact():
    sequence = "AUGCAUGC"
    gt = [(0, 5)]
    first, first_labels = sample_pair_indices(sequence, gt, 8, 42)
    second, second_labels = sample_pair_indices(sequence, gt, 8, 42)
    other, _ = sample_pair_indices(sequence, gt, 8, 2026)
    assert np.array_equal(first, second) and np.array_equal(first_labels, second_labels)
    assert not np.array_equal(first, other)
    valid = np.asarray(valid_pair_mask(sequence), dtype=bool)
    assert all(valid[i, j] for i, j in first)
    assert set(first_labels) == {0.0, 1.0}
    hidden = np.arange(24, dtype=np.float32).reshape(8, 3)
    pairs = np.asarray([[0, 5]])
    assert np.array_equal(symmetric_pair_features(hidden, pairs),
                          symmetric_pair_features(hidden, pairs[:, ::-1]))
    with pytest.raises(ValueError):
        sample_pair_indices(sequence, gt, 0, 42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_probe_uses_cuda_and_fixed_optimization_protocol():
    features = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32,
                            device="cuda").repeat(32, 1)
    labels = torch.tensor([0, 1, 1, 1], dtype=torch.float32, device="cuda").repeat(32)
    model, history = fit_linear_probe(features, labels, steps=100, learning_rate=0.05,
                                      log_every=50)
    assert history[-1]["weighted_nll"] < history[0]["weighted_nll"]
    assert all(parameter.is_cuda for parameter in model.parameters())
    logits = predict_logits(model, features)
    assert logits.is_cuda and torch.sigmoid(logits).mean() > 0.5
    with pytest.raises(RuntimeError):
        fit_linear_probe(features.cpu(), labels.cpu(), steps=1)
