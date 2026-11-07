import pytest
import torch

from tools.ripe_structrfm import enable_ripe_structrfm, make_tiny_bert


def test_ripe_bert_extends_beyond_absolute_position_limit_and_backprops():
    model = make_tiny_bert(max_position_embeddings=8)
    ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
    with pytest.raises((IndexError, RuntimeError)):
        model(input_ids=ids)
    model = enable_ripe_structrfm(model, "r3", vocab={"A": 1, "C": 2, "G": 3, "U": 4, "T": 4})
    output = model(input_ids=ids)
    assert output.last_hidden_state.shape == (1, 12, 12)
    output.last_hidden_state.square().mean().backward()
    assert model.encoder.layer[0].attention.self.ripe.nu_raw.grad is not None
    assert model.encoder.layer[0].attention.self.ripe.kappa_raw.grad is not None


def test_ripe_bert_eval_is_deterministic():
    model = enable_ripe_structrfm(make_tiny_bert(max_position_embeddings=8), "r3",
                                  vocab={"A": 1, "C": 2, "G": 3, "U": 4, "T": 4}).eval()
    ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
    with torch.no_grad():
        first = model(input_ids=ids).last_hidden_state
        second = model(input_ids=ids).last_hidden_state
    torch.testing.assert_close(first, second)
