import pytest
import torch

from tools.ripe_structrfm import enable_ripe_structrfm, make_tiny_bert
from rnajepa.ripe import RipeConfig, RipePositionalBias


def test_ripe_bert_extends_beyond_absolute_position_limit_and_backprops():
    model = make_tiny_bert(max_position_embeddings=8)
    ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
    with pytest.raises((IndexError, RuntimeError)):
        model(input_ids=ids)
    model = enable_ripe_structrfm(model, "r3", vocab={"A": 0, "C": 1, "G": 2, "U": 3, "T": 3})
    output = model(input_ids=ids)
    assert output.last_hidden_state.shape == (1, 12, 12)
    output.last_hidden_state.square().mean().backward()
    assert model.encoder.layer[0].attention.self.ripe.nu_raw.grad is not None
    assert model.encoder.layer[0].attention.self.ripe.kappa_raw.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ripe_bert_added_parameters_follow_encoder_device():
    model = make_tiny_bert().cuda()
    model = enable_ripe_structrfm(model, "r3", vocab={"A": 0, "C": 1, "G": 2, "U": 3, "T": 3})
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    assert model(input_ids=ids).last_hidden_state.shape == (1, 4, 12)
    for layer in model.encoder.layer:
        assert all(parameter.is_cuda for parameter in layer.attention.self.ripe.parameters())
        assert layer.attention.self.ripe.centers.is_cuda


def test_ripe_bert_eval_is_deterministic():
    model = enable_ripe_structrfm(make_tiny_bert(max_position_embeddings=8), "r3",
                                  vocab={"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}).eval()
    ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
    with torch.no_grad():
        first = model(input_ids=ids).last_hidden_state
        second = model(input_ids=ids).last_hidden_state
    torch.testing.assert_close(first, second)


def test_structrfm_official_ids_map_to_canonical_complement_pairs():
    model = enable_ripe_structrfm(make_tiny_bert(), "r3",
                                  vocab={"A": 5, "C": 7, "G": 8, "U": 6, "T": 0})
    ripe = model.encoder.layer[0].attention.self.ripe
    assert ripe.token_id_map.tolist() == [3, 7, 7, 7, 7, 0, 3, 1, 2, 7, 7, 7, 7, 7, 7, 7]
    official_ids = torch.tensor([[5, 7, 8, 6, 5, 7, 8, 6]])
    canonical_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    canonical = RipePositionalBias(RipeConfig(n_heads=model.config.num_attention_heads, mode="r3"))
    torch.testing.assert_close(ripe(official_ids), canonical(canonical_ids), rtol=0, atol=0)
    with pytest.raises(ValueError, match="vocab"):
        enable_ripe_structrfm(make_tiny_bert(), "r3")
