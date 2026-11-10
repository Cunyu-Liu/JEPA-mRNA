import torch

from rnajepa.decision_head import FlatDecisionHead


def reference(head, h, ids):
    z = head.pair_repr(h)
    scores = head.score_pair_matrix(z, ids)
    types = head.type_head.symmetrize(head.type_head.cross(h, h))
    return scores, types


def test_pair_scorer_chunks_match_whole_matrix_forward_and_backward():
    torch.manual_seed(42)
    head = FlatDecisionHead(d_model=16, d_z=8, hidden=64, scorer="pairmlp",
                            pair_latent_dim=16, chunk_size=0, ripe_mode="r6")
    h = torch.randn(1, 137, 16, requires_grad=True)
    ids = torch.randint(0, 4, (1, 137))
    expected_scores, expected_types = reference(head, h, ids)
    expected_scores.sum().backward()
    expected_gradient = h.grad.detach().clone()
    h.grad = None
    scores, types = head.pair_scores_and_types(h, ids)
    torch.testing.assert_close(scores, expected_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(types, expected_types, rtol=0, atol=0)
    scores.sum().backward()
    torch.testing.assert_close(h.grad, expected_gradient, rtol=1e-5, atol=1e-5)
    print("max_score_error", (scores - expected_scores).abs().max().item())
    print("max_gradient_error", (h.grad - expected_gradient).abs().max().item())
    lengths = torch.tensor([137])
    without_types = head(h, ids, lengths=lengths, include_types=False)
    with_types = head(h, ids, lengths=lengths)
    assert without_types.pair_types is None
    torch.testing.assert_close(without_types.scores, with_types.scores, rtol=0, atol=0)


if __name__ == "__main__":
    test_pair_scorer_chunks_match_whole_matrix_forward_and_backward()
    print("PASS pair-scorer chunk forward and backward equivalence")
