import numpy as np
import torch

from rnajepa.decision_head import FlatDecisionHead, HeadOnlyModel
from rnajepa.encoder import BASE_TO_INDEX
from tools.structrfm_decode_round48 import decode_scores
from tools.xens2_eval import r2d_scores


def test_greedy_reports_crossings_while_dp_is_legal():
    scores = np.full((6, 6), -np.inf)
    mask = np.zeros((6, 6), dtype=bool)
    for pair in ((0, 4), (1, 5)):
        scores[pair] = 1
        mask[pair] = True
    greedy, crossings, conflicts = decode_scores(scores, mask, "greedy")
    assert greedy == [(0, 4), (1, 5)]
    assert crossings == 1 and conflicts == 0
    dynamic, crossings, conflicts = decode_scores(scores, mask, "nussinov")
    assert len(dynamic) == 1 and crossings == 0 and conflicts == 0


def test_greedy_reports_shared_nucleotide_conflicts():
    scores = np.full((7, 7), -np.inf)
    mask = np.zeros((7, 7), dtype=bool)
    for pair in ((0, 5), (1, 5)):
        scores[pair] = 1
        mask[pair] = True
    greedy, crossings, conflicts = decode_scores(scores, mask, "greedy")
    assert greedy == [(0, 5), (1, 5)]
    assert crossings == 0 and conflicts == 1
    dynamic, crossings, conflicts = decode_scores(scores, mask, "nussinov")
    assert len(dynamic) == 1 and crossings == 0 and conflicts == 0


def test_turner_override_is_applied_before_calibration_without_changing_default():
    head = FlatDecisionHead(d_model=16, d_z=8, scorer="resnet2d", chunk_size=0).eval()
    h = torch.randn(1, 7, 16)
    ids = torch.tensor([[BASE_TO_INDEX[base] for base in "ACGUAAC"]])
    lengths = torch.tensor([7])
    with torch.no_grad():
        default = head(h, ids, lengths=lengths)
        explicit_default = head(h, ids, lengths=lengths, prior_weight_override=None)
        zero = head(h, ids, lengths=lengths, prior_weight_override=0.0)
        raw, _ = head.pair_scores_and_types(h, ids)
        expected = head.calibration(raw, lengths)
        expected = torch.where(default.mask, expected, torch.full_like(expected, float("-inf")))
    torch.testing.assert_close(default.scores, explicit_default.scores)
    torch.testing.assert_close(zero.scores, expected)
    assert head.prior_weight.item() == 1

    wrapped = HeadOnlyModel(head)
    embedding = h[0].numpy()
    wrapped_default = r2d_scores(wrapped, "ACGUAAC", embedding, "cpu")
    wrapped_zero = r2d_scores(wrapped, "ACGUAAC", embedding, "cpu",
                              prior_weight_override=0.0)
    torch.testing.assert_close(wrapped_default, default.scores[0])
    torch.testing.assert_close(wrapped_zero, expected[0])
