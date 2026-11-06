import pytest
import torch

import train_plan_a
import xens2_eval
from rnajepa.decision_head import FlatDecisionHead
from structrfm_round48 import install_structrfm_training


WEIGHTS = "/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/structrfm_round48"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_structrfm_validation_turner_prior_stays_on_cuda():
    install_structrfm_training(train_plan_a, xens2_eval, WEIGHTS)
    encoder, _ = train_plan_a.load_encoder("cuda", torch.float32)
    encoder.eval()
    head = FlatDecisionHead(d_model=768, d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0).cuda().eval()
    with torch.no_grad():
        scores = xens2_eval.plana_scores(encoder, head, "AUGCAUGC", "cuda")
    assert scores.shape == (8, 8)
    assert scores.is_cuda
    assert torch.isfinite(scores).any()
