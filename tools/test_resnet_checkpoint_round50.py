import torch

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.train_decision import (_resnet_batch_norm_state,
                                    _restore_resnet_batch_norm_state)


def test_checkpoint_preserves_scores_gradients_and_single_batch_norm_update():
    torch.manual_seed(17)
    plain = FlatDecisionHead(d_model=8, d_z=4, hidden=4, scorer="resnet2d",
                             resnet_blocks=2, chunk_size=0, grad_checkpoint=False)
    optimized = FlatDecisionHead(d_model=8, d_z=4, hidden=4, scorer="resnet2d",
                                 resnet_blocks=2, chunk_size=0, grad_checkpoint=True)
    optimized.load_state_dict(plain.state_dict())
    plain.train()
    optimized.train()
    z_plain = torch.randn(1, 7, 7, 4, requires_grad=True)
    z_optimized = z_plain.detach().clone().requires_grad_(True)
    plain_scores = plain.score_pair_matrix(z_plain)
    optimized_scores = optimized.score_pair_matrix(z_optimized)
    torch.testing.assert_close(optimized_scores, plain_scores, rtol=0, atol=0)
    plain_state = _resnet_batch_norm_state(plain)
    optimized_state = _resnet_batch_norm_state(optimized)
    plain_scores.square().mean().backward()
    optimized_scores.square().mean().backward()
    _restore_resnet_batch_norm_state(plain_state)
    _restore_resnet_batch_norm_state(optimized_state)
    for plain_bn, optimized_bn in zip(plain.resnet2d.modules(),
                                      optimized.resnet2d.modules()):
        if isinstance(plain_bn, torch.nn.BatchNorm2d):
            assert torch.equal(plain_bn.running_mean, optimized_bn.running_mean)
            assert torch.equal(plain_bn.running_var, optimized_bn.running_var)
            assert plain_bn.num_batches_tracked == optimized_bn.num_batches_tracked == 1
    torch.testing.assert_close(z_optimized.grad, z_plain.grad, rtol=0, atol=0)
