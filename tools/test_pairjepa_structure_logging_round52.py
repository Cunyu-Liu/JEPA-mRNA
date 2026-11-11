import torch

from train_plan_a import training_batch_structure_metrics


assert torch.cuda.is_available(), "real CUDA is required"
matrix = torch.full((1, 6, 6), -float("inf"), device="cuda", requires_grad=True)
with torch.no_grad():
    matrix[0, 0, 5] = 10.0
    matrix[0, 5, 0] = 10.0
no_prior = matrix.detach().clone()
before = matrix.detach().clone()
rng_before = torch.cuda.get_rng_state()
result = training_batch_structure_metrics(matrix, no_prior, [6], ["GAAAAC"], [[(0, 5)]])
assert torch.equal(matrix.detach(), before)
assert matrix.grad is None
assert torch.equal(torch.cuda.get_rng_state(), rng_before)
for name in ("nussinov", "greedy", "greedy_turner0"):
    assert result[name] == {"tp": 1, "fp": 0, "fn": 0, "micro_f1": 1.0, "macro_f1": 1.0}
empty = training_batch_structure_metrics(matrix, no_prior, [6], ["GAAAAC"], [[]])
for name in ("nussinov", "greedy", "greedy_turner0"):
    assert empty[name]["tp"] == 0 and empty[name]["fp"] == 1 and empty[name]["fn"] == 0
    assert empty[name]["micro_f1"] == empty[name]["macro_f1"] == 0.0
print("PASS CUDA DP/greedy/Turner=0 diagnostics, exact counts, no gradient/RNG/model mutation")
