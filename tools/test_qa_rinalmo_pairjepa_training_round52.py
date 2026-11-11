from qa_rinalmo_j1_collapse_round52 import sustained_collapse
from qa_rinalmo_pairjepa_training_round52 import ARM_CONFIG, obvious_collapse


assert ARM_CONFIG == {"j1_block_ema": ("ema", "structured_block"),
                      "j1_fixed": ("fixed", "structured_block"),
                      "j_random": ("random", "structured_block")}
row = {"target_variance": 1.0, "prediction_variance": 0.5,
       "target_effective_rank": 8.0, "prediction_effective_rank": 7.0,
       "target_channel_variance": [0.1] * 490,
       "prediction_channel_variance": [0.2] * 490}
assert not obvious_collapse(row)
for key in ("target_variance", "prediction_variance"):
    changed = dict(row)
    changed[key] = 0.0
    assert obvious_collapse(changed)
for key in ("target_effective_rank", "prediction_effective_rank"):
    changed = dict(row)
    changed[key] = 1.0
    assert obvious_collapse(changed)
changed = dict(row)
changed["target_channel_variance"] = [0.0] * 466 + [0.1] * 24
assert obvious_collapse(changed)
changed["target_channel_variance"] = [0.0] * 465 + [0.1] * 25
assert not obvious_collapse(changed)
collapsed = {prefix + "_effective_rank": 2.0 for prefix in ("target", "prediction")}
collapsed.update({prefix + "_" + scope + "_cosine": 0.999
                  for prefix in ("target", "prediction") for scope in ("cross_position", "cross_sequence")})
assert not sustained_collapse([collapsed, collapsed])
assert sustained_collapse([collapsed, collapsed, collapsed])
healthy = dict(collapsed, target_effective_rank=4.0)
assert not sustained_collapse([collapsed, healthy, collapsed, collapsed])
print("PASS target/context, constant and sustained high-correlation collapse failure injection")
