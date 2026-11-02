#!/usr/bin/env python3
from verify_three_baselines_clean_round39 import score_pairs

rows = [
    {"name": "a", "pairs": [[0, 3], [1, 2]]},
    {"name": "b", "pairs": [[0, 2]]},
]
predictions = [
    {"name": "a", "pairs": [[0, 3], [0, 2], [0, 3]]},
    {"name": "b", "pairs": []},
]
tp, fp, fn, f1 = score_pairs(rows, predictions, "pairs")
assert (tp, fp, fn) == (1, 1, 2)
assert f1 == 2 / 5
try:
    score_pairs([{"name": "wrong", "pairs": []}], predictions[:1], "pairs")
except AssertionError:
    pass
else:
    raise AssertionError("row/prediction name mismatch was accepted")
try:
    score_pairs(rows, predictions[:1], "pairs")
except AssertionError:
    pass
else:
    raise AssertionError("row/prediction count mismatch was accepted")
print("THREE_BASELINE_CLEAN_SCORE_TEST_PASS")
