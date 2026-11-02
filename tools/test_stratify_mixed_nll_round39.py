#!/usr/bin/env python3
from stratify_mixed_nll_round39 import BUCKETS, bucket_for_length, stratify

assert [bucket_for_length(length) for length in [1, 100, 101, 200, 201, 400, 401]] == [
    "1-100", "1-100", "101-200", "101-200", "201-400", "201-400", "401+"]
rows = [
    {"name": "short", "seq": "A" * 100, "pairs": [[0, 3], [1, 2]]},
    {"name": "long", "seq": "A" * 401, "pairs": [[0, 4]]},
]
predictions = [
    {"name": "short", "pairs": [[0, 3], [0, 2]]},
    {"name": "long", "pairs": []},
]
result = stratify(rows, predictions)
assert (result["1-100"]["tp"], result["1-100"]["fp"], result["1-100"]["fn"]) == (1, 1, 1)
assert result["1-100"]["f1"] == 0.5
assert result["1-100"]["pred_pairs_per_nt"] == 0.02
assert result["1-100"]["gt_pairs_per_nt"] == 0.02
assert (result["401+"]["tp"], result["401+"]["fp"], result["401+"]["fn"]) == (0, 0, 1)
assert result["401+"]["f1"] == 0
assert result["401+"]["pred_pairs_per_nt"] == 0
assert result["101-200"]["n"] == 0 and result["101-200"]["pred_pairs_per_nt"] is None
print("MIXED_NLL_STRATIFICATION_TEST_PASS")
