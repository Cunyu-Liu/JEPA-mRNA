import math

from qa_rinalmo_capacity_replication_round52 import compare_error, compare_representation


def error(value):
    row = {"sequence_sha256": "a" * 64, "length": 10, "losses": {"total": value, "nll": value}}
    return {"arm": "length_s42", "training_seed": 42, "best_step": 1, "selection_seed": 42,
            "checkpoint_sha256": "c", "run_meta_sha256": "r", "code_sha256": "p",
            "train_data_sha256": "t", "validation_data_sha256": "v",
            "train": {"per_sequence": [row] * 196}, "validation": {"per_sequence": [row] * 196}}


def representation(value):
    row = {"sequence_sha256": "a" * 64, "pairs": [[0, 5]], "labels": [1.0], "logits": [value]}
    return {"arm": "length_s42", "training_seed": 42, "best_step": 1, "selection_seed": 42,
            "checkpoint_sha256": "c", "run_meta_sha256": "r", "code_sha256": "p",
            "train_data_sha256": "t", "validation_data_sha256": "v", "protocol": {},
            "linear_weight_sha256": "w", "linear_bias_sha256": "b",
            "train": {"records": [row] * 196}, "validation": {"records": [row] * 196}}


assert math.isclose(compare_error(error(1.0), error(1.0 + 1e-9)), 1e-9, abs_tol=1e-15)
assert math.isclose(compare_representation(representation(1.0), representation(1.0 + 1e-6)),
                    1e-6, abs_tol=1e-15)
for left, right, function in ((error(1.0), error(1.1), compare_error),
                              (representation(1.0), representation(1.1), compare_representation)):
    try:
        function(left, right)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-equivalent replication must be rejected")
print("PASS capacity replication tolerances and failure injection")
