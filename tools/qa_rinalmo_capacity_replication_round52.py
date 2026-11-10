import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_error(original, repeat):
    for key in ("arm", "training_seed", "best_step", "selection_seed", "checkpoint_sha256",
                "run_meta_sha256", "code_sha256", "train_data_sha256", "validation_data_sha256"):
        assert original[key] == repeat[key]
    maximum = 0.0
    for split in ("train", "validation"):
        left = original[split]["per_sequence"]
        right = repeat[split]["per_sequence"]
        assert len(left) == len(right) == 196
        for first, second in zip(left, right):
            assert first["sequence_sha256"] == second["sequence_sha256"]
            assert first["length"] == second["length"]
            assert first["losses"].keys() == second["losses"].keys()
            for key in first["losses"]:
                maximum = max(maximum, abs(first["losses"][key] - second["losses"][key]))
    assert maximum <= 1e-8
    return maximum


def compare_representation(original, repeat):
    for key in ("arm", "training_seed", "best_step", "selection_seed", "checkpoint_sha256",
                "run_meta_sha256", "code_sha256", "train_data_sha256", "validation_data_sha256",
                "protocol", "linear_weight_sha256", "linear_bias_sha256"):
        assert original[key] == repeat[key]
    maximum = 0.0
    for split in ("train", "validation"):
        left = original[split]["records"]
        right = repeat[split]["records"]
        assert len(left) == len(right) == 196
        for first, second in zip(left, right):
            assert first["sequence_sha256"] == second["sequence_sha256"]
            assert first["pairs"] == second["pairs"] and first["labels"] == second["labels"]
            assert len(first["logits"]) == len(second["logits"])
            for a, b in zip(first["logits"], second["logits"]):
                maximum = max(maximum, abs(a - b))
    assert maximum <= 1e-5
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audits = Path(args.root) / "audits/round52"
    results = {}
    for arm in ("length", "olr"):
        key = f"{arm}_s42"
        error_path = audits / f"rinalmo_capacity_error_{key}.json"
        error_repeat_path = audits / f"rinalmo_capacity_error_{key}.repeat.json"
        representation_path = audits / f"rinalmo_capacity_representation_{key}.json"
        representation_repeat_path = audits / f"rinalmo_capacity_representation_{key}.repeat.json"
        error = json.loads(error_path.read_text())
        error_repeat = json.loads(error_repeat_path.read_text())
        representation = json.loads(representation_path.read_text())
        representation_repeat = json.loads(representation_repeat_path.read_text())
        results[key] = {
            "error_max_abs_loss_difference": compare_error(error, error_repeat),
            "representation_max_abs_logit_difference": compare_representation(
                representation, representation_repeat),
            "error_sha256": sha256_file(error_path), "error_repeat_sha256": sha256_file(error_repeat_path),
            "representation_sha256": sha256_file(representation_path),
            "representation_repeat_sha256": sha256_file(representation_repeat_path),
        }
    result = {
        "status": "PASS_RINALMO_CAPACITY_INDEPENDENT_CUDA_REPLICATION",
        "scientific_conclusion": False, "equivalent": True, "arms": sorted(results),
        "results": results, "tolerances": {
            "error_max_abs_loss_difference": 1e-8,
            "representation_max_abs_logit_difference": 1e-5,
            "linear_weight_and_bias_sha256": "exact",
        },
        "code_sha256": sha256_file(__file__),
        "limitations": [
            "Replication covers both arms at seed42 for error and representation probes; all six first-pass probes remain required.",
            "Exact checkpoint/data/protocol/linear-weight identity is required; tolerances cover only numeric GPU replication.",
            "No capacity expansion, structure-performance, or test-set claim is licensed by replication alone.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
