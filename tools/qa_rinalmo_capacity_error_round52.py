import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_losses(rows):
    keys = rows[0]["losses"]
    return {key: math.fsum(row["losses"][key] for row in rows) / len(rows) for key in keys}


def validate_probe(input_path, probe_code):
    input_path = Path(input_path)
    probe = json.loads(input_path.read_text())
    assert probe["status"] == "PASS_RINALMO_FROZEN_MODEL_TRAIN_VL0_ERROR_PROBE"
    assert probe["scientific_conclusion"] is False
    assert probe["allow_capacity_expansion"] is False
    assert probe["arm"].startswith(("round52_rinalmo_length_s", "round52_rinalmo_olr_s"))
    assert probe["selection_seed"] == 42
    assert probe["resolved_device"] == "cuda" and probe["peak_cuda_bytes"] > 0
    assert probe["code_sha256"] == sha256_file(probe_code)
    checkpoint = Path(probe["checkpoint"])
    run_meta_path = checkpoint.parent / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    assert probe["checkpoint_sha256"] == sha256_file(checkpoint)
    assert probe["run_meta_sha256"] == sha256_file(run_meta_path)
    assert run_meta["arm"] == probe["arm"] and run_meta["seed"] == probe["training_seed"]
    assert run_meta["best_step"] == probe["best_step"]
    assert run_meta["best_validation_loss"] == probe["validation"]["mean_losses"]["total"]
    assert probe["validation_data_sha256"] == sha256_file(run_meta["validation"]["data"])
    assert probe["train"]["n_sequences"] == probe["validation"]["n_sequences"] == 196
    assert len(probe["train"]["per_sequence"]) == len(probe["train_lengths"]) == 196
    assert len(probe["validation"]["per_sequence"]) == len(probe["validation_lengths"]) == 196
    assert [row["length"] for row in probe["train"]["per_sequence"]] == probe["train_lengths"]
    assert [row["length"] for row in probe["validation"]["per_sequence"]] == probe["validation_lengths"]
    train = mean_losses(probe["train"]["per_sequence"])
    validation = mean_losses(probe["validation"]["per_sequence"])
    for split, recomputed in (("train", train), ("validation", validation)):
        for key, value in recomputed.items():
            assert math.isclose(value, probe[split]["mean_losses"][key], rel_tol=1e-12, abs_tol=1e-10)
    assert math.isclose(probe["total_train_minus_validation"], train["total"] - validation["total"],
                        rel_tol=0, abs_tol=1e-9)
    assert all(len(row["sequence_sha256"]) == 64
               for split in ("train", "validation") for row in probe[split]["per_sequence"])
    return {
        "status": "PASS_RINALMO_CAPACITY_ERROR_INDEPENDENT_QA",
        "scientific_conclusion": False, "allow_capacity_expansion": False,
        "arm": probe["arm"], "training_seed": probe["training_seed"],
        "best_step": probe["best_step"], "selection_seed": probe["selection_seed"],
        "train_mean_losses": train, "validation_mean_losses": validation,
        "total_train_minus_validation": train["total"] - validation["total"],
        "checkpoint_sha256": probe["checkpoint_sha256"],
        "run_meta_sha256": probe["run_meta_sha256"],
        "probe_code_sha256": probe["code_sha256"],
        "input_sha256": sha256_file(input_path),
        "train_data_sha256": probe["train_data_sha256"],
        "validation_data_sha256": probe["validation_data_sha256"],
        "teacher_manifest_sha256": probe["teacher_manifest_sha256"],
        "validation_teacher_manifest_sha256": probe["validation_teacher_manifest_sha256"],
        "limitations": [
            "Independent arithmetic, identity, and hash QA of a frozen-checkpoint diagnostic; no test-set access.",
            "This QA alone cannot authorize C-wide or C-local; stability, representation probes, and all L/O arms are required.",
            "Train/VL0 objective errors are not structure F1 or proof of sufficient capacity.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--probe-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_probe(args.input, args.probe_code)
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
