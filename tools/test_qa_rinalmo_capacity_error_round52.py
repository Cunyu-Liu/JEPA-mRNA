import hashlib
import json
from pathlib import Path

from qa_rinalmo_capacity_error_round52 import validate_probe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(lengths, total):
    return [{"length": length, "sequence_sha256": hashlib.sha256(str(length).encode()).hexdigest(),
             "losses": {"total": total + index, "nll": total + index - 0.5}}
            for index, length in enumerate(lengths)]


def main(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    validation = tmp_path / "vl0.jsonl"
    validation.write_text("validation")
    meta = {"arm": "round52_rinalmo_length_s42", "seed": 42, "best_step": 8000,
            "best_validation_loss": 197.5,
            "validation": {"data": str(validation)}}
    (run / "run_meta.json").write_text(json.dumps(meta))
    code = tmp_path / "probe.py"
    code.write_text("probe")
    train_rows = rows([10] * 196, 201)
    validation_rows = rows([20] * 196, 100)
    train_mean = {key: sum(row["losses"][key] for row in train_rows) / 196 for key in ("total", "nll")}
    validation_mean = {key: sum(row["losses"][key] for row in validation_rows) / 196
                       for key in ("total", "nll")}
    probe = {"status": "PASS_RINALMO_FROZEN_MODEL_TRAIN_VL0_ERROR_PROBE",
             "scientific_conclusion": False, "allow_capacity_expansion": False,
             "arm": meta["arm"], "training_seed": 42, "selection_seed": 42, "best_step": 8000,
             "resolved_device": "cuda", "peak_cuda_bytes": 1,
             "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
             "run_meta_sha256": sha(run / "run_meta.json"), "code_sha256": sha(code),
             "train_data_sha256": "train", "validation_data_sha256": sha(validation),
             "teacher_manifest_sha256": "teacher", "validation_teacher_manifest_sha256": "vteacher",
             "train_lengths": [10] * 196, "validation_lengths": [20] * 196,
             "train": {"n_sequences": 196, "mean_losses": train_mean, "per_sequence": train_rows},
             "validation": {"n_sequences": 196, "mean_losses": validation_mean,
                            "per_sequence": validation_rows},
             "total_train_minus_validation": train_mean["total"] - validation_mean["total"]}
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    result = validate_probe(path, code)
    assert result["status"] == "PASS_RINALMO_CAPACITY_ERROR_INDEPENDENT_QA"
    assert not result["allow_capacity_expansion"]
    probe["validation"]["mean_losses"]["total"] += 1
    path.write_text(json.dumps(probe))
    try:
        validate_probe(path, code)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect validation mean must be rejected")
    print("PASS capacity-error independent QA and failure injection")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        main(Path(directory))
