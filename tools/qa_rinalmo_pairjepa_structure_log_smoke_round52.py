import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    meta = json.loads((root / "run_meta.json").read_text())
    assert meta["arm"] == "round52_rinalmo_pairjepa_structure_log_smoke" and meta["seed"] == 42
    assert meta["status"] == "completed" and meta["steps_completed"] == 1
    assert meta["schedule"]["steps"] == 1 and not meta["converged"]
    assert meta["scorer"] == "pairjepa" and meta["objective"]["jepa_context"] == "structured_block"
    assert meta["environment"]["resolved_device"].startswith("cuda")
    rows = [json.loads(line) for line in (root / "jepa_diagnostics.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 1
    structure = rows[0]["training_batch_structure"]
    assert set(structure) == {"nussinov", "greedy", "greedy_turner0"}
    for metrics in structure.values():
        assert set(metrics) == {"tp", "fp", "fn", "micro_f1", "macro_f1"}
        denominator = 2 * metrics["tp"] + metrics["fp"] + metrics["fn"]
        expected = 2 * metrics["tp"] / denominator if denominator else 0.0
        assert math.isclose(metrics["micro_f1"], expected, rel_tol=0, abs_tol=1e-12)
        assert 0 <= metrics["macro_f1"] <= 1
    result = {
        "status": "PASS_PAIRJEPA_STRUCTURE_LOG_ONE_STEP_CUDA_QA",
        "scientific_conclusion": False, "allow_pairjepa_training": False,
        "structure": structure, "device": "cuda",
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "diagnostics_sha256": sha256_file(root / "jepa_diagnostics.jsonl"),
        "train_log_sha256": sha256_file(root / "train_log.jsonl"),
        "qa_sha256": sha256_file(__file__),
        "limitations": [
            "One-step integration evidence only; training-batch structure metrics are never used for early stopping or arm selection.",
            "This is not a converged JEPA efficacy, calibration, or structure-performance result.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
