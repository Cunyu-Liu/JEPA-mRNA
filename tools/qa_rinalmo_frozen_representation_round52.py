import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute_split(split):
    logits = []
    labels = []
    for row in split["records"]:
        assert len(row["sequence_sha256"]) == 64 and row["length"] > 0
        assert len(row["pairs"]) == len(row["labels"]) == len(row["logits"])
        assert 0 < len(row["pairs"]) <= 256
        assert all(label in (0.0, 1.0) for label in row["labels"])
        assert all(len(pair) == 2 and 0 <= min(pair) < max(pair) < row["length"]
                   for pair in row["pairs"])
        assert all(math.isfinite(value) for value in row["logits"])
        logits.extend(row["logits"])
        labels.extend(row["labels"])
    positives = sum(labels)
    negatives = len(labels) - positives
    assert positives > 0 and negatives > 0
    positive_weight = negatives / positives
    losses = [max(logit, 0) - logit * label + math.log1p(math.exp(-abs(logit)))
              for logit, label in zip(logits, labels)]
    weights = [positive_weight if label else 1.0 for label in labels]
    nll = math.fsum(loss * weight for loss, weight in zip(losses, weights)) / len(losses)
    brier = math.fsum((1 / (1 + math.exp(-logit)) - label) ** 2
                      for logit, label in zip(logits, labels)) / len(labels)
    assert math.isclose(nll, split["weighted_nll"], rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(brier, split["brier"], rel_tol=1e-8, abs_tol=1e-6)
    assert len(logits) == split["n_pairs"]
    assert positives == split["n_positive"] and negatives == split["n_negative"]
    return {"weighted_nll": nll, "brier": brier, "n_pairs": len(logits),
            "n_positive": int(positives), "n_negative": int(negatives)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--probe-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe = json.loads(Path(args.input).read_text())
    assert probe["status"] == "PASS_FROZEN_RINALMO_REPRESENTATION_LINEAR_PROBE"
    assert probe["scientific_conclusion"] is False and probe["allow_capacity_expansion"] is False
    assert probe["selection_seed"] == 42 and probe["resolved_device"] == "cuda"
    assert probe["code_sha256"] == sha256_file(args.probe_code)
    checkpoint = Path(probe["checkpoint"])
    run_meta_path = checkpoint.parent / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    assert probe["checkpoint_sha256"] == sha256_file(checkpoint)
    assert probe["run_meta_sha256"] == sha256_file(run_meta_path)
    assert run_meta["arm"] == probe["arm"] and run_meta["seed"] == probe["training_seed"]
    assert run_meta["best_step"] == probe["best_step"]
    protocol = probe["protocol"]
    assert protocol["full_passes"] == 200 and protocol["samples_per_sequence"] == 256
    assert protocol["early_stopping_or_validation_tuning"] is False
    train = recompute_split(probe["train"])
    validation = recompute_split(probe["validation"])
    result = {
        "status": "PASS_FROZEN_REPRESENTATION_INDEPENDENT_QA",
        "scientific_conclusion": False, "allow_capacity_expansion": False,
        "arm": probe["arm"], "training_seed": probe["training_seed"],
        "best_step": probe["best_step"], "train": train, "validation": validation,
        "checkpoint_sha256": probe["checkpoint_sha256"],
        "run_meta_sha256": probe["run_meta_sha256"],
        "probe_code_sha256": probe["code_sha256"],
        "input_sha256": sha256_file(args.input),
        "qa_sha256": sha256_file(__file__),
        "limitations": [
            "Independent arithmetic and identity QA of frozen-representation logits, not structure-performance evidence.",
            "This QA does not authorize capacity expansion or replace training stability and error probes.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
