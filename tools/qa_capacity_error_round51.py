import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequences(path):
    return [json.loads(line)["seq"].upper().replace("T", "U")
            for line in Path(path).read_text().splitlines() if line.strip()]


def independently_summarize(split):
    rows = split["per_sequence"]
    assert split["n_sequences"] == len(rows) == 196
    assert len({row["sequence_sha256"] for row in rows}) == 196
    assert all(row["length"] > 0 and row["n_gt_pairs"] >= 0 for row in rows)
    keys = tuple(sorted(rows[0]["losses"]))
    assert all(set(row["losses"]) == set(keys) for row in rows)
    means = {key: math.fsum(row["losses"][key] for row in rows) / len(rows)
             for key in keys}
    assert all(math.isfinite(value) for value in means.values())
    for key, value in means.items():
        assert math.isclose(value, split["mean_losses"][key],
                            rel_tol=1e-12, abs_tol=1e-12)
    buckets = {}
    for name, lower, upper in (("<=100", 0, 100), ("100-200", 100, 200),
                               ("200-400", 200, 400), (">400", 400, math.inf)):
        selected = [row for row in rows if lower < row["length"] <= upper]
        buckets[name] = {
            "n": len(selected),
            "mean_losses": {key: (math.fsum(row["losses"][key] for row in selected)
                                  / len(selected) if selected else None) for key in keys},
        }
    return means, buckets


def validate_document(document, root, seed):
    assert document["status"] == "PASS_FROZEN_MODEL_TRAIN_VL0_ERROR_PROBE"
    assert document["scientific_conclusion"] is False
    assert document["training_seed"] == seed and document["selection_seed"] == 42
    assert document["resolved_device"].startswith("cuda")
    assert document["gpu_name"] == "NVIDIA GeForce RTX 4090"
    run = root / "runs" / f"loss_length_s{seed}"
    meta = json.loads((run / "run_meta.json").read_text())
    assert document["arm"] == meta["arm"] and document["best_step"] == meta["best_step"]
    assert document["checkpoint_sha256"] == sha256_file(run / "best.pt")
    assert document["run_meta_sha256"] == sha256_file(run / "run_meta.json")
    train_means, train_buckets = independently_summarize(document["train"])
    validation_means, validation_buckets = independently_summarize(document["validation"])
    assert math.isclose(validation_means["total"], meta["best_validation_loss"],
                        rel_tol=0, abs_tol=1e-6)
    train_data = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    validation_data = root / "data/validation/vl0_196.jsonl"
    assert document["train_data_sha256"] == sha256_file(train_data)
    assert document["validation_data_sha256"] == sha256_file(validation_data)
    chosen = np.random.default_rng(42).choice(42564, size=196, replace=False)
    train_sequences = sequences(train_data)
    assert len(train_sequences) == 42564
    expected_train = [train_sequences[int(index)] for index in chosen]
    expected_validation = sequences(validation_data)
    assert [row["sequence_sha256"] for row in document["train"]["per_sequence"]] == [
        hashlib.sha256(seq.encode()).hexdigest() for seq in expected_train]
    assert [row["sequence_sha256"] for row in document["validation"]["per_sequence"]] == [
        hashlib.sha256(seq.encode()).hexdigest() for seq in expected_validation]
    assert set(expected_train).isdisjoint(expected_validation)
    return {
        "seed": seed,
        "best_step": meta["best_step"],
        "train_mean_losses": train_means,
        "validation_mean_losses": validation_means,
        "train_minus_validation": {key: train_means[key] - validation_means[key]
                                   for key in train_means},
        "train_length_buckets": train_buckets,
        "validation_length_buckets": validation_buckets,
        "checkpoint_sha256": document["checkpoint_sha256"],
        "code_sha256": document["code_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label", default="v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    reports = {}
    for seed in (42, 2026, 3407):
        path = root / "audits/round51" / f"capacity_error_{args.label}_s{seed}.json"
        reports[str(seed)] = validate_document(json.loads(path.read_text()), root, seed)
    assert len({report["code_sha256"] for report in reports.values()}) == 1
    train_identities = []
    for seed in (42, 2026, 3407):
        path = root / "audits/round51" / f"capacity_error_{args.label}_s{seed}.json"
        payload = json.loads(path.read_text())
        train_identities.append([row["sequence_sha256"]
                                 for row in payload["train"]["per_sequence"]])
    assert train_identities[0] == train_identities[1] == train_identities[2]
    keys = reports["42"]["train_mean_losses"].keys()
    output = {
        "status": "PASS_CAPACITY_ERROR_INDEPENDENT_QA",
        "scientific_conclusion": False,
        "label": args.label,
        "reports": reports,
        "three_seed_mean": {
            "train": {key: math.fsum(reports[str(seed)]["train_mean_losses"][key]
                                     for seed in (42, 2026, 3407)) / 3 for key in keys},
            "validation": {key: math.fsum(reports[str(seed)]["validation_mean_losses"][key]
                                          for seed in (42, 2026, 3407)) / 3 for key in keys},
        },
        "limitations": [
            "This QA independently recomputes sequence identities and loss aggregates, not structure F1.",
            "Training/VL0 error proximity alone neither proves sufficient capacity nor licenses widening.",
            "No test set, checkpoint selection, or architecture change is involved in this diagnostic.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
