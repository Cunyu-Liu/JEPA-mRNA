import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from qa_rinalmo_b0_training_round54 import read_jsonl, sha256_file, validate_run

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
SOURCE = ROOT / "data/rna_complex_round49"
LENGTH_DATA = SOURCE / "rinalmo_d1_length_round54/1022"
TRAIN = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
AUX_TEACHER = SOURCE / "rinalmo_full_d1/teacher_complex"


def sequence_hash(sequence):
    return hashlib.sha256(sequence.upper().replace("T", "U").encode()).hexdigest()


def write_atomic(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"run", "buckets", "per_sequence_coverage"}}))


def training_qa(seed):
    run = ROOT / "runs" / f"round54_d1_s{seed}"
    report = validate_run(run, seed, sha256_file(LENGTH_DATA / "train_d1.jsonl"), arm="d1")
    meta = json.loads((run / "run_meta.json").read_text())
    baseline = json.loads((ROOT / "runs" / f"round54_b0_s{seed}/run_meta.json").read_text())
    assert meta["data"] == str(LENGTH_DATA / "train_d1.jsonl")
    assert meta["schedule"] == baseline["schedule"]
    assert meta["validation"] == baseline["validation"]
    assert meta["backbone_report"] == baseline["backbone_report"]
    assert meta["scorer"] == baseline["scorer"] == "resnet2d"
    assert meta["pair_latent_dim"] == baseline["pair_latent_dim"] == 64
    assert meta["ripe"] == baseline["ripe"]
    assert meta["objective"]["aux_teacher"] == str(AUX_TEACHER)
    assert meta["objective"]["teacher_storage"] == {"lazy": True, "shard_cache_size": 1}
    for key in ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal",
                "distill_kind", "rlcd_reward", "beta", "n_bins", "tau", "nll_normalization",
                "lambda_jepa", "jepa_target", "jepa_context", "jepa_sigreg_weight"):
        assert meta["objective"][key] == baseline["objective"][key]
    assert meta["objective"]["lambda_jepa"] == 0.0
    train_rows = read_jsonl(LENGTH_DATA / "train_d1.jsonl")
    base_rows = read_jsonl(TRAIN)
    assert train_rows[:len(base_rows)] == base_rows
    raw_new = [row for row in read_jsonl(SOURCE / "decontamination/complex_train.jsonl")
               if len(row["seq"]) <= 1022]
    expected = [sequence_hash(row["seq"]) for row in raw_new]
    assert len(expected) == len(set(expected)) == 155
    assert Counter(sequence_hash(row["seq"]) for row in train_rows[len(base_rows):]) == Counter(expected)
    ledger = read_jsonl(run / "compute_ledger.jsonl")
    assert [row["step"] for row in ledger] == list(range(1, meta["steps_completed"] + 1))
    assert all(len(row["sequence_sha256"]) == 4 for row in ledger)
    assert all(len(row["sequence_sha256"]) == len(row["lengths"]) for row in ledger)
    counts = Counter(key for row in ledger for key in row["sequence_sha256"] if key in set(expected))
    coverage = {key: counts.get(key, 0) for key in expected}
    report["status"] = "PASS_D1_SINGLE_SEED_TRAINING_QA"
    report["training_data_sha256"] = sha256_file(LENGTH_DATA / "train_d1.jsonl")
    report["teacher_manifest_sha256"] = sha256_file(AUX_TEACHER / "manifest.json")
    report["teacher_source_sha256"] = sha256_file(SOURCE / "decontamination/complex_train.jsonl")
    report["new_chains"] = len(expected)
    report["new_chains_seen_at_least_once"] = sum(value > 0 for value in coverage.values())
    report["new_chain_batch_appearances"] = sum(coverage.values())
    report["new_chain_coverage_complete"] = all(coverage.values())
    report["per_sequence_coverage"] = coverage
    report["scientific_conclusion"] = False
    return report


def pair_metrics(predicted, truth):
    predicted, truth = set(predicted), set(truth)
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def holdout_qa(seed, arm):
    assert arm in {"b0", "d1"}
    run = ROOT / "runs" / f"round54_{arm}_s{seed}"
    meta = json.loads((run / "run_meta.json").read_text())
    data = LENGTH_DATA / "complex_holdout.jsonl"
    result_path = ROOT / "evals/round54" / f"{arm}_s{seed}_complex_holdout_1022/result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "PASS" and result["device"] == "cuda"
    assert result["n"] == result["raw_n"] == 96 and result["n_long_unsupported"] == 0
    assert result["decoder"] == "nussinov" and result["dp_backend"] == "c"
    assert result["repeats"] == 3 and result["score_forward_count_per_sequence"] == 1
    assert result["prior_weight_override"] is None and result["prior_weight_trained"] is not None
    assert result["data_sha256"] == sha256_file(data)
    assert result["checkpoint_sha256"] == sha256_file(run / "best.pt")
    assert result["checkpoint_step"] == meta["best_step"]
    assert meta["status"] == "completed" and meta["converged"]
    rows = read_jsonl(data)
    full_rows = read_jsonl(SOURCE / "decontamination/complex_holdout.jsonl")
    assert rows == [row for row in full_rows if len(row["seq"]) <= 1022]
    assert len(full_rows) == 202 and len(rows) == 96
    assert result["predictions"][1] == result["predictions"][0]
    assert result["predictions"][2] == result["predictions"][0]
    buckets = {"all": [], "le400": [], "401_1022": []}
    for index, (row, prediction, saved) in enumerate(zip(rows, result["predictions"][0], result["per_sequence"])):
        assert prediction["name"] == row.get("name", "") == saved["name"]
        assert saved["length"] == len(row["seq"])
        predicted = [tuple(pair) for pair in prediction["pred_pairs"]]
        truth = [tuple(pair) for pair in row["pairs"]]
        assert len(predicted) == len(set(predicted))
        assert all(0 <= left < right < len(row["seq"]) for left, right in predicted)
        assert all(row["seq"][left] + row["seq"][right] in {"AU", "UA", "GC", "CG", "GU", "UG"}
                   for left, right in predicted)
        assert all(right - left > 3 for left, right in predicted)
        assert not any(a < c < b < d or c < a < d < b
                       for (a, b), (c, d) in zip(predicted, predicted[1:]))
        metrics = pair_metrics(predicted, truth)
        for key in ("tp", "fp", "fn"):
            assert saved[key] == metrics[key]
        assert math.isclose(saved["f1"], metrics["f1"], rel_tol=0, abs_tol=1e-12)
        buckets["all"].append(metrics)
        buckets["le400" if len(row["seq"]) <= 400 else "401_1022"].append(metrics)
    summaries = {}
    for name, metrics in buckets.items():
        tp, fp, fn = (sum(row[key] for row in metrics) for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        summaries[name] = {"n": len(metrics), "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall,
                           "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                           "macro_f1": sum(row["f1"] for row in metrics) / len(metrics)}
    assert math.isclose(summaries["all"]["micro_f1"], result["micro"]["f1"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summaries["all"]["macro_f1"], result["macro_f1"], rel_tol=0, abs_tol=1e-12)
    return {"status": f"PASS_{arm.upper()}_COMPLEX_HOLDOUT_1022_QA", "arm": arm, "seed": seed,
            "checkpoint_step": meta["best_step"], "best_vl0": meta["best_validation_loss"],
            "result_sha256": sha256_file(result_path), "data_sha256": sha256_file(data),
            "checkpoint_sha256": result["checkpoint_sha256"], "buckets": summaries,
            "full_holdout_n": 202, "evaluated_n": 96, "out_of_length_protocol_n": 106,
            "out_of_length_protocol": "L>1022; retained and not counted as zero or unsupported success",
            "selection_use": "directional check only; not early stopping, calibration or arm selection"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["training", "holdout"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407], required=True)
    parser.add_argument("--arm", choices=["b0", "d1"], default="d1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = training_qa(args.seed) if args.mode == "training" else holdout_qa(args.seed, args.arm)
    write_atomic(args.output, report)


if __name__ == "__main__":
    main()
