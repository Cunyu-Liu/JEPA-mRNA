import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from eval.ss.metrics import PairLevelMetrics
from rnajepa.harness import valid_pair_mask


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def micro_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def validate_result(path, checkpoint, data, mode):
    path, checkpoint, data = Path(path), Path(checkpoint), Path(data)
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and result["device"] == "cuda"
    assert result["raw_n"] == result["n"] == 196 and result["n_long_unsupported"] == 0
    assert result["repeats"] == 3 and result["score_forward_count_per_sequence"] == 1
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["data_sha256"] == sha256_file(data)
    decoder, prior = ("nussinov", 0.0) if mode == "turner0" else (mode, None)
    assert result["decoder"] == decoder and result["prior_weight_override"] == prior
    run_meta = json.loads((checkpoint.parent / "run_meta.json").read_text())
    assert run_meta["status"] == "completed" and run_meta["converged"]
    assert run_meta["best_step"] == result["checkpoint_step"]
    assert run_meta["scorer"] == "pairmlp" and run_meta["arm"].startswith("round52_rinalmo_length_s")
    assert run_meta["validation"]["data"] == str(data)
    records = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert len(records) == len(result["per_sequence"]) == 196
    assert len(result["predictions"]) == 3
    tp = fp = fn = 0
    per_sequence_f1 = []
    for index, (source, row) in enumerate(zip(records, result["per_sequence"])):
        sequence = source["seq"].upper().replace("T", "U")
        ground_truth = [tuple(pair) for pair in source["pairs"]]
        assert row["name"] == source.get("name", "") and row["length"] == len(sequence)
        repeat_pairs = []
        for repeat in result["predictions"]:
            prediction = repeat[index]
            assert prediction["name"] == row["name"]
            pairs = [tuple(pair) for pair in prediction["pred_pairs"]]
            repeat_pairs.append(pairs)
        assert repeat_pairs[0] == repeat_pairs[1] == repeat_pairs[2]
        metrics = PairLevelMetrics.from_pairs(repeat_pairs[0], ground_truth,
                                              L=len(sequence), mask=valid_pair_mask(sequence))
        assert (row["tp"], row["fp"], row["fn"]) == (metrics.tp, metrics.fp, metrics.fn)
        assert math.isclose(row["f1"], metrics.f1, rel_tol=0, abs_tol=1e-12)
        tp += metrics.tp
        fp += metrics.fp
        fn += metrics.fn
        per_sequence_f1.append(metrics.f1)
    micro = micro_metrics(tp, fp, fn)
    for key, value in micro.items():
        assert math.isclose(value, result["micro"][key], rel_tol=0, abs_tol=1e-12)
    macro = math.fsum(per_sequence_f1) / len(per_sequence_f1)
    assert math.isclose(macro, result["macro_f1"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(result["micro_inf"], math.sqrt(micro["precision"] * micro["recall"]),
                        rel_tol=0, abs_tol=1e-12)
    return {"file": path.name, "mode": mode, "micro": micro, "macro_f1": macro,
            "checkpoint_sha256": result["checkpoint_sha256"], "result_sha256": sha256_file(path),
            "n": 196, "repeat_predictions_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--family", choices=["jmlp"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    control = json.loads((root / "audits/round52/rinalmo_jmlp_training_qa.json").read_text())
    assert control["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA" and control["base"] == "length"
    results = {}
    for seed in (42, 2026, 3407):
        checkpoint = root / "runs" / f"round52_rinalmo_length_s{seed}/best.pt"
        for mode in ("nussinov", "greedy", "turner0"):
            path = root / "evals/round52" / f"rinalmo_jmlp_length_s{seed}_vl0_{mode}.json"
            results[f"{seed}.{mode}"] = validate_result(
                path, checkpoint, root / "data/validation/vl0_196.jsonl", mode)
    aggregates = {}
    for mode in ("nussinov", "greedy", "turner0"):
        values = [results[f"{seed}.{mode}"]["micro"]["f1"] for seed in (42, 2026, 3407)]
        aggregates[mode] = {"seed_micro_f1": values, "mean_micro_f1": statistics.mean(values),
                            "population_sd_micro_f1": statistics.pstdev(values)}
    result = {"status": "PASS_JMLP_VL0_STRUCTURE_INDEPENDENT_QA", "scientific_conclusion": False,
              "results": results, "aggregates": aggregates,
              "control_sha256": sha256_file(root / "audits/round52/rinalmo_jmlp_training_qa.json"),
              "qa_sha256": sha256_file(__file__),
              "limitations": [
                  "VL0-only structural diagnostics; no test set was accessed or used for tuning, calibration or selection.",
                  "Concurrent wall times are not formal speed results; calibration/confidence fields are not independently accepted here.",
                  "These are not final benchmark or Rfam results and do not establish JEPA efficacy.",
              ]}
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
