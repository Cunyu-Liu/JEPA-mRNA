import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


SPLITS = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair_set(rows, minimum_span=0):
    return {tuple(pair) for pair in rows if abs(pair[1] - pair[0]) >= minimum_span}


def count_pairs(predicted, truth, minimum_span=0):
    pred = pair_set(predicted, minimum_span)
    gt = pair_set(truth, minimum_span)
    return {"tp": len(pred & gt), "fp": len(pred - gt), "fn": len(gt - pred)}


def scores(counts, empty_f1):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else empty_f1
    recall = tp / (tp + fn) if tp + fn else empty_f1
    denominator = 2 * tp + fp + fn
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / denominator if denominator else empty_f1,
            "inf": math.sqrt(precision * recall)}


def close(left, right):
    assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12), (left, right)


def assert_nested(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_nested(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and not isinstance(right, bool)
        close(left, right)
    else:
        assert left == right


def length_bucket(length):
    return 0 if length <= 100 else 1 if length <= 200 else 2 if length <= 400 else 3


def validate_result(result, raw_rows, decoder="nussinov", prior_override=None,
                    require_pairjepa=True):
    assert result["device"] == "cuda" and result["gpu_name"]
    assert result["encoder_load"]["strict_identity"]
    assert result["repeats"] == 3 and result["score_forward_count_per_sequence"] == 1
    assert result["pairjepa_teacher_forward_guard"] is require_pairjepa
    assert result["decoder"] == decoder and result["prior_weight_override"] == prior_override
    assert result["raw_n"] == len(raw_rows)
    assert result["n"] + result["n_long_unsupported"] == result["raw_n"]
    assert result["status"] in {"PASS", "PARTIAL_STRUCTRFM_POSITION_LIMIT"}
    unsupported = result["long_unsupported"]
    assert result["n_long_unsupported"] == len(unsupported)
    indices = [row["index"] for row in unsupported]
    assert len(indices) == len(set(indices)) and all(0 <= index < len(raw_rows) for index in indices)
    for index, row in zip(indices, unsupported):
        assert row["name"] == raw_rows[index].get("name", "")
        assert row["length"] == len(raw_rows[index]["seq"])
    supported = [index for index in range(len(raw_rows)) if index not in set(indices)]
    assert result["n"] == len(supported) == len(result["per_sequence"])
    assert len(result["predictions"]) == 3
    assert result["predictions"][1] == result["predictions"][0]
    assert result["predictions"][2] == result["predictions"][0]
    assert all(len(repeat) == result["n"] for repeat in result["predictions"])
    total = {"tp": 0, "fp": 0, "fn": 0}
    span_totals = {30: {"tp": 0, "fp": 0, "fn": 0}, 100: {"tp": 0, "fp": 0, "fn": 0}}
    buckets = [{"n": 0, "tp": 0, "fp": 0, "fn": 0, "f1": []} for _ in range(4)]
    macro_f1, macro_inf, span_macro = [], [], {30: [], 100: []}
    for index, row, prediction in zip(supported, result["per_sequence"],
                                      result["predictions"][0]):
        source = raw_rows[index]
        name = source.get("name", "")
        assert row["name"] == prediction["name"] == name
        assert row["length"] == len(source["seq"])
        predicted = prediction["pred_pairs"]
        assert len(predicted) == len(pair_set(predicted))
        assert all(isinstance(pair, list) and len(pair) == 2 and
                   0 <= pair[0] < pair[1] < len(source["seq"]) for pair in predicted)
        counts = count_pairs(predicted, source["pairs"])
        metric = scores(counts, 1.0)
        for key in ("tp", "fp", "fn"):
            assert row[key] == counts[key]
            total[key] += counts[key]
        close(row["f1"], metric["f1"])
        close(row["inf"], metric["inf"])
        macro_f1.append(metric["f1"])
        macro_inf.append(metric["inf"])
        bucket = buckets[length_bucket(row["length"])]
        bucket["n"] += 1
        bucket["f1"].append(metric["f1"])
        for key in ("tp", "fp", "fn"):
            bucket[key] += counts[key]
        for span in (30, 100):
            span_counts = count_pairs(predicted, source["pairs"], span)
            span_metric = scores(span_counts, 1.0)["f1"]
            for key in ("tp", "fp", "fn"):
                assert row[f"span_{span}"][key] == span_counts[key]
                span_totals[span][key] += span_counts[key]
            close(row[f"span_{span}"]["f1"], span_metric)
            span_macro[span].append(span_metric)
    micro = scores(total, 0.0)
    for key in ("tp", "fp", "fn"):
        assert result["micro"][key] == total[key]
    for key in ("f1", "precision", "recall"):
        close(result["micro"][key], micro[key])
    close(result["micro_inf"], micro["inf"])
    close(result["macro_f1"], statistics.fmean(macro_f1))
    close(result["macro_inf"], statistics.fmean(macro_inf))
    assert result["calibration"]["n"] > 0
    assert all(math.isfinite(result["calibration"][key]) for key in ("ece", "brier", "nll"))
    length_stats = []
    for bucket in buckets:
        denominator = 2 * bucket["tp"] + bucket["fp"] + bucket["fn"]
        length_stats.append({"n": bucket["n"], "tp": bucket["tp"], "fp": bucket["fp"],
                             "fn": bucket["fn"],
                             "micro_f1": 2 * bucket["tp"] / denominator if denominator else None,
                             "macro_f1": statistics.fmean(bucket["f1"]) if bucket["f1"] else None})
    span_stats = {}
    for span in (30, 100):
        denominator = 2 * span_totals[span]["tp"] + span_totals[span]["fp"] + span_totals[span]["fn"]
        span_stats[f"span_{span}"] = {
            **span_totals[span],
            "micro_f1": 2 * span_totals[span]["tp"] / denominator if denominator else None,
            "macro_f1": statistics.fmean(span_macro[span])}
    return {"f1": micro["f1"], "macro_f1": statistics.fmean(macro_f1),
            "micro_inf": micro["inf"], "macro_inf": statistics.fmean(macro_inf),
            **total, "precision": micro["precision"], "recall": micro["recall"],
            "length_buckets": length_stats, **span_stats}


def validate_cell(root, experiment, split, seed, run_meta, checkpoint_hash, checkpoint_step):
    path = root / f"eval_decision/{experiment}/s{seed}/{split}.json"
    result = json.loads(path.read_text())
    data_path = root / f"data/structrfm_pretrain_clean/{split}.jsonl"
    audit = json.loads(data_path.with_suffix(".audit.json").read_text())
    assert result["data_sha256"] == audit["clean_sha256"] == sha256(data_path)
    assert result["raw_n"] == audit["clean_n"]
    assert result["checkpoint_sha256"] == checkpoint_hash
    assert result["checkpoint_step"] == checkpoint_step == run_meta["best_step"]
    decoder = "greedy" if experiment.endswith("_greedy") else "nussinov"
    prior = 0.0 if experiment.endswith("_turner0") else None
    stats = validate_result(result, load_jsonl(data_path), decoder, prior,
                            require_pairjepa=run_meta["scorer"] == "pairjepa")
    return path, result, stats


def qa_experiment(root, experiment):
    root = Path(root)
    base = experiment.removesuffix("_greedy").removesuffix("_turner0")
    sources, runs = {}, {}
    manifest = json.loads((root / "audits/round48/pairjepa_preregistered.json").read_text())
    seeds = manifest["training"].get("seeds", [0, 1, 2])
    for seed in seeds:
        run = root / f"runs/{base}_s{seed}"
        meta = json.loads((run / "run_meta.json").read_text())
        assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
        assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
        assert meta["seed"] == seed and meta["scorer"] in {"pairjepa", "pairmlp"}
        expected_context = "context_v1" if meta["scorer"] == "pairjepa" else None
        assert meta["objective"].get("jepa_context") == expected_context
        assert meta["environment"]["resolved_device"].startswith("cuda")
        checkpoint = run / "best.pt"
        checkpoint_hash = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["meta"]["objective"].get("jepa_context") == expected_context
        runs[seed] = meta, checkpoint_hash, payload["step"]
        sources[str(checkpoint.relative_to(root))] = checkpoint_hash
    split_stats = {}
    for split in SPLITS:
        cells = []
        for seed in seeds:
            path, result, stats = validate_cell(root, experiment, split, seed, *runs[seed])
            sources[str(path.relative_to(root))] = sha256(path)
            cells.append({"seed": seed, **stats, "n": result["n"],
                          "n_long_unsupported": result["n_long_unsupported"],
                          "status": result["status"], "result_sha256": sha256(path),
                          "checkpoint_sha256": result["checkpoint_sha256"],
                          "ece": result["calibration"]["ece"],
                          "brier": result["calibration"]["brier"],
                          "nll": result["calibration"]["nll"]})
        split_stats[split] = {"n": cells[0]["n"] + cells[0]["n_long_unsupported"],
                              "cells": cells,
                              "f1_mean": statistics.fmean(cell["f1"] for cell in cells),
                              "f1_sd": statistics.pstdev(cell["f1"] for cell in cells),
                              "macro_f1_mean": statistics.fmean(cell["macro_f1"] for cell in cells),
                              "macro_f1_sd": statistics.pstdev(cell["macro_f1"] for cell in cells),
                              "macro_inf_mean": statistics.fmean(cell["macro_inf"] for cell in cells),
                              "micro_inf_mean": statistics.fmean(cell["micro_inf"] for cell in cells),
                              "ece_mean": statistics.fmean(cell["ece"] for cell in cells),
                              "brier_mean": statistics.fmean(cell["brier"] for cell in cells),
                              "nll_mean": statistics.fmean(cell["nll"] for cell in cells),
                              "n_long_unsupported": sum(cell["n_long_unsupported"] for cell in cells)}
        assert all(cell["n"] + cell["n_long_unsupported"] == split_stats[split]["n"] for cell in cells)
    summary_path = root / f"eval_decision/{experiment}/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"].startswith("PASS_")
    assert_nested({split: summary["splits"][split] for split in SPLITS}, split_stats)
    sources[str(summary_path.relative_to(root))] = sha256(summary_path)
    sota_path = root / f"eval_decision/{experiment}/sota_comparison.json"
    if sota_path.exists():
        sota = json.loads(sota_path.read_text())
        assert sota["result_sha256"] == sha256(summary_path)
        sources[str(sota_path.relative_to(root))] = sha256(sota_path)
    return {"status": "PASS_PAIRJEPA_INDEPENDENT_QA", "complete": True,
            "experiment": experiment, "n_cells": len(SPLITS) * 3, "split_stats": split_stats,
            "file_sha256": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = qa_experiment(args.root, args.experiment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "experiment": args.experiment,
                      "cells": output["n_cells"], "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
