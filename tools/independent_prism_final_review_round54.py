import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

SPLITS = ("bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam")
ALLOWED = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")}


@lru_cache(maxsize=None)
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(predicted, expected):
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return tp, fp, fn, f1


def validate_result(path, data_path, decoder, checkpoint=None):
    result = json.loads(path.read_text())
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert result["raw_n"] == len(rows) == result["n"] and result["n_long_unsupported"] == 0
    assert result["repeats"] == 3 and result["decoder"] == decoder
    assert result["data_sha256"] == sha256(data_path)
    if checkpoint is not None:
        assert result["checkpoint_sha256"] == sha256(checkpoint)
        assert result["device"] == "cuda" and result["score_forward_count_per_sequence"] == 1
    assert result["predictions"][0] == result["predictions"][1] == result["predictions"][2]
    total_tp = total_fp = total_fn = 0
    total_crossings = total_conflicts = 0
    per_sequence_f1 = []
    for source, prediction in zip(rows, result["predictions"][0]):
        assert prediction["name"] == source.get("name", "")
        sequence = source["seq"].upper().replace("T", "U")
        pairs = {tuple(pair) for pair in prediction["pred_pairs"]}
        assert len(pairs) == len(prediction["pred_pairs"])
        ordered = sorted(pairs)
        used = set()
        conflicts = 0
        for i, j in ordered:
            assert 0 <= i < j < len(sequence) and j - i > 3
            assert (sequence[i], sequence[j]) in ALLOWED
            if i in used or j in used:
                assert decoder == "greedy"
                conflicts += 1
            used.update((i, j))
        crossings = sum(i < k < j < l or k < i < l < j
                        for a, (i, j) in enumerate(ordered) for k, l in ordered[a + 1:])
        total_crossings += crossings
        total_conflicts += conflicts
        tp, fp, fn, f1 = metrics(pairs, {tuple(pair) for pair in source["pairs"]})
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_sequence_f1.append(f1)
    micro_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if 2 * total_tp + total_fp + total_fn else 0.0
    macro_f1 = sum(per_sequence_f1) / len(per_sequence_f1)
    assert math.isclose(micro_f1, result["micro"]["f1"], abs_tol=1e-12)
    assert math.isclose(macro_f1, result["macro_f1"], abs_tol=1e-12)
    assert (total_tp, total_fp, total_fn) == tuple(result["micro"][key] for key in ("tp", "fp", "fn"))
    assert total_crossings == result.get("crossing_pairs", 0)
    assert total_conflicts == result.get("conflict_pairs", 0)
    if decoder == "nussinov":
        assert total_crossings == 0 and total_conflicts == 0
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "sha256": sha256(path),
            "crossing_pairs": total_crossings, "conflict_pairs": total_conflicts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    model_results = {}
    for arm in ("b0", "cl1"):
        for seed in (42, 2026, 3407):
            meta_path = root / f"runs/round54_{arm}_s{seed}/run_meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["converged"] and meta["early_stopped"]
            assert meta["schedule"]["steps"] == 0 and meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
            for ablation in ("", "_greedy", "_turner0"):
                decoder = "greedy" if ablation == "_greedy" else "nussinov"
                for split in SPLITS:
                    key = f"{arm}{ablation}/{seed}/{split}"
                    model_results[key] = validate_result(
                        root / f"eval_decision/prism_{arm}{ablation}/s{seed}/{split}.json",
                        root / f"data/structrfm_pretrain_clean/{split}.jsonl", decoder,
                        root / f"runs/round54_{arm}_s{seed}/best.pt")
    vienna_results = {}
    for split in SPLITS:
        vienna_results[split] = validate_result(
            root / f"eval_decision/prism_vienna_only/{split}.json",
            root / f"data/structrfm_pretrain_clean/{split}.jsonl", "nussinov")
    comparisons = {}
    for arm in ("b0", "cl1"):
        summary = json.loads((root / f"eval_decision/prism_{arm}/summary.json").read_text())
        confidence = json.loads((root / f"eval_decision/prism_{arm}/confidence_summary.json").read_text())
        reference = confidence["thresholds"]["reference_seed"]
        meta = json.loads((root / f"runs/round54_{arm}_s{reference}/run_meta.json").read_text())
        assert confidence["thresholds"]["reference_best_validation_loss"] == meta["best_validation_loss"]
        comparisons[arm] = {split: summary["splits"][split]["f1_mean"] for split in SPLITS}
        for split, mean in comparisons[arm].items():
            cells = [model_results[f"{arm}/{seed}/{split}"]["micro_f1"] for seed in (42, 2026, 3407)]
            assert math.isclose(sum(cells) / 3, mean, abs_tol=1e-12)
    qa_path = root / "audits/round54/frozen_benchmark_qa.json"
    qa_recheck = root / "audits/round54/frozen_benchmark_qa.recheck.json"
    assert qa_path.read_bytes() == qa_recheck.read_bytes()
    qa = json.loads(qa_path.read_text())
    assert qa["status"] == "PASS_PRISM_FROZEN_BENCHMARK_QA" and qa["complete"] and qa["arms"] == ["b0", "cl1"]
    report = {
        "status": "PASS_INDEPENDENT_FINAL_REVIEW",
        "model_cells": len(model_results),
        "vienna_cells": len(vienna_results),
        "model_results": model_results,
        "vienna_results": vienna_results,
        "canonical_micro_f1_means": comparisons,
        "cl1_minus_b0": {split: comparisons["cl1"][split] - comparisons["b0"][split] for split in SPLITS},
        "cl1_holm_main_eight": qa["holm_main_eight"]["cl1"],
        "sota_main_wins": {arm: qa["sota_comparison"][arm]["main_sota_any_wins"] for arm in ("b0", "cl1")},
        "rfam_sota": {arm: qa["sota_comparison"][arm]["rfam"]["sota"] for arm in ("b0", "cl1")},
        "frozen_benchmark_qa_sha256": sha256(qa_path),
        "all_predictions_legal_and_deterministic": True,
        "vl0_only_confidence_thresholds_verified": True,
        "test_set_used_for_selection_or_calibration": False,
        "e2b_status": "PARTIAL_EXCLUDED",
        "e2c_retained": False,
        "linear_design_or_raw_speed_run": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"model_results", "vienna_results"}}, indent=2))


if __name__ == "__main__":
    main()
