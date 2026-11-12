import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from eval.ss.metrics import PairLevelMetrics
from rnajepa.harness import valid_pair_mask

SPLITS = ("bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root, arm, seed, split, ablation):
    mode = "nussinov" if ablation != "greedy" else "greedy"
    experiment = f"prism_{arm}{ablation}"
    path = root / f"eval_decision/{experiment}/s{seed}/{split}.json"
    result = json.loads(path.read_text())
    checkpoint = root / f"runs/round54_{arm}_s{seed}/best.pt"
    data = root / f"data/structrfm_pretrain_clean/{split}.jsonl"
    assert result["device"] == "cuda" and result["repeats"] == 3
    assert result["score_forward_count_per_sequence"] == 1
    assert result["checkpoint_sha256"] == sha256(checkpoint)
    assert result["data_sha256"] == sha256(data)
    assert result["decoder"] == mode
    assert result["prior_weight_override"] == (0.0 if ablation == "_turner0" else None)
    meta = json.loads((checkpoint.parent / "run_meta.json").read_text())
    assert meta["arm"] == f"round54_{arm}_s{seed}" and meta["converged"] and meta["early_stopped"]
    assert meta["best_step"] == result["checkpoint_step"]
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert result["raw_n"] == len(rows)
    assert result["n"] + result["n_long_unsupported"] == len(rows)
    assert result["n_long_unsupported"] == 0 and result["status"] == "PASS"
    tp = fp = fn = 0
    for index, (source, row) in enumerate(zip(rows, result["per_sequence"])):
        sequence = source["seq"].upper().replace("T", "U")
        expected = [tuple(pair) for pair in source["pairs"]]
        assert row["name"] == source.get("name", "") and row["length"] == len(sequence)
        repeats = []
        for prediction in result["predictions"]:
            pairs = [tuple(pair) for pair in prediction[index]["pred_pairs"]]
            repeats.append(pairs)
            assert all(mask[i, j] for i, j in pairs for mask in [valid_pair_mask(sequence)])
        assert repeats[0] == repeats[1] == repeats[2]
        metrics = PairLevelMetrics.from_pairs(repeats[0], expected, L=len(sequence), mask=valid_pair_mask(sequence))
        assert (row["tp"], row["fp"], row["fn"]) == (metrics.tp, metrics.fp, metrics.fn)
        assert math.isclose(row["f1"], metrics.f1, abs_tol=1e-12)
        tp += metrics.tp
        fp += metrics.fp
        fn += metrics.fn
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    assert math.isclose(micro_f1, result["micro"]["f1"], abs_tol=1e-12)
    macro = statistics.mean(row["f1"] for row in result["per_sequence"])
    assert math.isclose(macro, result["macro_f1"], abs_tol=1e-12)
    return {"micro_f1": micro_f1, "macro_f1": macro, "tp": tp, "fp": fp, "fn": fn,
            "sha256": sha256(path), "n": result["n"], "complete": True}


def holm(values):
    ordered = sorted(values.items(), key=lambda row: row[1])
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def bootstrap(baseline, candidate, seed):
    names = sorted(set(baseline) & set(candidate))
    assert names
    left = np.array([baseline[name]["f1"] for name in names])
    right = np.array([candidate[name]["f1"] for name in names])
    rng = np.random.default_rng(seed)
    differences = right - left
    draws = []
    for _ in range(2000):
        indices = rng.integers(0, len(names), len(names))
        draws.append(differences[indices].mean())
    probability = (1 + min(np.mean(np.asarray(draws) <= 0), np.mean(np.asarray(draws) >= 0))) * 2
    return {"n": len(names), "mean_delta": float(differences.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "p_two_sided": float(min(1.0, probability))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    arms = (root / "audits/round54/frozen_arms.txt").read_text().split()
    assert arms and "b0" in arms and set(arms) <= {"b0", "cl1", "e2b"}
    assert not (root / "audits/round54/e2c_vl0_decision.json").read_text().endswith('"retain_physics_dp": true\n}')
    results = {}
    for arm in arms:
        for ablation in ("", "_greedy", "_turner0"):
            for seed in (42, 2026, 3407):
                for split in SPLITS:
                    key = f"{arm}{ablation}/{seed}/{split}"
                    results[key] = validate(root, arm, seed, split, ablation)
        confidence = json.loads((root / f"eval_decision/prism_{arm}/confidence_summary.json").read_text())
        assert confidence["status"] == "PASS_STRUCTRFM_CONFIDENCE_VL0_FROZEN_THRESHOLDS"
        assert set(confidence["splits"]) == set(SPLITS)
        for split in SPLITS:
            path = root / f"eval_decision/prism_{arm}/s42/{split}.json"
            result = json.loads(path.read_text())
            assert result["repeats"] == 3 and result["score_forward_count_per_sequence"] == 1
    paired = {}
    for arm in arms:
        if arm == "b0":
            continue
        paired[arm] = {}
        for split in SPLITS:
            baseline = json.loads((root / f"eval_decision/prism_b0/s42/{split}.json").read_text())["per_sequence"]
            candidate = json.loads((root / f"eval_decision/prism_{arm}/s42/{split}.json").read_text())["per_sequence"]
            baseline = {row["name"]: row for row in baseline}
            candidate = {row["name"]: row for row in candidate}
            paired[arm][split] = bootstrap(baseline, candidate, 42)
    report = {"status": "PASS_PRISM_FROZEN_BENCHMARK_QA", "arms": arms,
              "n_cells": len(results), "results": results, "paired_bootstrap": paired,
              "holm_main_eight": {arm: holm({split: value["p_two_sided"] for split, value in cells.items() if split != "rfam_fam"})
                                  for arm, cells in paired.items()},
              "confidence": {arm: sha256(root / f"eval_decision/prism_{arm}/confidence_summary.json") for arm in arms},
              "sota_comparison": {arm: json.loads((root / f"eval_decision/prism_{arm}/sota_comparison.json").read_text()) for arm in arms},
              "test_set_used_for_selection_or_calibration": False,
              "linear_design_or_raw_speed_run": False,
              "limitations": ["Descriptive frozen evaluation, not a guarantee of SOTA or best confidence.",
                              "Rfam has RiNALMo pretraining-contact risk and is reported separately.",
                              "No unreviewed combination of independent modules is included."]}
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": report["status"], "arms": arms, "n_cells": len(results),
                      "main_sota_wins": {arm: report["sota_comparison"][arm]["main_sota_any_wins"] for arm in arms}}, indent=2))


if __name__ == "__main__":
    main()
