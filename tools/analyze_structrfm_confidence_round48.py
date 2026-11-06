import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


COVERAGE = np.array([1.0, 0.95, 0.90, 0.75, 0.50, 0.25])
SPLITS = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam", "chanrg_test",
          "chanrg_gena", "chanrg_genc", "chanrg_genf"]


def risk_curve(confidence, f1, thresholds):
    risks = []
    for threshold in thresholds:
        keep = confidence >= threshold
        risks.append(1.0 - float(f1[keep].mean()) if keep.any() else float("nan"))
    risks = np.array(risks)
    aurc = float(np.trapezoid(risks[::-1], COVERAGE[::-1])) if np.isfinite(risks).all() else None
    return risks, aurc


def evaluate(rows, thresholds):
    confidence = np.array([row["structure_conf"] for row in rows], dtype=np.float64)
    f1 = np.array([row["f1"] for row in rows], dtype=np.float64)
    risks, aurc = risk_curve(confidence, f1, thresholds)
    correlation = float(spearmanr(confidence, f1).statistic)
    return {"n": len(rows), "spearman": correlation if np.isfinite(correlation) else None,
            "coverage": COVERAGE.tolist(), "risk": risks.tolist(), "aurc": aurc,
            "vl0_thresholds": np.asarray(thresholds).tolist()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--vl0-only", action="store_true")
    args = parser.parse_args()
    experiment = "structrfm_full" if args.full_model else "structrfm_frozen"
    run_prefix = "structrfm_jev_s" if args.full_model else "structrfm_frozen_jev_s"
    output = args.root / f"eval_decision/{experiment}"
    metadata = [json.loads((args.root / f"runs/{run_prefix}{seed}/run_meta.json").read_text())
                for seed in range(3)]
    reference_seed = int(np.argmin([meta["best_validation_loss"] for meta in metadata]))
    reference = json.loads((output / f"vl0_s{reference_seed}.json").read_text())["per_sequence"]
    thresholds = np.quantile([row["structure_conf"] for row in reference], 1 - COVERAGE)
    threshold_record = {"reference_seed": reference_seed,
                        "reference_best_validation_loss": metadata[reference_seed]["best_validation_loss"],
                        "coverage": COVERAGE.tolist(), "thresholds": thresholds.tolist(),
                        "protocol": "reference seed has minimum VL0 validation loss; thresholds use only its VL0 structure confidence; no test fitting"}
    write_json(output / "confidence_vl0_thresholds.json", threshold_record)
    summary = {"status": "PASS_STRUCTRFM_CONFIDENCE_VL0_FROZEN_THRESHOLDS",
               "experiment": experiment, "thresholds": threshold_record, "vl0": {}, "splits": {}}
    for seed in range(3):
        result = json.loads((output / f"vl0_s{seed}.json").read_text())
        assert result["repeats"] == 3
        summary["vl0"][f"s{seed}"] = evaluate(result["per_sequence"], thresholds)
    if not args.vl0_only:
        for split in SPLITS:
            cells = {}
            for seed in range(3):
                result = json.loads((output / f"s{seed}/{split}.json").read_text())
                assert result["repeats"] == 3
                cells[f"s{seed}"] = evaluate(result["per_sequence"], thresholds)
            summary["splits"][split] = cells
    write_json(output / "confidence_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
