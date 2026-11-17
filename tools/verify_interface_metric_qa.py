import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
MODES = ("rinalmo", "h0", "handcrafted", "random", "length", "gc")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary_path = ROOT / "converged_baseline_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "PASS_SEQUENCE_ONLY_INTERFACE_PROBES_WITH_H2_ALIAS_AND_STRUCTURE_BLOCKED"
    differences = []
    n_runs = 0
    for mode in MODES:
        for seed in (42, 2026, 3407):
            directory = ROOT / "converged_baselines" / f"{mode}_seed{seed}"
            result = json.loads((directory / "result.json").read_text())
            assert result["converged"] and result["stop_reason"] == "internal_validation_plateau_30_epochs"
            assert result["checkpoint_sha256"] == sha256(directory / "best.pt")
            with np.load(directory / "holdout_predictions.npz", allow_pickle=False) as saved:
                probabilities = saved["probabilities"]
                labels = saved["labels"]
                lengths = saved["lengths"]
            prediction = probabilities >= result["threshold"]
            recomputed = {"micro_auprc": float(average_precision_score(labels, probabilities)),
                          "macro_f1_binary": float(f1_score(labels, prediction, average="macro")),
                          "micro_f1": float(f1_score(labels, prediction, average="binary")),
                          "interface_residue_recall": float((prediction & (labels == 1)).sum() / max(1, (labels == 1).sum()))}
            per_chain = []
            offset = 0
            for length in lengths:
                length = int(length)
                per_chain.append(float(f1_score(labels[offset:offset + length], prediction[offset:offset + length], average="binary")))
                offset += length
            recomputed["macro_f1_per_chain"] = float(np.mean(per_chain))
            for key, value in recomputed.items():
                difference = abs(value - result[key])
                assert difference < 2e-8, (mode, seed, key, value, result[key], difference)
                differences.append(difference)
            n_runs += 1
    output = {"status": "PASS_INDEPENDENT_SKLEARN_INTERFACE_METRIC_QA",
              "scientific_scope": "third-party metric recomputation only; not model selection or complete F2",
              "n_runs": n_runs, "max_abs_metric_difference": max(differences),
              "sklearn_version": __import__("sklearn").__version__,
              "summary_sha256": sha256(summary_path), "source_sha256": sha256(__file__)}
    (ROOT / "independent_metric_qa.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
