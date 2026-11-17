import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
MODES = ("rinalmo", "h0", "handcrafted", "random", "length", "gc")


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def metrics(probabilities, labels, threshold):
    order = np.argsort(-probabilities, kind="stable")
    distinct = np.where(np.diff(probabilities[order]))[0]
    indices = np.r_[distinct, len(labels) - 1]
    true_positives = np.cumsum(labels[order])[indices]
    precision = true_positives / (indices + 1)
    previous = np.r_[0, true_positives[:-1]]
    auprc = float(np.sum((true_positives - previous) * precision / max(1, labels.sum())))
    prediction = probabilities >= threshold
    tp = int(np.sum(prediction & (labels == 1)))
    fp = int(np.sum(prediction & (labels == 0)))
    fn = int(np.sum(~prediction & (labels == 1)))
    tn = int(np.sum(~prediction & (labels == 0)))
    return {"micro_auprc": auprc, "macro_f1_binary": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
            "interface_residue_recall": tp / max(1, tp + fn), "n_residues": len(labels),
            "n_positive": int(labels.sum()), "positive_prevalence": float(labels.mean())}


def bucket(length):
    return "<100" if length < 100 else "100-499" if length < 500 else "500-1499" if length < 1500 else ">=1500"


def main():
    summary_path = ROOT / "converged_baseline_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "PASS_SEQUENCE_ONLY_INTERFACE_PROBES_WITH_H2_ALIAS_AND_STRUCTURE_BLOCKED"
    labels = {r["name"]: r for r in map(json.loads, (ROOT / "holdout_contacts.jsonl").open())}
    output = {"status": "PASS_DESCRIPTIVE_INTERFACE_STRATA", "scientific_scope": "frozen holdout strata only; not model selection; structure-aware arms remain blocked",
              "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "modes": {}}
    for mode in MODES:
        seed_rows = []
        for seed in (42, 2026, 3407):
            directory = ROOT / "converged_baselines" / f"{mode}_seed{seed}"
            result = json.loads((directory / "result.json").read_text())
            with np.load(directory / "holdout_predictions.npz", allow_pickle=False) as saved:
                probabilities = saved["probabilities"]
                truth = saved["labels"]
                names = saved["names"]
                lengths = saved["lengths"]
                groups = {"all": np.ones(len(names), dtype=bool)}
                for name in sorted({bucket(len(record["sequence"])) for record in labels.values()}):
                    groups[f"length:{name}"] = np.asarray([bucket(len(labels[n]["sequence"])) == name for n in names])
                methods = sorted({method for record in labels.values() for method in record["experimental_methods"]})
                for method in methods:
                    groups[f"method:{method}"] = np.asarray([method in labels[n]["experimental_methods"] for n in names])
                row = {}
                for group, selected in groups.items():
                    indices = np.concatenate([np.arange(sum(lengths[:i]), sum(lengths[:i + 1]))
                                              for i, include in enumerate(selected) if include])
                    if not len(indices):
                        row[group] = None
                        continue
                    row[group] = {**metrics(probabilities[indices], truth[indices], result["threshold"]),
                                  "n_chains": int(selected.sum())}
                seed_rows.append({"seed": seed, "strata": row})
        entry = {"seeds": seed_rows, "mean": {}}
        for group in seed_rows[0]["strata"]:
            if any(row["strata"][group] is None for row in seed_rows):
                entry["mean"][group] = None
                continue
            entry["mean"][group] = {metric: float(np.mean([row["strata"][group][metric] for row in seed_rows]))
                for metric in ("micro_auprc", "macro_f1_binary", "interface_residue_recall", "positive_prevalence")}
            entry["mean"][group]["n_chains"] = seed_rows[0]["strata"][group]["n_chains"]
        output["modes"][mode] = entry
    path = ROOT / "interface_stratified_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
