import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
MODES = ("rinalmo", "h0", "h2", "h0_structure", "h2_structure", "handcrafted", "random", "length", "gc")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    qa = json.loads((ROOT / "label_qa_summary.json").read_text())
    assert qa["status"] == "PASS_INDEPENDENT_INTERFACE_LABEL_QA"
    assert not qa["train_holdout_group_overlap"] and not qa["train_holdout_pdb_overlap"]
    labels = {r["name"]: r for r in map(json.loads, (ROOT / "holdout_contacts.jsonl").open())}
    gate = json.loads((ROOT / "structure_feature_gate.json").read_text())
    assert gate["status"] == "BLOCKED_STRUCTURE_FEATURE_FULL_LENGTH_CUDA_OOM"
    assert not gate["structure_arms_started"]
    summary = {"status": "PASS_SEQUENCE_ONLY_INTERFACE_PROBES_WITH_STRUCTURE_BLOCKED",
               "scientific_scope": "sequence-only frozen representation probes with verified contacts; structure-aware arms remain engineering-blocked; not complete F2 or a secondary-structure accuracy rescue; RiNALMo pretraining exposure unknown",
               "label_qa_sha256": sha256(ROOT / "label_qa_summary.json"),
               "structure_feature_gate_sha256": sha256(ROOT / "structure_feature_gate.json"), "modes": {}}
    for mode in MODES:
        if mode.endswith("_structure"):
            summary["modes"][mode] = {"status": gate["status"], "evidence_sha256": gate["log_sha256"],
                                      "not_a_model_efficacy_result": True}
            continue
        rows = []
        for seed in (42, 2026, 3407):
            directory = ROOT / "converged_baselines" / f"{mode}_seed{seed}"
            result = json.loads((directory / "result.json").read_text())
            assert result["mode"] == mode and result["seed"] == seed
            assert result["converged"] and result["stop_reason"] == "internal_validation_plateau_30_epochs"
            assert result["label_qa_sha256"] == sha256(ROOT / "label_qa_summary.json")
            assert not result["inference_structure_labels_used"]
            assert result["checkpoint_sha256"] == sha256(directory / "best.pt")
            with np.load(directory / "holdout_predictions.npz", allow_pickle=False) as saved:
                names = saved["names"].tolist()
                assert len(names) == len(labels) and set(names) == set(labels)
                lengths = saved["lengths"]
                assert lengths.sum() == len(saved["labels"]) == len(saved["probabilities"])
                offset = 0
                probabilities = []
                expected_labels = []
                per_chain_f1 = []
                per_chain_macro = []
                for name, length in zip(names, lengths):
                    length = int(length)
                    p = saved["probabilities"][offset:offset + length]
                    y = saved["labels"][offset:offset + length]
                    truth = np.asarray([r["interface"] for r in labels[name]["residues"]], dtype=np.float64)
                    assert np.array_equal(y, truth)
                    assert len(p) == len(y) and np.isfinite(p).all()
                    probabilities.extend(p)
                    expected_labels.extend(y)
                    prediction = p >= result["threshold"]
                    tp = int(np.sum(prediction & (y == 1)))
                    fp = int(np.sum(prediction & (y == 0)))
                    fn = int(np.sum(~prediction & (y == 1)))
                    tn = int(np.sum(~prediction & (y == 0)))
                    pf1 = f1(tp, fp, fn)
                    nf1 = f1(tn, fn, fp)
                    per_chain_f1.append(pf1)
                    per_chain_macro.append((pf1 + nf1) / 2)
                    offset += length
                probabilities = np.asarray(probabilities)
                expected_labels = np.asarray(expected_labels)
                prediction = probabilities >= result["threshold"]
                tp = int(np.sum(prediction & (expected_labels == 1)))
                fp = int(np.sum(prediction & (expected_labels == 0)))
                fn = int(np.sum(~prediction & (expected_labels == 1)))
                tn = int(np.sum(~prediction & (expected_labels == 0)))
                positive_f1 = f1(tp, fp, fn)
                negative_f1 = f1(tn, fn, fp)
                order = np.argsort(-probabilities, kind="stable")
                positives = expected_labels[order]
                auprc = float(np.sum(np.cumsum(positives) / np.arange(1, len(positives) + 1) * positives) / positives.sum())
                assert abs(auprc - result["micro_auprc"]) < 1e-12
                assert abs(positive_f1 - result["micro_f1"]) < 1e-12
                assert abs((positive_f1 + negative_f1) / 2 - result["macro_f1_binary"]) < 1e-12
                assert abs(np.mean(per_chain_f1) - result["macro_f1_per_chain"]) < 1e-12
                assert abs(np.mean(per_chain_macro) - result["macro_f1_binary_per_chain"]) < 1e-12
                assert (tp, fp, fn, tn) == tuple(result[key] for key in ("tp", "fp", "fn", "tn"))
            rows.append({"seed": seed, "micro_auprc": result["micro_auprc"],
                         "macro_f1_binary": result["macro_f1_binary"],
                         "macro_f1_per_chain": result["macro_f1_per_chain"],
                         "micro_f1": result["micro_f1"], "threshold": result["threshold"],
                         "best_epoch": result["best_epoch"], "epochs": result["epochs"],
                         "checkpoint_sha256": result["checkpoint_sha256"],
                         "predictions_sha256": sha256(directory / "holdout_predictions.npz"),
                         "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"]})
        entry = {"seeds": rows}
        for metric in ("micro_auprc", "macro_f1_binary", "macro_f1_per_chain", "micro_f1"):
            values = [row[metric] for row in rows]
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_sd"] = float(np.std(values))
        summary["modes"][mode] = entry
    summary["descriptive_effects"] = {
        "h2_minus_h0_auprc": summary["modes"]["h2"]["micro_auprc_mean"] - summary["modes"]["h0"]["micro_auprc_mean"],
        "h2_minus_h0_macro_f1_binary": summary["modes"]["h2"]["macro_f1_binary_mean"] - summary["modes"]["h0"]["macro_f1_binary_mean"],
    }
    output = ROOT / "converged_baseline_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "sha256": sha256(output), "effects": summary["descriptive_effects"]}, indent=2))


if __name__ == "__main__":
    main()
