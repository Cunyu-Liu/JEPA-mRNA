#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from eval.ss.metrics import PairLevelMetrics
from prepare_lineardesign_round49 import atomic_json, dbn_pairs, sha256


def summarize(predictions, rows):
    if len(predictions) != len(rows) or not rows:
        raise ValueError("predictions and candidates must have the same nonzero length")
    metrics = []
    tp = fp = fn = 0
    for prediction, row in zip(predictions, rows):
        result = PairLevelMetrics.from_pairs([tuple(pair) for pair in prediction["pred_pairs"]],
                                             [tuple(pair) for pair in row["pairs"]], L=len(row["seq"]))
        tp += result.tp
        fp += result.fp
        fn += result.fn
        metrics.append({"name": row["name"], "f1_vs_lineardesign_mfe": result.f1,
                        "precision_vs_lineardesign_mfe": result.precision,
                        "recall_vs_lineardesign_mfe": result.recall,
                        "n_predicted_pairs": len(prediction["pred_pairs"]),
                        "n_lineardesign_pairs": len(row["pairs"])})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"per_candidate": metrics, "micro_f1_vs_lineardesign_mfe": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "macro_f1_vs_lineardesign_mfe": sum(row["f1_vs_lineardesign_mfe"] for row in metrics) / len(metrics)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.root / "eval_decision/lineardesign_round49/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "PASS_LINEARDESIGN_FROZEN_CANDIDATES"
    data_path = manifest_path.parent / "candidates.jsonl"
    assert sha256(data_path) == manifest["candidate_data_sha256"]
    rows = [json.loads(line) for line in data_path.open() if line.strip()]
    assert len(rows) == manifest["n"] == 3
    assert all([list(pair) for pair in dbn_pairs(row["dbn"])] == row["pairs"] for row in rows)
    outputs = {}
    sources = {str(manifest_path.relative_to(args.root)): sha256(manifest_path),
               str(data_path.relative_to(args.root)): sha256(data_path)}
    for model in manifest["models"]:
        path = args.root / f"eval_decision/lineardesign_round49/{model['experiment']}.json"
        result = json.loads(path.read_text())
        sources[str(path.relative_to(args.root))] = sha256(path)
        assert result["status"] == "PASS" and result["device"] == "cuda"
        assert result["raw_n"] == result["n"] == len(rows) and result["repeats"] == 3
        assert result["data_sha256"] == manifest["candidate_data_sha256"]
        assert result["checkpoint_sha256"] == model["checkpoint_sha256"]
        assert result["checkpoint_step"] == model["best_step"]
        assert result["score_forward_count_per_sequence"] == 1
        assert result["prior_weight_override"] is None and result["decoder"] == "nussinov"
        assert all(repeat[index]["pred_pairs"] == result["predictions"][0][index]["pred_pairs"]
                   for repeat in result["predictions"] for index in range(len(rows)))
        outputs[model["experiment"]] = {"seed": model["seed"], "best_step": model["best_step"],
                                       "best_validation_loss": model["best_validation_loss"],
                                       **summarize(result["predictions"][0], rows),
                                       "wall_seconds": result["wall_seconds"]}
    output = {"status": "PASS_LINEARDESIGN_MODEL_EVALUATION", "n_candidates": len(rows),
              "scope": "agreement of frozen secondary-structure models with LinearDesign MFE structures on three official coding examples",
              "selection_rule": manifest["selection_rule"], "models": outputs,
              "mfe_discrepancy_note": "LinearDesign-reported MFE is retained separately from independent ViennaRNA 2.7.2 MFE; equality is not claimed",
              "candidates": [{key: row[key] for key in ["name", "protein", "seq", "dbn", "design_lambda",
                              "official_reported_mfe_kcal_mol", "independent_viennarna_mfe_kcal_mol",
                              "official_reported_cai", "independent_cai", "gc_fraction"]} for row in rows],
              "wet_lab_validation": False, "designed_structure_ground_truth": False,
              "expression_stability_or_immunogenicity_claim": False,
              "general_noncoding_design_claim": False, "source_hashes": sources}
    atomic_json(args.out, output)
    assert json.loads(args.out.read_text()) == output
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
