import hashlib
import json
from pathlib import Path

import torch

from train_interface_baseline import Probe, load_split, metrics, raw_predictions, sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
MODES = ("rinalmo", "h0", "handcrafted", "random", "length", "gc")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen inference")
    for mode in MODES:
        for seed in (42, 2026, 3407):
            directory = ROOT / "converged_baselines" / f"{mode}_seed{seed}"
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text())
            legacy = directory / "result_before_auprc_tie_fix.json"
            if not legacy.exists():
                legacy.write_text(json.dumps(result, indent=2) + "\n")
            training = load_split("train", mode)
            validation_groups = {group for i, group in enumerate(sorted({row["group"] for row in training})) if i % 5 == 0}
            validation = [row for row in training if row["group"] in validation_groups]
            holdout = load_split("holdout", mode)
            use_structure = mode.endswith("_structure")
            dimension = validation[0]["x"].shape[1] + (validation[0]["structure"].shape[1] if use_structure else 0)
            model = Probe(dimension).cuda()
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            with torch.no_grad():
                validation_predictions = raw_predictions(validation, model, use_structure)
                holdout_predictions = raw_predictions(holdout, model, use_structure)
            validation_metrics = metrics(validation_predictions, result["threshold"])
            holdout_metrics = metrics(holdout_predictions, result["threshold"])
            assert validation_metrics["threshold"] == holdout_metrics["threshold"] == result["threshold"]
            for key in ("macro_f1_binary", "micro_f1", "macro_f1_per_chain", "macro_f1_binary_per_chain",
                        "precision", "interface_residue_recall", "tp", "fp", "fn", "tn"):
                assert holdout_metrics[key] == result[key], (mode, seed, key, holdout_metrics[key], result[key])
            for key in ("micro_auprc", "macro_auprc_positive_chains", "n_auprc_chains"):
                result[key] = holdout_metrics[key]
            result["validation_metrics_at_threshold"] = validation_metrics
            result["auprc_tie_policy"] = "distinct score thresholds, sklearn average_precision_score compatible"
            result["legacy_result_before_auprc_tie_fix"] = str(legacy)
            result["metric_revision_source_sha256"] = sha256(Path(__file__).with_name("train_interface_baseline.py"))
            result["metric_recompute_source_sha256"] = sha256(__file__)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(mode, seed, result["micro_auprc"], flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
