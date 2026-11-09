import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    root = Path(root)
    decision_path = root / "audits/round48/ripe_decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["complete"] and decision["allow_pair_jepa"]
    for relative, expected in decision["source_hashes"].items():
        assert sha256(root / relative) == expected, relative
    manifest_path = root / "audits/round48/ripe_preregistered.json"
    ripe_manifest = json.loads(manifest_path.read_text())
    base = decision["next_base"]
    scorer = ripe_manifest["training"]["scorer"]
    latent_dim = 490
    reuse_mlp = scorer == "pairmlp" and ripe_manifest["training"]["pair_latent_dim"] == latent_dim
    arms = [
        {"arm": "pairjepa_mlp", "scorer": "pairmlp", "target": None, "lambda_jepa": 0.0,
         "reuse_baseline": reuse_mlp},
        {"arm": "pairjepa_ema", "scorer": "pairjepa", "target": "ema", "lambda_jepa": 0.1,
         "reuse_baseline": False},
        {"arm": "pairjepa_noema", "scorer": "pairjepa", "target": "fixed", "lambda_jepa": 0.1,
         "reuse_baseline": False},
        {"arm": "pairjepa_random", "scorer": "pairjepa", "target": "random", "lambda_jepa": 0.1,
         "reuse_baseline": False},
    ]
    return {"status": "PASS_PAIRJEPA_PREREGISTERED_QUEUE",
            "ripe_decision_sha256": sha256(decision_path),
            "ripe_manifest_sha256": sha256(manifest_path),
            "selection_rule": "route-two next_base selected only by three-seed mean best VL0; no test metric",
            "baseline": base,
            "training": {"data": "TR1c", "validation": "VL0_196", "steps": 0,
                         "batch_size": 2, "patience": 6, "min_delta": 1e-6,
                         "seeds": [0, 1, 2], "head_lr": 1e-4, "backbone_lr": 1e-5,
                         "pair_latent_dim": latent_dim, "jepa_mask_prob": 0.15,
                         "jepa_rank_samples": 512, "jepa_log_every": 25,
                         "jepa_objective_version": "context_v1",
                         "jepa_context": "training_only_normalized_3x3_excluding_center_and_masked_pairs",
                         "max_parallel_training": 2},
            "ripe_mode": base["ripe_mode"], "ripe_encoder": base["ripe_encoder"],
            "arms": arms, "repeats": 3, "splits": 13,
            "lambda_rule": "lambda_jepa=0.1 frozen before P3; no test-based tuning",
            "inference_rule": "one scorer forward; teacher forward replaced by fail-closed guard",
            "capacity_rule": "J-mlp is the no-auxiliary near-parameter control; if it is identical to J0, reuse J0 rather than retrain",
            "boundary": "engineering configuration only; no Pair-JEPA scientific claim before three-seed A/B/C and anti-collapse analysis"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "baseline": result["baseline"], "arms": result["arms"]}, indent=2))


if __name__ == "__main__":
    main()
