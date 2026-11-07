import argparse
import hashlib
import json
import statistics
from pathlib import Path


ARMS = (("structrfm_full", "structrfm_jev_s"),
        ("structrfm_s3", "structrfm_s3_pairmlp490_s"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root):
    root = Path(root)
    decision_path = root / "audits/round48/route1_decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["complete"] and decision["allow_ripe"]
    for relative, expected in decision["source_hashes"].items():
        assert sha256(root / relative) == expected, relative
    gate_path = root / "audits/round48/ripe_cuda_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "PASS_RIPE_CUDA_SYNTHETIC"
    assert gate["device"] == "cuda" and set(gate["lengths"]) >= {515, 1024}
    candidates = []
    for experiment, prefix in ARMS:
        runs = []
        for seed in range(3):
            run_dir = root / "runs" / f"{prefix}{seed}"
            meta = json.loads((run_dir / "run_meta.json").read_text())
            assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
            assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
            assert meta["environment"]["resolved_device"].startswith("cuda")
            assert meta["ripe"]["mode"] is None and not meta["ripe"]["encoder"]
            checkpoint = run_dir / "best.pt"
            runs.append({"seed": seed, "best_step": meta["best_step"],
                         "best_validation_loss": meta["best_validation_loss"],
                         "checkpoint_sha256": sha256(checkpoint)})
        candidates.append({"experiment": experiment, "run_prefix": prefix,
                           "mean_best_vl0": statistics.fmean(r["best_validation_loss"] for r in runs),
                           "runs": runs, "meta": meta})
    selected = min(candidates, key=lambda row: (row["mean_best_vl0"], row["experiment"]))
    meta = selected["meta"]
    modes = ["log", "r1", "r2", "r3", "r5", "r6"]
    arms = [{"arm": f"ripe_head_{mode}", "mode": mode, "encoder": False} for mode in modes]
    arms.append({"arm": "ripe_encoder_r3", "mode": "r3", "encoder": True})
    return {"status": "PASS_RIPE_PREREGISTERED_QUEUE",
            "selection_rule": "minimum three-seed mean of best VL0 losses among completed S2/S3; no test metric",
            "selected": selected, "candidates": candidates,
            "route1_decision_sha256": sha256(decision_path),
            "cuda_gate_sha256": sha256(gate_path),
            "training": {"data": "TR1c", "validation": "VL0_196", "steps": 0,
                         "batch_size": 2, "patience": 6, "min_delta": 1e-6,
                         "seeds": [0, 1, 2], "head_lr": 1e-4, "backbone_lr": 1e-5,
                         "scorer": meta["scorer"], "pair_latent_dim": meta["pair_latent_dim"]},
            "arms": arms, "repeats": 3, "splits": 13,
            "capacity_control": "the opposite completed S2/S3 arm; compare PairMLP-490 (608817) with RIPE+ResNet2D (608759), delta 58 parameters",
            "qk_norm_changed": False,
            "attention_precision": "RIPE bias cast to the original attention-score dtype; no FP32 softmax promotion",
            "long_sequence_boundary": "head-only arms retain the official 512nt limit; encoder-only R3 is the independent extrapolation arm",
            "next": "three-seed natural convergence then A/B/C; no test-based arm selection"}


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
                      "selected": result["selected"]["experiment"],
                      "mean_best_vl0": result["selected"]["mean_best_vl0"]}, indent=2))


if __name__ == "__main__":
    main()
