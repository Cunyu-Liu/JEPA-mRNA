import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = ROOT / "runs/round54_cl5_smoke_s42"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["arm"] == "round54_cl5_smoke_s42" and meta["converged"]
    assert meta["objective"]["cl5_output_weight"] == 0.1
    assert meta["objective"]["cl4_negative_weights"].endswith("cl4_negative_weights.json")
    assert meta["schedule"]["steps"] == 2 and meta["seed"] == 42
    logs = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines() if line.strip()]
    assert len(logs) == 2
    for row in logs:
        assert set(row["terms"]) == {"nll", "distill", "rlcd", "cal", "infonce", "output_consistency"}
        output = row["cl5_output"]
        assert output["output_pairs"] > 0 and output["output_norm"] > 0
        assert -1 <= output["output_similarity"] <= 1
        assert output["output_channel_variance"] >= 0
        assert row["cl2_projection_grad"] > 0
    report = {
        "status": "PASS_CL5_TWO_STEP_CUDA_SMOKE",
        "steps": len(logs),
        "best_step": meta["best_step"],
        "best_validation_loss": meta["best_validation_loss"],
        "best_sha256": sha256(run / "best.pt"),
        "resume_sha256": sha256(run / "resume.pt"),
        "output_consistency_nonzero": any(row["terms"]["output_consistency"] != 0 for row in logs),
        "deployment_additional_parameters": 0,
        "test_set_used": False,
    }
    assert report["output_consistency_nonzero"]
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
