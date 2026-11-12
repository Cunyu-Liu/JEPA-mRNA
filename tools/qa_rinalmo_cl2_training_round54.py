import argparse
import json
import math
import statistics
from pathlib import Path

from qa_rinalmo_b0_training_round54 import read_jsonl, sha256_file, validate_run

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
CL0 = ROOT / "data/cl0_structure_contrast_round54/clusters.jsonl"


def training(seed):
    run = ROOT / "runs" / f"round54_cl2_s{seed}"
    report = validate_run(run, seed, sha256_file(DATA), arm="cl2")
    meta = json.loads((run / "run_meta.json").read_text())
    baseline = json.loads((ROOT / "runs" / f"round54_b0_s{seed}/run_meta.json").read_text())
    assert meta["schedule"] == baseline["schedule"] and meta["validation"] == baseline["validation"]
    assert meta["data"] == baseline["data"] == str(DATA)
    assert meta["objective"]["cl1_token_mask"] == 0.15
    assert meta["objective"]["cl2_infonce_weight"] == 0.1
    assert meta["objective"]["cl2_temperature"] == 0.1
    assert meta["objective"]["cl2_max_negatives"] == 32
    assert meta["objective"]["cl0_clusters"] == str(CL0)
    logs = read_jsonl(run / "train_log.jsonl")
    assert all(row["cl1_views"] == 2 and len(row["cl1_masked_tokens"]) == 2 for row in logs)
    assert all(row["cl2_projection_grad"] > 0 for row in logs)
    assert all(set(row["terms"]) == {"nll", "distill", "rlcd", "cal", "infonce"} for row in logs)
    diagnostics = read_jsonl(run / "cl2_diagnostics.jsonl")
    legal = {row["sequence_sha256"]: row["legal_negative"] for row in read_jsonl(CL0)
             if row["split"] == "train"}
    for row in diagnostics:
        assert row["negative_count"] > 0 and row["representation_norm"] > 0
        assert -1 <= row["positive_similarity"] <= 1
        assert -1 <= row["negative_similarity"] * meta["objective"]["cl2_temperature"] <= 1
        assert row["channel_variance"] >= 0 and math.isfinite(row["effective_rank"])
        assert all(legal[sequence] for sequence in row["negative_sequence_sha256"])
    from qa_loss_optimizer_training_round51 import stable_load
    state = stable_load(run / "resume.pt")
    best = stable_load(run / "best.pt")
    assert best["cl2"]["model"]
    if "cl2" in state:
        assert state["cl2"]["queue_z"] is not None
    assert not any("projection" in key for key in best["model"]["head"])
    report.update({"status": "PASS_CL2_SINGLE_SEED_TRAINING_QA", "logged_steps": len(logs),
                   "diagnostic_steps": len(diagnostics), "deployment_projection_head": False,
                   "resume_projection_state": "PRESENT" if "cl2" in state else "PARTIAL_MISSING_FINAL_PROJECTION",
                   "effective_rank_range": [min(row["effective_rank"] for row in diagnostics),
                                            max(row["effective_rank"] for row in diagnostics)],
                   "positive_similarity_range": [min(row["positive_similarity"] for row in diagnostics),
                                                 max(row["positive_similarity"] for row in diagnostics)],
                   "representation_norm_range": [min(row["representation_norm"] for row in diagnostics),
                                                 max(row["representation_norm"] for row in diagnostics)],
                   "scientific_conclusion": False})
    return report


def summary():
    runs = {}
    for seed in (42, 2026, 3407):
        report = json.loads((ROOT / f"audits/round54/cl2_s{seed}_training_qa.json").read_text())
        assert report["status"] == "PASS_CL2_SINGLE_SEED_TRAINING_QA"
        runs[str(seed)] = report
    values = [row["best_vl0"] for row in runs.values()]
    b0 = json.loads((ROOT / "audits/round54/b0_training_qa.json").read_text())["runs"]
    cl1 = json.loads((ROOT / "audits/round54/cl1_training_summary.json").read_text())["runs"]
    return {"status": "PASS_CL2_THREE_SEED_DESCRIPTIVE_QA", "runs": runs,
            "mean_best_vl0": statistics.mean(values), "population_sd_best_vl0": statistics.pstdev(values),
            "paired_delta_b0": {seed: runs[seed]["best_vl0"] - b0[seed]["best_vl0"] for seed in runs},
            "paired_delta_cl1": {seed: runs[seed]["best_vl0"] - cl1[seed]["best_vl0"] for seed in runs},
            "test_set_used": False, "scientific_conclusion": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summary() if args.summary else training(args.seed)
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
