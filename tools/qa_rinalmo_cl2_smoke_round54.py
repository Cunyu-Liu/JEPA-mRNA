import argparse
import hashlib
import json
from pathlib import Path

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
CL0 = ROOT / "data/cl0_structure_contrast_round54/clusters.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["arm"] == "round54_cl2_smoke_s42" and meta["seed"] == 42
    assert meta["status"] == "completed" and meta["steps_completed"] == 2
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["scorer"] == "resnet2d" and meta["ripe"]["mode"] is None
    assert meta["objective"]["cl1_token_mask"] == 0.15
    assert meta["objective"]["cl2_infonce_weight"] == 0.1
    assert meta["objective"]["cl2_temperature"] == 0.1
    assert meta["objective"]["cl2_max_negatives"] == 32
    assert meta["objective"]["cl0_clusters"] == str(CL0)
    logs = read_jsonl(run / "train_log.jsonl")
    assert [row["step"] for row in logs] == [1]
    assert logs[0]["cl1_views"] == 2 and logs[0]["cl2_projection_grad"] > 0
    assert set(logs[0]["terms"]) == {"nll", "distill", "rlcd", "cal", "infonce"}
    diagnostics = read_jsonl(run / "cl2_diagnostics.jsonl")
    assert len(diagnostics) == 1 and diagnostics[0]["negative_count"] > 0
    legal = {row["sequence_sha256"]: row["legal_negative"] for row in read_jsonl(CL0)
             if row["split"] == "train"}
    for sequence in diagnostics[0]["negative_sequence_sha256"]:
        assert legal[sequence]
    resume = stable_load(run / "resume.pt")
    assert resume["step"] == 2 and resume["cl2"]["queue_z"] is not None
    best = stable_load(run / "best.pt")
    assert best["cl2"]["model"] and not any("projection" in key for key in best["model"]["head"])
    report = {"status": "PASS_CL2_INFO_NCE_CUDA_ENGINEERING", "run": meta["arm"],
              "train_log": logs, "diagnostics": diagnostics,
              "run_meta_sha256": sha256_file(run / "run_meta.json"),
              "resume_sha256": sha256_file(run / "resume.pt"),
              "cl0_sha256": sha256_file(CL0), "deployment_projection_head": False,
              "scientific_conclusion": False}
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "negative_count": diagnostics[0]["negative_count"],
                      "projection_grad": logs[0]["cl2_projection_grad"]}, indent=2))


if __name__ == "__main__":
    main()
