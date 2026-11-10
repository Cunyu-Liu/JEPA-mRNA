import argparse
import hashlib
import json
from pathlib import Path

import torch

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    run = root / "runs" / f"round52_rinalmo_r0_s{args.seed}"
    meta = json.loads((run / "run_meta.json").read_text())
    validation = read_jsonl(run / "validation_log.jsonl")
    assert len(validation) >= 2
    assert [row["step"] for row in validation[:2]] == [500, 1000]
    assert all(row["bad_evaluations"] == 0 and row["improved"] for row in validation[:2])
    best_record = min(validation, key=lambda row: row["validation_loss"])
    best = stable_load(run / "best.pt")
    assert best["step"] == best_record["step"]
    assert best["validation_loss"] == best_record["validation_loss"]
    resume = stable_load(run / "resume.pt")
    assert resume["step"] >= best["step"] and resume["stream_state"]
    assert resume["scheduler"]["base_lrs"] == [1e-4, 1e-5]
    assert resume["optimizer"]["param_groups"][0]["lr"] == 1e-4
    assert resume["optimizer"]["param_groups"][1]["lr"] == 1e-5
    assert meta["arm"] == f"round52_rinalmo_r0_s{args.seed}"
    assert meta["schedule"]["steps"] == 0 and meta["schedule"]["batch_size"] == 2
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    assert meta["objective"]["nll_normalization"] == "sum"
    assert meta["backbone_report"]["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    assert meta["environment"]["resolved_device"].startswith("cuda")
    output = {
        "status": "PASS_R0_PROGRESS_THREE_CHECK_QA",
        "scientific_conclusion": False,
        "seed": args.seed,
        "best_step": best["step"],
        "best_vl0": best["validation_loss"],
        "resume_step": resume["step"],
        "validation_curve": validation,
        "run_status_at_read": meta["status"],
        "scheduler_base_lrs": resume["scheduler"]["base_lrs"],
        "best_sha256": sha256_file(run / "best.pt"),
        "resume_sha256": sha256_file(run / "resume.pt"),
        "run_meta_sha256": sha256_file(run / "run_meta.json"),
        "validation_log_sha256": sha256_file(run / "validation_log.jsonl"),
        "notes": [
            "Progress evidence only; training has not naturally converged.",
            "The current resume need not equal best before early stopping restores best weights.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
