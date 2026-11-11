import argparse
import hashlib
import json
import re
from pathlib import Path

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
RUN = ROOT / "runs/round54_d1_s42"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    meta = json.loads((RUN / "run_meta.json").read_text())
    assert meta["arm"] == "round54_d1_s42" and meta["seed"] == 42
    assert meta["status"] == "running" and not meta["converged"]
    assert meta["best_step"] == 1500 and meta["best_validation_loss"] == 20.329000857636686
    assert meta["schedule"]["batch_size"] == 4 and meta["schedule"]["steps"] == 0
    assert meta["schedule"]["serial_encoder"] is False
    assert meta["scorer"] == "resnet2d" and meta["objective"]["lambda_jepa"] == 0.0
    resume = stable_load(RUN / "resume.pt")
    assert resume["step"] == 1500 and resume["stream_state"]
    attempts = {}
    for name in ("d1_s42.log", "d1_s42_resume1.log"):
        path = ROOT / "logs/round54" / name
        text = path.read_text()
        assert text.count("torch.OutOfMemoryError: CUDA out of memory") == 1
        steps = re.findall(r"\[plan-a\] step (\d+)/0", text)
        assert steps[-1] == "1900"
        attempts[name] = {"sha256": sha256_file(path), "last_logged_step": 1900,
                          "oom": re.search(r"torch\.OutOfMemoryError: CUDA out of memory\..*", text).group(0),
                          "resumed_from_1500": "resumed from step 1500" in text}
    assert attempts["d1_s42.log"]["resumed_from_1500"] is False
    assert attempts["d1_s42_resume1.log"]["resumed_from_1500"] is True
    ledger = read_jsonl(RUN / "compute_ledger.jsonl")
    first = [row for row in ledger if row["step"] <= 1500]
    repeated = ledger[1500:]
    assert len(first) == 1500 and repeated[0]["step"] == 1501
    assert first + repeated == ledger
    assert all(len(row["sequence_sha256"]) == 4 for row in ledger)
    report = {"status": "STOP_D1_REPEATED_LONG_BATCH_OOM", "run": meta["arm"],
              "best_step": 1500, "best_vl0": meta["best_validation_loss"],
              "last_successful_update": 1916, "failed_update": 1917,
              "failure_location": "ResNet2D BatchNorm backward recomputation",
              "attempts": attempts, "resume_step": resume["step"],
              "resume_sha256": sha256_file(RUN / "resume.pt"),
              "best_sha256": sha256_file(RUN / "best.pt"),
              "run_meta_sha256": sha256_file(RUN / "run_meta.json"),
              "ledger_sha256": sha256_file(RUN / "compute_ledger.jsonl"),
              "original_ledger_steps": 1916,
              "replayed_ledger_steps_after_resume": len(repeated),
              "configuration_changed_for_resume": False,
              "further_resume_allowed": False, "remaining_d1_seeds_started": False,
              "d1_effect_conclusion": False,
              "decision": "Retain B0; no batch reduction, chain skipping, precision change or further retry."}
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in
                      ("status", "last_successful_update", "failed_update", "replayed_ledger_steps_after_resume")}, indent=2))


if __name__ == "__main__":
    main()
