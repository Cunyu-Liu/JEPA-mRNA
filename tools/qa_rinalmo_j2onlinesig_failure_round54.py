import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run = ROOT / "runs/round54_j2onlinesig_s42"
    log = ROOT / "logs/round54/j2onlinesig_s42.log"
    text = log.read_text()
    assert "torch.OutOfMemoryError" in text and "input_projection" in text
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["objective"]["jepa_target"] == "online"
    rows = [json.loads(line) for line in (run / "jepa_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    assert rows[-1]["step"] == 200
    output = {
        "status": "STOP_NO_EMA_SHARED_ONLINE_TARGET_SIGREG_OOM",
        "scientific_conclusion": "The shared-online-target contract works, but OOM prevents anti-collapse or VL0 conclusions.",
        "allow_retry": False,
        "allow_three_seed": False,
        "allow_fixed_or_random": False,
        "last_step": rows[-1]["step"],
        "prediction_effective_rank": rows[-1]["prediction_effective_rank"],
        "prediction_cross_position_cosine": rows[-1]["prediction_cross_position_cosine"],
        "prediction_cross_sequence_cosine": rows[-1]["prediction_cross_sequence_cosine"],
        "log_sha256": sha256(log),
        "diagnostics_sha256": sha256(run / "jepa_diagnostics.jsonl"),
    }
    path = ROOT / "audits/round54/j2onlinesig_failure_qa.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
