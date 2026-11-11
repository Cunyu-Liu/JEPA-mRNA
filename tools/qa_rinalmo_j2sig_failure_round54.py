import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs = {}
    for name, log_name in (("round54_j2sig_s42", "j2sig_s42.log"),
                           ("round54_j2sig_r3_s42", "j2sig_r3_s42.log")):
        run = ROOT / "runs" / name
        log = ROOT / "logs/round54" / log_name
        text = log.read_text()
        assert "torch.OutOfMemoryError" in text and "input_projection" in text
        assert "DONE" not in text
        rows = [json.loads(line) for line in (run / "jepa_diagnostics.jsonl").read_text().splitlines() if line.strip()]
        assert rows[-1]["step"] == 200
        runs[name] = {
            "last_step": rows[-1]["step"],
            "prediction_effective_rank": rows[-1]["prediction_effective_rank"],
            "prediction_cross_position_cosine": rows[-1]["prediction_cross_position_cosine"],
            "prediction_cross_sequence_cosine": rows[-1]["prediction_cross_sequence_cosine"],
            "sigreg_loss": rows[-1]["sigreg_loss"],
            "log_sha256": sha256(log),
            "diagnostics_sha256": sha256(run / "jepa_diagnostics.jsonl"),
        }
    output = {
        "status": "STOP_ONLINE_TARGET_SIGREG_REPEATED_OOM",
        "scientific_conclusion": "Early rank improvement is present, but neither anti-collapse nor VL0 efficacy is established.",
        "allow_retry": False,
        "allow_three_seed": False,
        "allow_fixed_or_random": False,
        "allow_no_ema_shared_online_target_review": True,
        "runs": runs,
    }
    path = ROOT / "audits/round54/j2sig_failure_qa.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
