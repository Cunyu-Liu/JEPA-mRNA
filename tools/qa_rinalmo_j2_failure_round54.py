import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs = {}
    for name in ("round54_j2_s42", "round54_j2_r1_s42"):
        run = ROOT / "runs" / name
        log = (ROOT / "logs/round54" / (name.replace("round54_j2_", "j2_") + ".log"))
        if not log.exists():
            log = ROOT / "logs/round54/j2_s42.log"
        text = log.read_text()
        assert "linalg.eigh: The algorithm failed to converge" in text
        assert "step 200/0" in text and "DONE" not in text
        rows = [json.loads(line) for line in (run / "jepa_diagnostics.jsonl").read_text().splitlines() if line.strip()]
        assert [row["step"] for row in rows] == [1, 25, 50, 75, 100, 125, 150, 175, 200]
        assert max(row["prediction_effective_rank"] for row in rows[1:]) < 3
        runs[name] = {
            "steps_with_diagnostics": len(rows),
            "last_step": rows[-1]["step"],
            "max_prediction_effective_rank_after_step1": max(row["prediction_effective_rank"] for row in rows[1:]),
            "last_prediction_cross_position_cosine": rows[-1]["prediction_cross_position_cosine"],
            "last_prediction_cross_sequence_cosine": rows[-1]["prediction_cross_sequence_cosine"],
            "log_sha256": sha256(log),
            "diagnostics_sha256": sha256(run / "jepa_diagnostics.jsonl"),
        }
    output = {
        "status": "STOP_J2_REPEATED_RANK_DIAGNOSTIC_FAILURE",
        "scientific_conclusion": "J2 has persistent low-rank risk; the joint near-unit-collapse criterion is not established.",
        "allow_j2_retry": False,
        "allow_j3_context_fallback": True,
        "allow_fixed_or_random": False,
        "runs": runs,
    }
    path = ROOT / "audits/round54/j2_failure_qa.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
