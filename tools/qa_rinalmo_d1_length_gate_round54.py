import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    passed = json.loads((ROOT / "audits/round54/d1_length_1022_longest/result.json").read_text())
    failed = json.loads((ROOT / "audits/round54/d1_length_2048_longest/result.json").read_text())
    assert passed["status"] == "PASS_D1_LENGTH_FULL_B0_ENGINEERING" and passed["length"] == 975
    assert passed["run_meta"]["status"] == "completed" and passed["checkpoint_sha256"]
    assert failed["status"] == "PARTIAL_D1_LENGTH_FULL_B0_FAILURE" and failed["length"] == 2012
    log = (ROOT / "audits/round54/d1_length_2048_longest/train.log").read_text()
    assert "torch.OutOfMemoryError" in log and "batch_norm" in log
    output = {
        "status": "FREEZE_D1_LENGTH_1022",
        "scientific_conclusion": False,
        "selected_length_nt": 1022,
        "n_complex_train": 155,
        "n_complex_holdout": 96,
        "passed": {"threshold": 1022, "length": 975, "peak_gb": 14.1,
                   "result_sha256": sha256(ROOT / "audits/round54/d1_length_1022_longest/result.json")},
        "failed": {"threshold": 2048, "length": 2012, "reason": "CUDA OOM in ResNet2D BatchNorm backward",
                   "result_sha256": sha256(ROOT / "audits/round54/d1_length_2048_longest/result.json")},
        "not_attempted": {"threshold": 3000, "reason": "2048 failed; no repeated OOM rescue"},
        "limitations": ["Engineering gate only; no D1 accuracy or long-range generalization claim."],
    }
    path = ROOT / "audits/round54/d1_length_gate_qa.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
