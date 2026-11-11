import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    for seed in (2026, 3407):
        path = root / "audits/round52" / f"rinalmo_long_sequence_probe_s{seed}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS_LONG_SEQUENCE_ENGINEERING_PROBE"
        assert report["scientific_conclusion"] is False and report["resolved_device"] == "cuda"
        checkpoint = root / "runs" / f"round52_rinalmo_length_s{seed}/best.pt"
        assert report["checkpoint_sha256"] == sha256_file(checkpoint)
        assert [row["length"] for row in report["rows"]] == [512, 1022, 1024, 1500, 1869, 3819]
        for row in report["rows"]:
            assert row["status"] == "PASS_FULL_LENGTH_CUDA" and row["input_length_preserved"]
            assert row["output_shape"] == [row["length"], row["length"]]
            assert row["valid_pairs"] > 0 and len(row["score_sha256"]) == 64
            assert row["peak_allocated_bytes"] > 0 and row["peak_reserved_bytes"] > 0
        results[str(seed)] = {
            "probe_sha256": sha256_file(path), "checkpoint_sha256": report["checkpoint_sha256"],
            "rows": report["rows"],
        }
    result = {"status": "PASS_RINALMO_REMAINING_LONG_SEQUENCE_ENGINEERING_QA",
              "scientific_conclusion": False, "results": results,
              "qa_sha256": sha256_file(__file__),
              "limitations": ["Full-length forward and memory only; not long-range accuracy, DP quality or formal speed evidence.",
                              "seed42 was accepted previously and was not repeated."]}
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
