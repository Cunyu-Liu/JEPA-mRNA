import argparse
import json
from pathlib import Path

from speed_rinalmo_jmlp_round52 import prediction_digest, sha256_file, summarize_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--expected-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    expected = json.loads(Path(args.expected_result).read_text())
    runs = []
    for line in Path(args.log).read_text().splitlines():
        if line.startswith('{"repeat_complete"'):
            run = json.loads(line)
            run.pop("repeat_complete")
            runs.append(run)
    summary = summarize_runs(runs)
    assert [run["repeat"] for run in runs] == [0, 1, 2]
    assert summary["micro"] == expected["micro"]
    expected_pairs = [[list(pair) for pair in row["pred_pairs"]] for row in expected["predictions"][0]]
    assert summary["predictions_sha256"] == prediction_digest(rows, expected_pairs)
    assert expected["checkpoint_sha256"] == sha256_file(args.checkpoint)
    assert expected["data_sha256"] == sha256_file(args.data)
    result = {
        "status": "PASS_JMLP_EXCLUSIVE_RAW_SEQUENCE_SPEED", "scientific_conclusion": False,
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
        "data": args.data, "data_sha256": sha256_file(args.data),
        "encoder_load": expected["encoder_load"], "device": "cuda", "gpu_name": expected["gpu_name"],
        "decoder": "nussinov", "dp_backend": "c", "repeats": 3, "warmup_sequences": 32,
        "n": len(rows), "exclusive_gpu": True,
        "protocol": "single checkpoint, raw sequence B=1, full forward and required C DP once; three full repeats; predictions and F1 exactly match accepted benchmark",
        "runs": runs, "summary": summary, "code_sha256": sha256_file(__file__),
        "recovery": "Finalized from three completed repeat logs after fixing expected pred_pairs schema; no inference rerun",
        "limitations": ["Speed is a J-mlp control measurement, not a SOTA claim without same-hardware baselines.",
                        "No ensemble, cached embeddings, test-time augmentation or concurrent GPU tenants is used."],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
