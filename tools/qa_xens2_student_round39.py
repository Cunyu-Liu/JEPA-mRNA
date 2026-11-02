#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats == 3
    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    assert meta["converged"] and meta["early_stopped"] and meta["stop_reason"] == "early_stopping"
    checkpoint = args.run_dir / "best.pt"
    assert checkpoint.is_file()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(str(args.repo / name) for name in ("", "src", "tools", "eval"))
    results = []
    for repeat in range(args.repeats):
        out = args.out / f"repeat_{repeat}"
        log = (args.out / f"repeat_{repeat}.log").open("w")
        command = [sys.executable, str(args.repo / "eval/ss/evaluate_decision.py"),
                   "--checkpoint", str(checkpoint), "--data", str(args.data),
                   "--out", str(out), "--device", "cuda", "--exact-marginal-limit", "0",
                   "--tag", f"independent_gpu_qa_repeat_{repeat}", "--progress", "50"]
        with log:
            subprocess.run(command, cwd=args.repo, env=env, check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        result = json.loads((out / "result.json").read_text())
        assert result["device"] == "cuda" and result["n_sequences"] == 196
        assert Path(result["checkpoint"]).resolve() == checkpoint.resolve()
        assert Path(result["data"]).resolve() == args.data.resolve()
        results.append(result)
    identities = [json.dumps([[row["name"], row["pred_pairs"]] for row in result["per_sequence"]],
                             sort_keys=True, separators=(",", ":")) for result in results]
    assert len(set(identities)) == 1
    counts = [(result["pair_level"]["micro"]["tp"], result["pair_level"]["micro"]["fp"],
               result["pair_level"]["micro"]["fn"]) for result in results]
    assert counts == [counts[0]] * 3
    f1s = [result["pair_level"]["micro"]["f1"] for result in results]
    eces = [result["calibration"]["system1"]["ece"] for result in results]
    summary = {"schema_version": 1, "status": "PASS", "scope": "independent_gpu_qa_vl0_not_benchmark",
               "run_dir": str(args.run_dir), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(checkpoint), "data": str(args.data),
               "data_sha256": sha256(args.data), "n_sequences": 196,
               "prediction_sha256": hashlib.sha256(identities[0].encode()).hexdigest(),
               "tp_fp_fn": counts[0], "f1_mean": statistics.mean(f1s),
               "f1_sd": statistics.stdev(f1s), "ece_mean": statistics.mean(eces),
               "ece_sd": statistics.stdev(eces), "repeats": [
                   {"repeat": index, "f1": f1s[index], "ece": eces[index],
                    "result": str(args.out / f"repeat_{index}/result.json"),
                    "result_sha256": sha256(args.out / f"repeat_{index}/result.json")}
                   for index in range(args.repeats)]}
    (args.out / "qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
