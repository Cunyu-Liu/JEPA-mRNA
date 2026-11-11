import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rnajepa.distill import pair_indicator


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zchunk", type=int, default=8)
    args = parser.parse_args()
    assert args.zchunk > 0
    out = Path(args.out_dir)
    run = out / "run"
    if (run / "run_meta.json").exists():
        raise FileExistsError(f"refusing to overwrite {run}")
    rows = [json.loads(line) for line in Path(args.train_data).read_text().splitlines() if line.strip()]
    row = max(rows, key=lambda item: len(item["seq"]))
    assert len(row["seq"]) == 7224
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "input.jsonl").write_text(json.dumps(row) + "\n")
    sequence = row["seq"].upper().replace("T", "U")
    probabilities = pair_indicator(len(sequence), row["pairs"])
    shard = assets / "shard_00000.npz"
    np.savez(shard, sequences=np.asarray([sequence], dtype=object),
             probs=np.asarray([probabilities], dtype=object))
    manifest = {"teacher": "synthetic_pair_indicator_for_engineering_only",
                "version_lock": "not_a_scientific_teacher", "n_sequences": 1,
                "shards": [{"file": shard.name, "sha256": sha256_file(shard)}]}
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2))
    command = [
        sys.executable, "/home/rna_jev/tools/train_plan_a.py",
        "--data", str(assets / "input.jsonl"), "--teacher-dir", str(assets),
        "--out", str(run), "--steps", "1", "--batch-size", "1", "--seed", "42",
        "--save-every", "1", "--snapshot-every", "2000", "--snapshot-dir", str(out / "snapshots"),
        "--warmup-head-steps", "0", "--unfreeze-every", "1", "--unfreeze-per-step", "33",
        "--head-lr", "1e-5", "--backbone-lr", "1e-5", "--nll-normalization", "length",
        "--zchunk", str(args.zchunk), "--gpu-reserve-gb", "0", "--scorer", "pairmlp",
        "--pair-latent-dim", "490", "--ripe-mode", "r6",
    ]
    log_path = out / "train.log"
    with log_path.open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    result = {
        "status": "PASS_RINALMO_FULL_D1_LONGEST_TRAIN_CUDA_SMOKE" if process.returncode == 0 else "PARTIAL_RINALMO_FULL_D1_LONG_TRAIN_CUDA_FAILURE",
        "scientific_conclusion": False, "returncode": process.returncode,
        "length": len(sequence), "teacher": manifest["teacher"],
        "configuration": f"33 layers unfrozen, four objectives, NLL-length, PairMLP-490, R6, zchunk{args.zchunk}, batch1, one engineering step",
        "command": command, "train_log_sha256": sha256_file(log_path),
    }
    if (run / "run_meta.json").exists():
        meta = json.loads((run / "run_meta.json").read_text())
        result["run_meta"] = meta
        result["run_meta_sha256"] = sha256_file(run / "run_meta.json")
        if process.returncode == 0:
            assert meta["status"] == "completed" and meta["steps_completed"] == 1
            assert meta["environment"]["resolved_device"].startswith("cuda")
            train = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
            assert len(train) == 1 and np.isfinite(train[0]["loss"])
            result["train_record"] = train[0]
            result["resume_sha256"] = sha256_file(run / "resume.pt")
    (out / "qa.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
