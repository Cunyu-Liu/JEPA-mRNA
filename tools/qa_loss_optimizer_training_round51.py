import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_load(path):
    path = Path(path)
    for _ in range(20):
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        temporary = Path(f"/tmp/round51_final_qa_{os.getpid()}_{path.name}")
        shutil.copyfile(path, temporary)
        after = (path.stat().st_size, path.stat().st_mtime_ns)
        if before == after and temporary.stat().st_size == before[0]:
            value = torch.load(temporary, map_location="cpu", weights_only=False)
            temporary.unlink()
            return value
        temporary.unlink(missing_ok=True)
        time.sleep(1)
    raise RuntimeError(f"no stable checkpoint copy: {path}")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_run(root, arm, seed, normalization, backbone_lr, data_sha256):
    root = Path(root)
    meta = json.loads((root / "run_meta.json").read_text())
    expected_name = f"loss_{arm}_s{seed}"
    assert meta["arm"] == expected_name and meta["seed"] == seed
    assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
    assert meta["stop_reason"] == "early_stopping"
    assert meta["schedule"]["steps"] == 0
    assert meta["validation"]["every"] == 500 and meta["validation"]["patience"] == 6
    assert meta["objective"]["nll_normalization"] == normalization
    assert float(meta["schedule"]["backbone_lr"]) == backbone_lr
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["data"] == "/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"
    validation = read_jsonl(root / "validation_log.jsonl")
    expected_steps = list(range(500, meta["steps_completed"] + 1, 500))
    assert [row["step"] for row in validation] == expected_steps
    assert all(math.isfinite(row["validation_loss"]) for row in validation)
    best = min(validation, key=lambda row: row["validation_loss"])
    assert best["step"] == meta["best_step"]
    assert best["validation_loss"] == meta["best_validation_loss"]
    assert validation[-1]["bad_evaluations"] == 6
    train = read_jsonl(root / "train_log.jsonl")
    assert train and all(math.isfinite(row["loss"]) and math.isfinite(row["gnorm"]) for row in train)
    assert all(math.isfinite(value) for row in train for value in row["terms"].values())
    best_checkpoint = stable_load(root / "best.pt")
    assert best_checkpoint["step"] == meta["best_step"]
    assert best_checkpoint["validation_loss"] == meta["best_validation_loss"]
    resume = stable_load(root / "resume.pt")
    assert resume["step"] == meta["steps_completed"]
    assert resume["scheduler"]["_last_lr"] == [1e-4, backbone_lr]
    for part in ("encoder", "head"):
        left = best_checkpoint["model"][part]
        right = resume[part]
        assert list(left) == list(right)
        assert all(torch.equal(left[key], right[key]) for key in left)
    return {
        "run": expected_name,
        "seed": seed,
        "nll_normalization": normalization,
        "backbone_lr": backbone_lr,
        "steps_completed": meta["steps_completed"],
        "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"],
        "validation_points": len(validation),
        "validation_curve": validation,
        "max_vl0": max(row["validation_loss"] for row in validation),
        "max_abs_train_loss": max(abs(row["loss"]) for row in train),
        "max_gnorm": max(row["gnorm"] for row in train),
        "final_train": train[-1],
        "best_sha256": sha256_file(root / "best.pt"),
        "resume_sha256": sha256_file(root / "resume.pt"),
        "run_meta_sha256": sha256_file(root / "run_meta.json"),
        "data_sha256": data_sha256,
        "resume_weights_equal_best": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch-code-commit", default="d17dd1d39e1761b275836c3b2f9888af6e51304e")
    args = parser.parse_args()
    root = Path(args.root)
    data = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    data_sha256 = sha256_file(data)
    teacher_manifest = Path("/root/rna_jepa_data/ss_data/teacher/bprna_tr1c/manifest.json")
    runs = {}
    for arm, normalization, lr in (("length", "length", 1e-5), ("olr", "sum", 1e-6)):
        for seed in (42, 2026, 3407):
            report = validate_run(root / "runs" / f"loss_{arm}_s{seed}", arm, seed, normalization, lr, data_sha256)
            runs[report["run"]] = report
    aggregates = {}
    for arm in ("length", "olr"):
        values = [runs[f"loss_{arm}_s{seed}"]["best_vl0"] for seed in (42, 2026, 3407)]
        aggregates[arm] = {
            "mean_best_vl0": statistics.mean(values),
            "population_sd_best_vl0": statistics.pstdev(values),
            "best_vl0_values": values,
        }
    output = {
        "status": "PASS_LOSS_OPTIMIZER_TRAINING_THREE_CHECK_QA",
        "scientific_conclusion": False,
        "launch_code_commit": args.launch_code_commit,
        "data_sha256": data_sha256,
        "teacher_manifest_sha256": sha256_file(teacher_manifest),
        "runs": runs,
        "aggregates": aggregates,
        "notes": [
            "Natural early stopping and complete late curves are verified; this is not a test-set result.",
            "VL0 values under different NLL normalizations are not direct structure-performance comparisons.",
            "No capacity, interaction, Pair-JEPA, D0/D1, motif, or final-speed claim is allowed by this artifact alone.",
        ],
    }
    temporary = Path(args.output).with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": output["status"], "aggregates": aggregates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
