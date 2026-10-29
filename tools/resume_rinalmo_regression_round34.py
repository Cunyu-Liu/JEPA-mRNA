#!/usr/bin/env python3
"""Resume partial TE/EL runs without retraining completed, fully verified heads."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy import stats

MODULE_PATH = Path("/home/rna_jev/tools/rinalmo_regression_dual_round33_v4.py")
spec = importlib.util.spec_from_file_location("rinalmo_regression_original_v4", MODULE_PATH)
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)
HEAD_CLASSES = {"official_head": dual.OfficialRegressionHead, "jev_head": dual.JevRegressionHead}
MARKER = re.compile(r"^\[(?P<task>[^/]+)/(?P<head>official_head|jev_head)/s(?P<seed>[0-4])\] R2=.*? best=(?P<best>\d+)\s*$")
EPOCH = re.compile(r"^\[seed=(?P<seed>[0-4])/epoch=(?P<epoch>\d+)\] validation_loss=(?P<loss>[-+0-9.eE]+) best=")


def free_mib() -> int:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0]
    total, used = (int(value.strip()) for value in text.split(","))
    return total - used


def completed_segments(log_path: Path, task: str) -> dict[tuple[str, int], tuple[int, int, list[str]]]:
    lines = log_path.read_text(errors="strict").splitlines()
    markers = []
    for index, line in enumerate(lines):
        match = MARKER.match(line)
        if match and match.group("task") == task:
            markers.append((index, match))
    segments = {}
    previous = 0
    for index, match in markers:
        key = (match.group("head"), int(match.group("seed")))
        if key in segments:
            raise RuntimeError(f"duplicate completion marker: {key}")
        segments[key] = (previous, index, lines[previous:index], int(match.group("best")))
        previous = index + 1
    return segments


def parse_history(segment: list[str], seed: int, best_epoch: int, best_loss: float) -> list[dict]:
    points = {}
    for line in segment:
        match = EPOCH.match(line)
        if match and int(match.group("seed")) == seed:
            epoch = int(match.group("epoch"))
            if epoch == 1:
                points = {}
            elif epoch in points:
                raise RuntimeError(f"duplicate epoch in log: seed={seed} epoch={epoch}")
            points[epoch] = float(match.group("loss"))
    if not points:
        raise RuntimeError(f"no validation history for seed={seed}")
    logged_epochs = max(points)
    epochs_run = best_epoch + 32
    if sorted(points) != list(range(1, logged_epochs + 1)):
        raise RuntimeError(f"non-contiguous validation history for seed={seed}")
    if logged_epochs not in {epochs_run - 1, epochs_run}:
        raise RuntimeError(f"early-stopping length mismatch for seed={seed}: {logged_epochs} != {epochs_run} or its unlogged predecessor")
    if best_epoch not in points:
        raise RuntimeError(f"best epoch absent from log for seed={seed}")
    points[best_epoch] = best_loss
    return [{"epoch": epoch, "validation_loss": points.get(epoch)} for epoch in range(1, epochs_run + 1)]


def evaluate(model, arrays, indices, device="cuda", batch_size=32):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = np.asarray(indices[start:start + batch_size], dtype=int)
            x = torch.as_tensor(arrays["embs"][idx], dtype=torch.float32, device=device)
            mask = torch.as_tensor(arrays["padding_mask"][idx], dtype=torch.bool, device=device)
            predictions.append(model(x, mask).cpu().numpy())
    return np.concatenate(predictions)


def r2(truth, pred):
    return float(1 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2))


def restore_run(head_name, seed, arrays, labels, split, checkpoint, prediction, history):
    head_cls = HEAD_CLASSES[head_name]
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt["head"] != head_name or ckpt["seed"] != seed:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint}")
    model = head_cls()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.cuda().eval()
    test_idx = np.asarray(split["test"], dtype=int)
    valid_idx = np.asarray(split["validation"], dtype=int)
    raw_test = evaluate(model, arrays, test_idx)
    raw_valid = evaluate(model, arrays, valid_idx)
    del model
    torch.cuda.empty_cache()
    with np.load(prediction, allow_pickle=False) as saved:
        pred_test = saved["predictions"].copy()
        truth_test = saved["truth"].copy()
    if not np.array_equal(truth_test, labels[test_idx]):
        raise RuntimeError(f"saved truth mismatch: {prediction}")
    if not np.allclose(raw_test, pred_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"GPU re-inference mismatch before resume: {head_name} seed={seed}")
    pred_valid = raw_valid
    truth_valid = labels[valid_idx]
    return {
        "seed": seed, "device": "cuda", "converged": True, "early_stopped": True,
        "best_epoch": int(ckpt["best_epoch"]), "epochs_run": len(history),
        "best_validation_loss": float(ckpt["best_validation_loss"]), "validation_history": history,
        "validation_r2": r2(truth_valid, pred_valid),
        "validation_pcc": float(stats.pearsonr(pred_valid, truth_valid)[0]),
        "validation_spearman": float(stats.spearmanr(pred_valid, truth_valid)[0]),
        "test_r2": r2(truth_test, pred_test),
        "test_pcc": float(stats.pearsonr(pred_test, truth_test)[0]),
        "test_spearman": float(stats.spearmanr(pred_test, truth_test)[0]),
        "n_train": len(split["train"]), "n_validation": len(valid_idx), "n_test": len(test_idx),
        "label_mean": 0.0, "label_std": 1.0,
        "checkpoint": str(checkpoint), "checkpoint_sha256": dual.sha256(checkpoint),
        "prediction_file": str(prediction), "prediction_sha256": dual.sha256(prediction),
        "resume_restored": True,
        "history_source": "launch_log_6_decimal_except_best_checkpoint; unlogged_final_epoch_is_null",
        "history_unlogged_epochs": [point["epoch"] for point in history if point["validation_loss"] is None],
    }


def save_new_run(out_dir, head_name, seed, run):
    pred, truth, best_state = run.pop("pred_test"), run.pop("truth_test"), run.pop("best_state")
    checkpoint = out_dir / f"{head_name}_s{seed}_best.pt"
    prediction = out_dir / f"{head_name}_s{seed}_predictions.npz"
    if checkpoint.exists() or prediction.exists():
        raise RuntimeError(f"new run would overwrite existing artifacts: {head_name} seed={seed}")
    tmp_ckpt = checkpoint.with_suffix(".tmp")
    with tmp_ckpt.open("wb") as handle:
        torch.save({"model_state_dict": best_state, "head": head_name, "seed": seed,
                    "best_epoch": run["best_epoch"], "best_validation_loss": run["best_validation_loss"]}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_ckpt.replace(checkpoint)
    tmp_pred = prediction.with_suffix(".tmp")
    with tmp_pred.open("wb") as handle:
        np.savez_compressed(handle, predictions=pred, truth=truth)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_pred.replace(prediction)
    run.update({"checkpoint": str(checkpoint), "checkpoint_sha256": dual.sha256(checkpoint),
                "prediction_file": str(prediction), "prediction_sha256": dual.sha256(prediction),
                "resume_restored": False})
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["te", "el"])
    parser.add_argument("--cell-line", required=True, choices=["Muscle", "pc3", "HEK"])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result_path = args.out_dir / "result.json"
    log_path = args.out_dir.with_suffix(".launch.log")
    if result_path.exists() or not log_path.is_file():
        raise RuntimeError("resume requires no final result and an existing launch log")
    segments = completed_segments(log_path, args.task)
    candidates = {}
    for head_name in HEAD_CLASSES:
        for seed in dual.SEEDS:
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            key = (head_name, seed)
            if key in segments:
                if not checkpoint.is_file() or not prediction.is_file():
                    raise RuntimeError(f"completion marker without artifacts: {key}")
                candidates[key] = "restore"
            elif checkpoint.exists() or prediction.exists():
                raise RuntimeError(f"partial artifact without completion marker: {key}")
            else:
                candidates[key] = "train"
    print("RESUME_PLAN", json.dumps({f"{head}/s{seed}": action for (head, seed), action in candidates.items()}), flush=True)
    if args.dry_run:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    available = free_mib()
    if available < args.min_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {available} < {args.min_free_mib} MiB")
    initial_log_sha256 = dual.sha256(log_path)
    df, split = dual.load_cell(args.task, args.cell_line, args.fold)
    labels = df["te_log" if args.task == "te" else "rnaseq_log"].to_numpy(dtype=float)
    cache = dual.ROOT / f"data/rinalmo/{args.cell_line}_official_tokens.npz"
    arrays = dual.extract_embeddings(df, cache, "cuda", 64)
    outputs = {}
    restored = []
    for head_name, head_cls in HEAD_CLASSES.items():
        runs = []
        for seed in dual.SEEDS:
            key = (head_name, seed)
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            if candidates[key] == "restore":
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if segments[key][3] != int(ckpt["best_epoch"]):
                    raise RuntimeError(f"completion marker/checkpoint best epoch mismatch: {key}")
                history = parse_history(segments[key][2], seed, int(ckpt["best_epoch"]), float(ckpt["best_validation_loss"]))
                run = restore_run(head_name, seed, arrays, labels, split, checkpoint, prediction, history)
                restored.append(f"{head_name}/s{seed}")
                print(f"RESUME_RESTORE_PASS {args.task}/{head_name}/s{seed}", flush=True)
            else:
                run = dual.train_one(arrays, labels, split, head_cls, seed, "cuda", False, True)
                run = save_new_run(args.out_dir, head_name, seed, run)
                print(f"[{args.task}/{head_name}/s{seed}] R2={run['test_r2']:.5f} PCC={run['test_pcc']:.5f} best={run['best_epoch']}", flush=True)
            runs.append(run)
        outputs[head_name] = {"runs": runs,
                              "r2_mean": float(np.mean([run["test_r2"] for run in runs])),
                              "r2_sample_std": float(np.std([run["test_r2"] for run in runs], ddof=1)),
                              "pcc_mean": float(np.mean([run["test_pcc"] for run in runs])),
                              "spearman_mean": float(np.mean([run["test_spearman"] for run in runs]))}
    official = np.array([run["test_r2"] for run in outputs["official_head"]["runs"]])
    jev = np.array([run["test_r2"] for run in outputs["jev_head"]["runs"]])
    official_spearman = np.array([run["test_spearman"] for run in outputs["official_head"]["runs"]])
    jev_spearman = np.array([run["test_spearman"] for run in outputs["jev_head"]["runs"]])
    test = stats.ttest_rel(jev, official, alternative="greater")
    spearman_test = stats.ttest_rel(jev_spearman, official_spearman, alternative="greater")
    primary_delta = jev_spearman - official_spearman
    primary_delta_sd = float(np.std(primary_delta, ddof=1))
    statistics_ok = bool(primary_delta_sd > 0 and np.isfinite(spearman_test.statistic) and np.isfinite(spearman_test.pvalue))
    comparison = {"primary_metric": "spearman", "delta_r2_mean": float(np.mean(jev - official)),
                  "delta_spearman_mean": float(np.mean(primary_delta)),
                  "inference_scope": "paired training-seed effects; no test-set population inference",
                  "one_sided_p_value": float(test.pvalue) if np.isfinite(test.pvalue) else None,
                  "one_sided_spearman_p_value": float(spearman_test.pvalue) if np.isfinite(spearman_test.pvalue) else None,
                  "primary_delta_mean": float(np.mean(primary_delta)), "primary_delta_sample_sd": primary_delta_sd,
                  "primary_one_sided_p_value": float(spearman_test.pvalue) if np.isfinite(spearman_test.pvalue) else None,
                  "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
                  "significant_positive": bool(statistics_ok and np.mean(primary_delta) > 0 and spearman_test.pvalue < 0.05)}
    protocol = {"task": args.task, "cell_line": args.cell_line, "fold": args.fold,
                "sample_scope": "full_official_protocol", "device": "cuda",
                "single_variable": "head architecture on frozen RiNALMo token embeddings",
                "convergence": "validation loss; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
                "label_scaling": "none", "source_data": str(dual.ROOT / f"data/rinalmo/te_el_official/{args.cell_line}_sequence.csv"),
                "pretrained_weights": dual.WEIGHTS,
                "pretrained_sha256": dual.sha256(Path(dual.WEIGHTS) / "model.safetensors"),
                "pretrained_config": str(Path(dual.WEIGHTS) / "config.json"),
                "pretrained_config_sha256": dual.sha256(Path(dual.WEIGHTS) / "config.json"),
                "preflight_sha256": dual.sha256(Path("/home/rna_jev/tools/rinalmo_preflight.py")),
                "embedding_cache_meta": str(cache.with_suffix(".json")),
                "embedding_cache_meta_sha256": dual.sha256(cache.with_suffix(".json")),
                "seeds": list(dual.SEEDS), "optimizer": "Adam(lr=1e-3)", "train_batch_size": 32,
                "embedding_batch_size": 64, "code_sha256": dual.sha256(MODULE_PATH),
                "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip(),
                "resume_script_sha256": dual.sha256(Path(__file__).resolve()),
                "resume_initial_log_sha256": initial_log_sha256, "resume_restored_runs": restored}
    protocol["source_data_sha256"] = dual.sha256(Path(protocol["source_data"]))
    payload = {"schema_version": 1, "status": "PASS", "protocol": protocol,
               "heads": outputs, "comparison": comparison}
    for head_name, head_cls in HEAD_CLASSES.items():
        for run in outputs[head_name]["runs"]:
            saved = np.load(run["prediction_file"], allow_pickle=False)
            expected = r2(saved["truth"], saved["predictions"])
            assert np.isclose(expected, run["test_r2"], atol=1e-10)
            assert dual.sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            model = head_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert ckpt["seed"] == run["seed"] and run["device"] == "cuda" and run["early_stopped"]
            assert len(run["validation_history"]) == run["epochs_run"] == run["best_epoch"] + 32
            best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
            assert best_point["validation_loss"] == run["best_validation_loss"]
            run["independent_recheck"] = "PASS"
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=dual.json_default, allow_nan=False) + "\n")
    temporary.replace(result_path)
    assert json.loads(result_path.read_text())["status"] == "PASS"
    print(f"RINALMO_REGRESSION_DUAL_PASS task={args.task} delta={comparison['delta_r2_mean']:+.6f} p={comparison['one_sided_p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
