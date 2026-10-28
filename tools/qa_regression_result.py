#!/usr/bin/env python3
"""Independent saved-output and GPU re-inference QA for RiNALMo regression runs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats

MODULE_PATH = os.environ.get("REGRESSION_MODULE_PATH", "/home/rna_jev/tools/rinalmo_regression_dual_round33.py")
spec = importlib.util.spec_from_file_location("regression_dual_for_qa", MODULE_PATH)
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r2_score(truth, predictions):
    return float(1 - np.sum((truth - predictions) ** 2) / np.sum((truth - truth.mean()) ** 2))


def load_scope(task, cell_line, fold, limit):
    if task == "mrl":
        df, split = dual.load_mrl()
    else:
        df, split = dual.load_cell(task, cell_line, fold)
    if limit:
        for key in split:
            split[key] = split[key][:limit]
        used = sorted(split["train"] + split["validation"] + split["test"])
        remap = {old: new for new, old in enumerate(used)}
        df = df.iloc[used].copy().reset_index(drop=True)
        for key in split:
            split[key] = [remap[index] for index in split[key]]
    labels = df["rl"].to_numpy(dtype=float) if task == "mrl" else df["te_log" if task == "te" else "rnaseq_log"].to_numpy(dtype=float)
    return df, split, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--task", choices=["mrl", "te", "el"], required=True)
    parser.add_argument("--cell-line", default="HEK", choices=["Muscle", "pc3", "HEK"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU regression QA"
    result = json.loads(args.result.read_text())
    protocol = result["protocol"]
    assert result["status"] == "PASS"
    assert protocol["task"] == args.task and protocol["device"] == "cuda"
    assert protocol["cell_line"] == args.cell_line and protocol["fold"] == args.fold
    expected_scope = "engineering_smoke_subset_not_scientific_result" if args.limit else "full_official_protocol"
    assert protocol["sample_scope"] == expected_scope
    assert protocol["code_sha256"] == sha256(MODULE_PATH)
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    meta_path = Path(protocol["embedding_cache_meta"])
    assert sha256(meta_path) == protocol["embedding_cache_meta_sha256"]
    meta = json.loads(meta_path.read_text())
    assert meta["pretrained_sha256"] == protocol["pretrained_sha256"]
    assert meta["pretrained_config_sha256"] == protocol["pretrained_config_sha256"]
    assert meta["preflight_sha256"] == protocol["preflight_sha256"]
    assert meta["extractor_sha256"] == protocol["code_sha256"]
    embs_path = meta_path.with_suffix(".embs.npy")
    mask_path = meta_path.with_suffix(".mask.npy")
    assert meta["embedding_content_sha256"] == sha256(embs_path)
    assert meta["padding_mask_sha256"] == sha256(mask_path)
    df, split, labels = load_scope(args.task, args.cell_line, args.fold, args.limit)
    assert dual.sequence_sha256(df["utr"].astype(str).str.replace("<pad>", "", regex=False)) == meta["sequence_sha256"]
    embs = np.load(embs_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    expected_counts = (len(split["train"]), len(split["validation"]), len(split["test"]))
    if not args.limit:
        assert expected_counts == ((76319, 7600, 7600) if args.task == "mrl" else expected_counts)
    assert (len(split["train"]), len(split["validation"]), len(split["test"])) == expected_counts
    test_idx = np.asarray(split["test"], dtype=int)
    truth = labels[test_idx]
    head_classes = {"official_head": dual.OfficialRegressionHead, "jev_head": dual.JevRegressionHead}
    for head_name, head_cls in head_classes.items():
        runs = result["heads"][head_name]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        values = []
        for run in runs:
            assert run["device"] == "cuda" and run["converged"] and run["early_stopped"]
            assert run["independent_recheck"] == "PASS"
            assert run["epochs_run"] == run["best_epoch"] + 32
            assert len(run["validation_history"]) == run["epochs_run"]
            assert (run["n_train"], run["n_validation"], run["n_test"]) == expected_counts
            best = next(point for point in run["validation_history"] if point["epoch"] == run["best_epoch"])
            assert best["validation_loss"] == run["best_validation_loss"]
            assert sha256(run["prediction_file"]) == run["prediction_sha256"]
            assert sha256(run["checkpoint"]) == run["checkpoint_sha256"]
            with np.load(run["prediction_file"], allow_pickle=False) as saved:
                predictions = saved["predictions"].copy()
                assert np.array_equal(saved["truth"], truth)
            assert predictions.shape == (len(test_idx),)
            assert np.isclose(r2_score(truth, predictions), run["test_r2"], atol=1e-10)
            assert np.isclose(stats.pearsonr(predictions, truth)[0], run["test_pcc"], atol=1e-10)
            assert np.isclose(stats.spearmanr(predictions, truth)[0], run["test_spearman"], atol=1e-10)
            checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            assert checkpoint["seed"] == run["seed"] and checkpoint["head"] == head_name
            model = head_cls()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model = model.cuda().eval()
            reproduced = []
            with torch.inference_mode():
                for start in range(0, len(test_idx), 32):
                    idx = test_idx[start:start + 32]
                    x = torch.as_tensor(embs[idx], dtype=torch.float32, device="cuda")
                    mask = torch.as_tensor(masks[idx], dtype=torch.bool, device="cuda")
                    reproduced.append(model(x, mask).cpu().numpy())
            reproduced = np.concatenate(reproduced) * run["label_std"] + run["label_mean"]
            if args.task == "mrl":
                reproduced = np.maximum(reproduced, 0.0)
            assert np.allclose(reproduced, predictions, atol=1e-5, rtol=1e-5), f"GPU re-inference mismatch: {head_name} seed={run['seed']}"
            values.append(run["test_r2"])
            print("REGRESSION_QA_RUN_PASS", head_name, run["seed"], flush=True)
            del model, checkpoint
            torch.cuda.empty_cache()
        assert np.isclose(np.mean(values), result["heads"][head_name]["r2_mean"], atol=1e-12)
        assert np.isclose(np.std(values, ddof=1), result["heads"][head_name]["r2_sample_std"], atol=1e-12)
    official = np.asarray([run["test_r2"] for run in result["heads"]["official_head"]["runs"]])
    jev = np.asarray([run["test_r2"] for run in result["heads"]["jev_head"]["runs"]])
    comparison = result["comparison"]
    test = stats.ttest_rel(jev, official, alternative="greater")
    assert np.isclose(np.mean(jev - official), comparison["delta_r2_mean"], atol=1e-12)
    if comparison["one_sided_p_value"] is None:
        assert not np.isfinite(test.pvalue)
    else:
        assert np.isclose(float(test.pvalue), comparison["one_sided_p_value"], atol=1e-12)
    if "primary_metric" in comparison:
        official_spearman = np.asarray([run["test_spearman"] for run in result["heads"]["official_head"]["runs"]])
        jev_spearman = np.asarray([run["test_spearman"] for run in result["heads"]["jev_head"]["runs"]])
        spearman_test = stats.ttest_rel(jev_spearman, official_spearman, alternative="greater")
        assert np.isclose(np.mean(jev_spearman - official_spearman), comparison["delta_spearman_mean"], atol=1e-12)
        if comparison["one_sided_spearman_p_value"] is None:
            assert not np.isfinite(spearman_test.pvalue)
        else:
            assert np.isclose(float(spearman_test.pvalue), comparison["one_sided_spearman_p_value"], atol=1e-12)
        primary_delta = (jev - official) if comparison["primary_metric"] == "r2" else (jev_spearman - official_spearman)
        primary_test = test if comparison["primary_metric"] == "r2" else spearman_test
        assert np.isclose(np.mean(primary_delta), comparison["primary_delta_mean"], atol=1e-12)
        primary_delta_sd = float(np.std(primary_delta, ddof=1))
        statistics_ok = bool(primary_delta_sd > 0 and np.isfinite(primary_test.statistic) and np.isfinite(primary_test.pvalue))
        assert comparison["statistics_status"] == ("ok" if statistics_ok else "undefined_zero_variance")
        if comparison["primary_one_sided_p_value"] is None:
            assert not np.isfinite(primary_test.pvalue)
        else:
            assert np.isclose(float(primary_test.pvalue), comparison["primary_one_sided_p_value"], atol=1e-12)
        expected_significant = bool(statistics_ok and np.mean(primary_delta) > 0 and primary_test.pvalue < 0.05)
        assert comparison["significant_positive"] == expected_significant
    print("REGRESSION_RESULT_EXTERNAL_QA_PASS", json.dumps({
        "scope": "smoke" if args.limit else "full", "runs": 10,
        "result_sha256": sha256(args.result), "cuda_device": torch.cuda.get_device_name(0)}), flush=True)


if __name__ == "__main__":
    main()
