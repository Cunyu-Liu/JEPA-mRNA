#!/usr/bin/env python3
"""Independent full-result QA for RiNALMo 88-class ncRNA dual-head runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/rinalmo/ncrna_official"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 88))

    def forward(self, x):
        return self.net(x)


class JevHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1280, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 88))

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", choices=["clean", "bn200"], required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable; refusing silent CPU validation"
    torch.cuda.set_device(0)
    suffix = "" if args.noise == "clean" else "_bn200"
    result_path = ROOT / f"eval_decision/rinalmo_ncrna_{args.noise}_round33_full_v1/result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "PASS"
    protocol = result["protocol"]
    assert protocol["sample_scope"] == "full_dataset" and protocol["device"] == "cuda"
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["primary_metric"] == "weighted_f1; accuracy retained as secondary metric"
    assert protocol["pretrained_sha256"] == "0b3f73b5c80d267f0b7e92655b28f16088e3d5a595c6dae631b52142aa38bcda"
    current_card_path = Path(protocol["data_card"])
    current_card = json.loads(current_card_path.read_text())
    assert current_card["status"] == "PASS"
    assert current_card["counts"][f"test{suffix}"] == {"rows": 25342, "classes": 88}
    for name, expected_hash in current_card["files"].items():
        assert sha256(DATA / name) == expected_hash
    current_card_sha256 = sha256(current_card_path)
    frame = pd.read_csv(DATA / f"test{suffix}.csv")
    assert len(frame) == 25342
    truth = frame["class_id"].to_numpy(dtype=np.int64)
    assert sorted(frame["class_id"].unique().tolist()) == list(range(88))
    cache_path = DATA / f"ncrna_{args.noise}_rinalmo_cls_v1.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache["embs"].shape == (148530, 1280)
        assert np.array_equal(cache["labels"][-25342:], truth)
        embeddings = cache["embs"][-25342:]
    x = torch.as_tensor(embeddings, dtype=torch.float32, device="cuda")
    del embeddings
    summaries = {}
    official_values, jev_values = [], []
    for head_name, model_cls in [("official_head", OfficialHead), ("jev_head", JevHead)]:
        runs = result["heads"][head_name]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        values = []
        for run in runs:
            assert run["device"] == "cuda" and run["converged"] and run["early_stopped"]
            assert run["independent_recheck"] == "PASS"
            assert (run["n_train"], run["n_validation"], run["n_test"]) == (105864, 17324, 25342)
            history = run["validation_history"]
            assert len(history) == run["epochs_run"] == run["best_epoch"] + 32
            assert [point["epoch"] for point in history] == list(range(1, run["epochs_run"] + 1))
            assert all(np.isfinite(point["validation_ce"]) and point["validation_ce"] >= 0 for point in history)
            best = next(point for point in history if point["epoch"] == run["best_epoch"])
            assert best["validation_ce"] == run["best_validation_ce"]
            assert all(point["validation_ce"] >= run["best_validation_ce"] - 1e-5 for point in history[run["best_epoch"]:])
            prediction_path = Path(run["prediction_file"])
            checkpoint_path = Path(run["checkpoint"])
            assert sha256(prediction_path) == run["prediction_sha256"]
            assert sha256(checkpoint_path) == run["checkpoint_sha256"]
            with np.load(prediction_path, allow_pickle=False) as saved:
                predictions = saved["predictions"]
                assert np.array_equal(saved["truth"], truth)
            assert predictions.shape == (25342,) and predictions.dtype.kind in "iu"
            accuracy = float(accuracy_score(truth, predictions))
            weighted_f1 = float(f1_score(truth, predictions, average="weighted"))
            assert np.isclose(accuracy, run["accuracy"], atol=1e-12)
            assert np.isclose(weighted_f1, run["weighted_f1"], atol=1e-12)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert checkpoint["seed"] == run["seed"] and checkpoint["head"] == head_name
            assert checkpoint["best_epoch"] == run["best_epoch"]
            model = model_cls()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model = model.cuda().eval()
            reproduced = []
            with torch.inference_mode():
                for start in range(0, len(x), 4096):
                    reproduced.append(model(x[start:start + 4096]).argmax(dim=-1).cpu().numpy())
            reproduced = np.concatenate(reproduced)
            assert np.array_equal(reproduced, predictions), f"GPU re-inference mismatch: {head_name} seed={run['seed']}"
            values.append(weighted_f1)
            print(f"QA_RUN_PASS {args.noise} {head_name} seed={run['seed']} f1={weighted_f1:.10f}", flush=True)
            del model, checkpoint, reproduced
            torch.cuda.empty_cache()
        assert np.isclose(np.mean(values), result["heads"][head_name]["weighted_f1_mean"], atol=1e-12)
        assert np.isclose(np.std(values, ddof=1), result["heads"][head_name]["weighted_f1_sample_std"], atol=1e-12)
        summaries[head_name] = values
        if head_name == "official_head":
            official_values = np.asarray(values)
        else:
            jev_values = np.asarray(values)
    comparison = result["comparison"]
    test = stats.ttest_rel(jev_values, official_values, alternative="greater")
    assert np.isclose(np.mean(jev_values - official_values), comparison["delta_weighted_f1_mean"], atol=1e-12)
    assert np.isclose(float(test.pvalue), comparison["one_sided_p_value"], atol=1e-12)
    assert comparison["significant_positive"] == bool(np.mean(jev_values - official_values) > 0 and test.pvalue < 0.05)
    print("EXTERNAL_QA_PASS", args.noise, json.dumps({
        "result_sha256": sha256(result_path),
        "training_data_card_sha256": protocol["data_card_sha256"],
        "current_validated_data_card_sha256": current_card_sha256,
        "delta_weighted_f1_mean": comparison["delta_weighted_f1_mean"],
        "one_sided_p_value": comparison["one_sided_p_value"],
        "significant_positive": comparison["significant_positive"],
        "cuda_device": torch.cuda.get_device_name(0),
        "gpu_reinference_runs": 10,
    }), flush=True)


if __name__ == "__main__":
    main()
