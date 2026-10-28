#!/usr/bin/env python3
"""CPEB3/Spliceator 完整结果的外部 GPU 复算与 checkpoint 重推理。"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
TOOLS = Path("/home/rna_jev/tools")
sys.path.insert(0, str(TOOLS))


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(model, x, regression: bool, batch_size: int = 4096) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            value = model(x[start:start + batch_size])
            outputs.append((value if regression else torch.sigmoid(value)).cpu().numpy())
    return np.concatenate(outputs)


def check_run(run: dict, seed: int, head: str, loss_key: str) -> None:
    assert run["seed"] == seed and run["device"] == "cuda"
    assert run["converged"] and run["early_stopped"] and run["independent_recheck"] == "PASS"
    history = run["validation_history"]
    assert len(history) == run["epochs_run"] == run["best_epoch"] + 32
    assert [point["epoch"] for point in history] == list(range(1, run["epochs_run"] + 1))
    assert all(np.isfinite(point[loss_key]) and point[loss_key] >= 0 for point in history)
    best = next(point for point in history if point["epoch"] == run["best_epoch"])
    assert best[loss_key] == run[f"best_validation_{loss_key.split('_')[1]}"]
    assert all(point[loss_key] >= run[f"best_validation_{loss_key.split('_')[1]}"] - 1e-5
               for point in history[run["best_epoch"]:])
    assert sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
    assert sha256(Path(run["checkpoint"])) == run["checkpoint_sha256"]


def load_model(run: dict, head: str, model_cls):
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["seed"] == run["seed"] and checkpoint["head"] == head
    assert checkpoint["best_epoch"] == run["best_epoch"]
    model = model_cls()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.cuda().eval(), checkpoint


def validate_card(protocol: dict) -> None:
    card_path = Path(protocol["data_card"])
    card = json.loads(card_path.read_text())
    assert card["status"] == "PASS"
    assert sha256(card_path) == protocol["data_card_sha256"]


def run_cpeb3(result_path: Path, result: dict) -> dict:
    module = importlib.import_module("nb_cpeb3_dual_round33")
    protocol = result["protocol"]
    assert sha256(Path(module.__file__)) == protocol["code_sha256"]
    validate_card(protocol)
    frame = pd.read_csv(module.DATA / "cpeb3_all.csv")
    split = json.loads((module.DATA / "split.json").read_text())
    arrays = module.extract_embeddings(frame, module.DATA / "cpeb3_cls_v2.npz", "cuda", 32)
    test_idx = np.asarray(split["test"], dtype=int)
    truth = arrays["fit_mean"][test_idx]
    x = torch.as_tensor(arrays["embs"][test_idx], dtype=torch.float32, device="cuda")
    values = {}
    for head, model_cls in (("official_probe", module.OfficialFitnessHead), ("jev_head", module.JevFitnessHead)):
        head_values = []
        runs = result["heads"][head]["runs"]
        assert [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        for run in runs:
            check_run(run, run["seed"], head, "validation_mse")
            with np.load(run["prediction_file"], allow_pickle=False) as saved:
                saved_predictions = saved["predictions"].copy()
                assert np.array_equal(saved["truth"], truth)
            model, checkpoint = load_model(run, head, model_cls)
            predictions = batches(model, x, True)
            assert np.allclose(predictions, saved_predictions, atol=1e-5, rtol=1e-5)
            r2 = float(1 - np.sum((truth - predictions) ** 2) / np.sum((truth - truth.mean()) ** 2))
            pcc = float(stats.pearsonr(predictions, truth)[0])
            assert np.isclose(r2, run["test_r2"], atol=1e-10)
            assert np.isclose(pcc, run["test_pcc"], atol=1e-10)
            head_values.append(r2)
            print(f"QA_RUN_PASS cpeb3 {head} seed={run['seed']} r2={r2:.10f}", flush=True)
            del model, checkpoint, predictions
            torch.cuda.empty_cache()
        assert np.isclose(np.mean(head_values), result["heads"][head]["r2_mean"], atol=1e-12)
        assert np.isclose(np.std(head_values, ddof=1), result["heads"][head]["r2_sample_std"], atol=1e-12)
        values[head] = np.asarray(head_values)
    delta = float(np.mean(values["jev_head"] - values["official_probe"]))
    test = stats.ttest_rel(values["jev_head"], values["official_probe"], alternative="greater")
    comparison = result["comparison"]
    assert np.isclose(delta, comparison["delta_r2_mean"], atol=1e-12)
    assert np.isclose(float(test.pvalue), comparison["paired_t"]["pvalue"], atol=1e-12)
    assert comparison["significant_positive"] == bool(delta > 0 and test.pvalue < 0.05)
    return {"delta": delta, "p_value": float(test.pvalue), "gpu_reinference_runs": 10}


def run_splice(result_path: Path, result: dict, family: str, ss_type: str, code_override: Path | None) -> dict:
    if family == "nb":
        module = importlib.import_module("nb_splice_dual_round33")
        official_name, official_cls = "official_probe", module.OfficialSpliceHead
        cache = module.DATA / f"{ss_type}_nb_cls_v2.npz"
    else:
        if code_override:
            spec = importlib.util.spec_from_file_location("rinalmo_splice_dual_round33_original", code_override)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module("rinalmo_splice_dual_round33")
        official_name, official_cls = "official_head", module.OfficialSpliceHead
        cache = module.DATA / f"{ss_type}_rinalmo_cls_v2.npz"
    protocol = result["protocol"]
    assert sha256(Path(module.__file__)) == protocol["code_sha256"]
    validate_card(protocol)
    frame = module.base.load_frame(ss_type, 0) if family == "rinalmo" else module.load_frame(ss_type, 0)
    assert cache.is_file(), f"required embedding cache is missing: {cache}; refusing implicit GPU re-extraction"
    arrays = module.extract_embeddings(frame, cache, "cuda", 32)
    test_idx = frame.index[frame["split"] == "test"].to_numpy()
    truth = arrays["labels"][test_idx]
    species = frame.iloc[test_idx]["species"].to_numpy()
    x = torch.as_tensor(arrays["embs"][test_idx], dtype=torch.float32, device="cuda")
    values = {}
    for head, model_cls in ((official_name, official_cls), ("jev_head", module.JevSpliceHead)):
        head_values = []
        runs = result["heads"][head]["runs"]
        assert [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        for run in runs:
            check_run(run, run["seed"], head, "validation_bce")
            with np.load(run["prediction_file"], allow_pickle=False) as saved:
                saved_predictions = saved["predictions"].copy()
                assert np.array_equal(saved["truth"], truth)
            model, checkpoint = load_model(run, head, model_cls)
            predictions = batches(model, x, False, batch_size=len(x))
            assert np.allclose(predictions, saved_predictions, atol=1e-5, rtol=1e-5)
            accuracies = []
            for name in module.SPECIES:
                mask = species == name
                accuracy = float(np.mean((predictions[mask] >= 0.5) == truth[mask]))
                assert np.isclose(accuracy, run["cells"][name]["accuracy"], atol=1e-12), (name, accuracy, run["cells"][name]["accuracy"], int(np.sum((predictions[mask] >= 0.5) != (saved_predictions[mask] >= 0.5))))
                assert int(mask.sum()) == run["cells"][name]["n"]
                accuracies.append(accuracy)
            mean_accuracy = float(np.mean(accuracies))
            assert np.isclose(mean_accuracy, run["mean_accuracy"], atol=1e-12)
            head_values.append(mean_accuracy)
            print(f"QA_RUN_PASS {family}_{ss_type} {head} seed={run['seed']} accuracy={mean_accuracy:.10f}", flush=True)
            del model, checkpoint, predictions
            torch.cuda.empty_cache()
        assert np.isclose(np.mean(head_values), result["heads"][head]["accuracy_mean"], atol=1e-12)
        assert np.isclose(np.std(head_values, ddof=1), result["heads"][head]["accuracy_sample_std"], atol=1e-12)
        values[head] = np.asarray(head_values)
    delta = float(np.mean(values["jev_head"] - values[official_name]))
    test = stats.ttest_rel(values["jev_head"], values[official_name], alternative="greater")
    comparison = result["comparison"]
    assert np.isclose(delta, comparison["delta_accuracy_mean"], atol=1e-12)
    assert np.isclose(float(test.pvalue), comparison["one_sided_p_value"], atol=1e-12)
    assert comparison["significant_positive"] == bool(delta > 0 and test.pvalue < 0.05)
    return {"delta": delta, "p_value": float(test.pvalue), "gpu_reinference_runs": 10}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--kind", choices=["cpeb3", "nb_splice", "rinalmo_splice"], required=True)
    parser.add_argument("--ss-type", choices=["acceptor", "donor"])
    parser.add_argument("--code-override", type=Path, help="结果协议声明的原始训练脚本")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required; refusing CPU validation"
    torch.cuda.set_device(0)
    result = json.loads(args.result.read_text())
    assert result["status"] == "PASS"
    protocol = result["protocol"]
    assert protocol["sample_scope"] == "full_dataset" and protocol["device"] == "cuda"
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    if args.kind == "cpeb3":
        summary = run_cpeb3(args.result, result)
    else:
        assert args.ss_type
        summary = run_splice(args.result, result, "nb" if args.kind == "nb_splice" else "rinalmo", args.ss_type, args.code_override)
    receipt = {"status": "PASS", "result": str(args.result), "result_sha256": sha256(args.result),
               "kind": args.kind, "ss_type": args.ss_type, "cuda_device": torch.cuda.get_device_name(0), **summary}
    output = args.result.parent / "qa_result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print("EXTERNAL_QA_PASS", json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
