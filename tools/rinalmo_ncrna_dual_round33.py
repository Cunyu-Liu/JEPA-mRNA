#!/usr/bin/env python3
"""RiNALMo 官方 88 类 ncRNA 头与 JEV 头的五种子对照。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/home/rna_jev/tools")
import rinalmo_splice_dual_round33 as shared

shared.sha256 = shared.base.sha256
shared.json_default = shared.base.json_default

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/rinalmo/ncrna_official"
SEEDS = (0, 1, 2, 3, 4)


class OfficialNcRNAHead(nn.Module):
    def __init__(self, d=1280, hidden=256, n_classes=88):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class JevNcRNAHead(nn.Module):
    def __init__(self, d=1280, hidden=1024, n_classes=88):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def load_frame(boundary_noise: str, limit: int) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = DATA / f"{split}{boundary_noise}.csv"
        frame = pd.read_csv(path)
        if limit:
            frame = frame.iloc[:limit].copy()
        frames.append(pd.DataFrame({
            "sequences": frame["sequence"].astype(str),
            "labels": frame["class_id"].astype(int),
            "split": {"train": "train", "val": "validation", "test": "test"}[split],
        }))
    df = pd.concat(frames, ignore_index=True)
    assert df["labels"].between(0, 87).all()
    return df


def train_one(arrays: dict, df: pd.DataFrame, head_cls, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_idx = df.index[df["split"] == "train"].to_numpy()
    valid_idx = df.index[df["split"] == "validation"].to_numpy()
    test_idx = df.index[df["split"] == "test"].to_numpy()
    X = torch.tensor(arrays["embs"], dtype=torch.float32, device=device)
    y = torch.tensor(arrays["labels"], dtype=torch.long, device=device)
    model = head_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    validation_history = []
    max_epochs, patience, min_delta = 2000, 32, 1e-5
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_idx), device=device)
        for start in range(0, len(train_idx), 512):
            idx = torch.tensor(train_idx[permutation[start:start + 512].cpu().numpy()], device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(X[valid_idx]), y[valid_idx]))
        validation_history.append({"epoch": epoch, "validation_ce": valid_loss})
        if valid_loss < best_loss - min_delta:
            best_loss, best_epoch, bad = valid_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None or bad < patience:
        raise RuntimeError(f"not converged: seed={seed} best_epoch={best_epoch} bad={bad}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_valid = model(X[valid_idx]).argmax(dim=-1).cpu().numpy()
        pred_test = model(X[test_idx]).argmax(dim=-1).cpu().numpy()
    truth_valid = arrays["labels"][valid_idx]
    truth_test = arrays["labels"][test_idx]
    return {"seed": seed, "device": device, "converged": True, "early_stopped": True,
            "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_ce": best_loss,
            "validation_history": validation_history,
            "validation_accuracy": float(accuracy_score(truth_valid, pred_valid)),
            "validation_weighted_f1": float(f1_score(truth_valid, pred_valid, average="weighted")),
            "accuracy": float(accuracy_score(truth_test, pred_test)),
            "weighted_f1": float(f1_score(truth_test, pred_test, average="weighted")),
            "n_train": len(train_idx), "n_validation": len(valid_idx), "n_test": len(test_idx),
            "pred_test": pred_test, "truth_test": truth_test, "best_state": best_state}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-noise", default="", choices=["", "_bn200"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    df = load_frame(args.boundary_noise, args.limit)
    cache_name = f"ncrna{args.boundary_noise or '_clean'}_rinalmo_cls_v1{'_smoke' + str(args.limit) if args.limit else ''}.npz"
    arrays = shared.extract_embeddings(df, cache=DATA / cache_name, device=args.device, batch_size=args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for head_name, head_cls in [("official_head", OfficialNcRNAHead), ("jev_head", JevNcRNAHead)]:
        runs = []
        for seed in SEEDS:
            run = train_one(arrays, df, head_cls, seed, args.device)
            pred = run.pop("pred_test")
            truth = run.pop("truth_test")
            best_state = run.pop("best_state")
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            torch.save({"model_state_dict": best_state, "head": head_name, "seed": seed,
                        "best_epoch": run["best_epoch"], "best_validation_ce": run["best_validation_ce"]}, checkpoint)
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            np.savez_compressed(prediction, predictions=pred, truth=truth)
            run.update({"checkpoint": str(checkpoint), "checkpoint_sha256": shared.sha256(checkpoint),
                        "prediction_file": str(prediction), "prediction_sha256": shared.sha256(prediction)})
            runs.append(run)
            print(f"[{args.boundary_noise or 'clean'}/{head_name}/s{seed}] "
                  f"F1={run['weighted_f1']:.5f} acc={run['accuracy']:.5f} best={run['best_epoch']}", flush=True)
        outputs[head_name] = {
            "runs": runs,
            "weighted_f1_mean": float(np.mean([run["weighted_f1"] for run in runs])),
            "weighted_f1_sample_std": float(np.std([run["weighted_f1"] for run in runs], ddof=1)),
            "accuracy_mean": float(np.mean([run["accuracy"] for run in runs])),
            "accuracy_sample_std": float(np.std([run["accuracy"] for run in runs], ddof=1))}
    official = np.array([run["weighted_f1"] for run in outputs["official_head"]["runs"]])
    jev = np.array([run["weighted_f1"] for run in outputs["jev_head"]["runs"]])
    test = stats.ttest_rel(jev, official, alternative="greater")
    p_value = float(test.pvalue)
    comparison = {
        "primary_metric": "weighted_f1",
        "delta_weighted_f1_mean": float(np.mean(jev - official)),
        "one_sided_p_value": p_value if np.isfinite(p_value) else None,
        "delta_accuracy_mean": float(np.mean([r["accuracy"] for r in outputs["jev_head"]["runs"]]) -
                                          np.mean([r["accuracy"] for r in outputs["official_head"]["runs"]])),
        "statistics_status": "ok" if np.isfinite(p_value) else "undefined_zero_variance",
        "significant_positive": bool(np.mean(jev - official) > 0 and np.isfinite(p_value) and p_value < 0.05)}
    payload = {"schema_version": 1, "status": "PASS",
               "protocol": {"task": f"RiNALMo official 88-class ncRNA {args.boundary_noise or 'clean'}",
                            "device": args.device,
                            "sample_scope": "full_dataset" if not args.limit else "engineering_smoke_subset_not_scientific_result",
                            "split": "official train/val/test CSV; identical split for both heads; boundary_noise=" + (args.boundary_noise or "0"),
                            "convergence": "validation cross-entropy; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
                            "single_variable": "head architecture on frozen RiNALMo CLS; official head embed_dim=256; n_classes=88",
                            "primary_metric": "weighted_f1; accuracy retained as secondary metric",
                            "data_card": str(DATA / "data_card.json"), "data_card_sha256": shared.sha256(DATA / "data_card.json"),
                            "pretrained_weights": shared.WEIGHTS,
                            "pretrained_sha256": shared.sha256(Path(shared.WEIGHTS) / "model.safetensors"),
                            "seeds": list(SEEDS), "optimizer": "Adam(lr=1e-3)", "train_batch_size": 512,
                            "embedding_batch_size": args.batch_size,
                            "code_sha256": shared.sha256(Path(__file__).resolve()),
                            "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()},
               "heads": outputs, "comparison": comparison}
    for head_name, head_cls in [("official_head", OfficialNcRNAHead), ("jev_head", JevNcRNAHead)]:
        for run in outputs[head_name]["runs"]:
            saved = np.load(run["prediction_file"], allow_pickle=False)
            assert np.isclose(accuracy_score(saved["truth"], saved["predictions"]), run["accuracy"], atol=1e-10)
            assert np.isclose(f1_score(saved["truth"], saved["predictions"], average="weighted"),
                              run["weighted_f1"], atol=1e-10)
            assert shared.sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            model = head_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert ckpt["seed"] == run["seed"] and run["device"] == "cuda" and run["early_stopped"]
            assert len(run["validation_history"]) == run["epochs_run"]
            best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
            assert best_point["validation_ce"] == run["best_validation_ce"]
            run["independent_recheck"] = "PASS"
    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=shared.json_default, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(out_path.read_text())["status"] == "PASS"
    print(f"RINALMO_NCRNA_DUAL_PASS noise={args.boundary_noise or '0'} "
          f"delta_f1={comparison['delta_weighted_f1_mean']:+.6f} p={comparison['one_sided_p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
