#!/usr/bin/env python3
"""T-A129：三组 MRL 的 5-seed 双头收敛评测与配对统计。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from eval_mrl_dual import JevMLP, PaperMLP, load_encoder, encode

W = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
TAR = Path("/root/rna_jepa_data/downstream_data/GSE114002.tar")
DATASETS = {
    "mcherry_1": {"member": "GSM3130441_mcherry_1.csv.gz", "rl_col": 23},
    "designed_library": {"member": "GSM3130443_designed_library.csv.gz", "rl_col": 32},
    "varying_length": {"member": "GSM4084997_varying_length_25to100.csv.gz", "rl_col": 31},
}
SEEDS = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(tar: tarfile.TarFile, spec: dict) -> pd.DataFrame:
    with tar.extractfile(spec["member"]) as handle:
        df = pd.read_csv(handle, compression="gzip", header=None,
                         usecols=[1, spec["rl_col"]], low_memory=False)
    df.columns = ["seq", "rl"]
    df = df.dropna()
    df = df[df["seq"].map(lambda s: isinstance(s, str))].copy()
    df["seq"] = df["seq"].str.upper()
    df = df[df["seq"].map(lambda s: set(s) <= set("ACGT"))].copy()
    df["len"] = df["seq"].str.len()
    df = df[(df["len"] >= 25) & (df["len"] <= 100)].reset_index(drop=True)
    df["rl"] = df["rl"].astype(float)
    assert len(df) > 0 and np.isfinite(df["rl"]).all()
    return df


def embed_dataset(df: pd.DataFrame, cache: Path, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        if len(d["embs"]) == len(df) and np.array_equal(d["rl"], df["rl"].to_numpy()) and np.array_equal(d["lens"], df["len"].to_numpy()):
            return d["embs"], d["rl"], d["lens"]
        raise ValueError(f"cache identity mismatch: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    seqs = df["seq"].tolist()
    embs = np.zeros((len(seqs), 1280), dtype=np.float16)
    encoder, _ = load_encoder(device)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(seqs), 64):
            chunk = seqs[start:start + 64]
            ids = [encode(seq, device=device) for seq in chunk]
            max_len = max(int(x.shape[-1]) for x in ids)
            batch = torch.zeros((len(ids), max_len), dtype=torch.int64, device=device)
            mask = torch.zeros((len(ids), max_len), dtype=torch.bool, device=device)
            for k, row in enumerate(ids):
                values = row[0]
                batch[k, :len(values)] = values
                mask[k, :len(values)] = True
            out = encoder(input_ids=batch, attention_mask=mask)
            hidden = (out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state).float()
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            embs[start:start + len(chunk)] = pooled.cpu().numpy().astype(np.float16)
            if start % 32000 == 0:
                print(f"[emb] {start + len(chunk)}/{len(seqs)} elapsed={time.time() - t0:.1f}s", flush=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embs=embs, rl=df["rl"].to_numpy(), lens=df["len"].to_numpy())
    temporary.replace(cache)
    return embs, df["rl"].to_numpy(), df["len"].to_numpy()


def split_indices(lens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(2020)
    valid, test = [], []
    for length in np.unique(lens):
        idx = np.where(lens == length)[0]
        assert len(idx) >= 200, f"insufficient examples for disjoint validation/test at length {length}: {len(idx)}"
        chosen = rng.choice(idx, size=200, replace=False)
        valid.extend(chosen[:100].tolist())
        test.extend(chosen[100:].tolist())
    valid_mask = np.zeros(len(lens), dtype=bool)
    test_mask = np.zeros(len(lens), dtype=bool)
    valid_mask[valid] = True
    test_mask[test] = True
    train_mask = ~(valid_mask | test_mask)
    assert train_mask.sum() + valid_mask.sum() + test_mask.sum() == len(lens)
    return train_mask, valid_mask, test_mask


def train_one(embs: np.ndarray, rls: np.ndarray, masks: tuple[np.ndarray, np.ndarray, np.ndarray],
              head_cls, seed: int, device: str) -> dict:
    train_mask, valid_mask, test_mask = masks
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = head_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    X_train = torch.tensor(embs[train_mask], dtype=torch.float32, device=device)
    y_train = torch.tensor(rls[train_mask], dtype=torch.float32, device=device)
    X_valid = torch.tensor(embs[valid_mask], dtype=torch.float32, device=device)
    y_valid = torch.tensor(rls[valid_mask], dtype=torch.float32, device=device)
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    max_epochs, patience, min_delta = 2000, 32, 1e-5
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train), device=device)
        for start in range(0, len(X_train), 512):
            idx = permutation[start:start + 512]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(X_valid), y_valid))
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
        pred_valid = model(X_valid).cpu().numpy()
        pred_test = model(torch.tensor(embs[test_mask], dtype=torch.float32, device=device)).cpu().numpy()
    validation_pcc = float(stats.pearsonr(pred_valid, rls[valid_mask])[0])
    test_pcc = float(stats.pearsonr(pred_test, rls[test_mask])[0])
    if not np.isfinite(validation_pcc) or not np.isfinite(test_pcc):
        raise RuntimeError(f"degenerate predictions: seed={seed} validation_pcc={validation_pcc} test_pcc={test_pcc}")
    return {
        "seed": seed, "device": device, "converged": True, "early_stopped": True,
        "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_mse": best_loss,
        "validation_pcc": validation_pcc,
        "test_pcc": test_pcc,
        "n_train": int(train_mask.sum()), "n_validation": int(valid_mask.sum()), "n_test": int(test_mask.sum()),
        "test_predictions": pred_test, "test_truth": rls[test_mask],
        "best_state": best_state,
    }


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=W / "eval_decision" / "mrl_suite")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar = tarfile.open(TAR)
    outputs = {}
    for dataset, spec in DATASETS.items():
        print(f"[dataset] {dataset}", flush=True)
        df = load_dataset(tar, spec)
        cache = W / "data" / "mrl_suite" / f"{dataset}.meanpool.npz"
        embs, rls, lens = embed_dataset(df, cache, args.device)
        assert len(embs) == len(df) == len(rls) == len(lens)
        masks = split_indices(lens)
        dataset_result = {"n_sequences": len(df), "embedding_cache": str(cache),
                          "embedding_cache_sha256": sha256(cache), "heads": {}}
        for head_name, head_cls in (("paper_head", PaperMLP), ("jev_head", JevMLP)):
            runs = []
            for seed in SEEDS:
                run = train_one(embs, rls, masks, head_cls, seed, args.device)
                predictions = run.pop("test_predictions")
                truth = run.pop("test_truth")
                best_state = run.pop("best_state")
                checkpoint_path = args.out_dir / f"{dataset}_{head_name}_s{seed}_best.pt"
                torch.save({"model_state_dict": best_state, "head": head_name,
                            "dataset": dataset, "seed": seed,
                            "best_epoch": run["best_epoch"],
                            "best_validation_mse": run["best_validation_mse"],
                            "single_variable": "head architecture"}, checkpoint_path)
                run["checkpoint"] = str(checkpoint_path)
                run["checkpoint_sha256"] = sha256(checkpoint_path)
                pred_path = args.out_dir / f"{dataset}_{head_name}_s{seed}_predictions.npz"
                np.savez_compressed(pred_path, predictions=predictions, truth=truth)
                run["prediction_file"] = str(pred_path)
                run["prediction_sha256"] = sha256(pred_path)
                runs.append(run)
                print(f"[{dataset}/{head_name}/s{seed}] PCC={run['test_pcc']:.4f} best_epoch={run['best_epoch']}", flush=True)
            dataset_result["heads"][head_name] = {
                "runs": runs,
                "pcc_mean": float(np.mean([run["test_pcc"] for run in runs])),
                "pcc_sample_std": float(np.std([run["test_pcc"] for run in runs], ddof=1)),
            }
        paper = np.array([run["test_pcc"] for run in dataset_result["heads"]["paper_head"]["runs"]])
        jev = np.array([run["test_pcc"] for run in dataset_result["heads"]["jev_head"]["runs"]])
        delta = jev - paper
        test = stats.ttest_rel(jev, paper, alternative="greater")
        dataset_result["comparison"] = {
            "delta_mean": float(delta.mean()), "delta_sample_std": float(delta.std(ddof=1)),
            "paired_t_statistic": float(test.statistic), "one_sided_p_value": float(test.pvalue),
            "test": "paired t-test across five pre-registered seeds; Holm correction across three datasets",
        }
        outputs[dataset] = dataset_result
    p_values = [record["comparison"]["one_sided_p_value"] for record in outputs.values()]
    for (dataset, record), adjusted in zip(outputs.items(), holm(p_values)):
        record["comparison"]["holm_p_value"] = adjusted
        record["comparison"]["significant_positive"] = bool(record["comparison"]["delta_mean"] > 0 and adjusted < 0.05)
    payload = {
        "schema_version": 1, "status": "PASS", "protocol": {
            "datasets": DATASETS, "seeds": list(SEEDS), "split": "fixed per-length 100 validation + 100 test; remainder train",
            "convergence": "best validation MSE; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
            "device": args.device, "single_variable": "head architecture on frozen RiNALMo-giga mean-pooled embeddings",
        },
        "results": outputs,
    }
    out_path = args.out_dir / "mrl_suite_result.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = json.loads(out_path.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and len(checked["results"]) == 3
    for dataset, record in checked["results"].items():
        comp = record["comparison"]
        print(f"RESULT {dataset} paper={record['heads']['paper_head']['pcc_mean']:.4f} "
              f"jev={record['heads']['jev_head']['pcc_mean']:.4f} delta={comp['delta_mean']:+.4f} "
              f"Holm={comp['holm_p_value']:.4g} positive={comp['significant_positive']}")
    print(f"MRL_SUITE_PASS out={out_path} sha256={sha256(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
