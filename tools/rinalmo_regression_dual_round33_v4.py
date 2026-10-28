#!/usr/bin/env python3
"""RiNALMo official regression head versus JEV head with exact task protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/rna_jev/tools")
from rinalmo_preflight import WEIGHTS, load_encoder, encode

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
TAR = Path("/root/rna_jepa_data/downstream_data/GSE114002.tar")
SEEDS = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.dtype, pd.api.extensions.ExtensionDtype)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sequence_sha256(sequences) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(str(sequence).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class ResNet1DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim), nn.ELU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim), nn.ELU(inplace=True))
    def forward(self, x):
        return self.conv_net(x) + x


class OfficialRegressionHead(nn.Module):
    def __init__(self, c_in=1280, embed_dim=32, num_blocks=6, dropout=0.2):
        super().__init__()
        self.linear_in = nn.Linear(c_in, embed_dim)
        self.blocks = nn.ModuleList([ResNet1DBlock(embed_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(embed_dim, 1)
    def forward(self, x, padding_mask):
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = self.linear_in(x).permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x[padding_mask, :] = 0.0
        x = x.sum(dim=-2) / (~padding_mask).sum(dim=-1)[:, None]
        return self.linear_out(self.dropout(x)).squeeze(-1)


class JevRegressionHead(nn.Module):
    def __init__(self, c_in=1280, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c_in, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    def forward(self, x, padding_mask):
        weights = (~padding_mask).float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.net(pooled).squeeze(-1)


def top100(df: pd.DataFrame) -> set[int]:
    indices = set()
    for length in range(25, 101):
        subset = df[df["len"] == length].sort_values("total_reads", ascending=False)
        indices.update(subset.iloc[:100].index.tolist())
    return indices


def load_mrl() -> tuple[pd.DataFrame, dict]:
    with tarfile.open(TAR) as tar:
        with tar.extractfile("GSM4084997_varying_length_25to100.csv.gz") as handle:
            df = pd.read_csv(handle, compression="gzip")
    df = df.dropna(subset=["rl"]).copy()
    df = df[(df["set"] == "random") | (df["set"] == "human")].copy()
    df = df[df["total_reads"] >= 10].copy()
    df.drop_duplicates("utr", keep=False, inplace=True)
    df.reset_index(inplace=True)
    df["len"] = df["utr"].str.len()
    df = df[(df["len"] >= 25) & (df["len"] <= 100)].copy().reset_index(drop=True)
    assert df.index.equals(pd.RangeIndex(len(df)))
    random_df = df[df["set"] == "random"]
    human_df = df[df["set"] == "human"]
    valid_idx = top100(random_df)
    test_idx = top100(human_df)
    train_idx = set(random_df.index) - valid_idx
    assert not valid_idx & test_idx
    used = sorted(train_idx | valid_idx | test_idx)
    remap = {old: new for new, old in enumerate(used)}
    split = {"train": [remap[index] for index in sorted(train_idx)],
             "validation": [remap[index] for index in sorted(valid_idx)],
             "test": [remap[index] for index in sorted(test_idx)]}
    df = df.iloc[used].copy().reset_index(drop=True)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(df)
    assert df.index.equals(pd.RangeIndex(len(df)))
    return df, split


def load_cell(task: str, cell_line: str, fold: int) -> tuple[pd.DataFrame, dict]:
    path = ROOT / f"data/rinalmo/te_el_official/{cell_line}_sequence.csv"
    if not path.is_file():
        raise FileNotFoundError(f"official TE/EL data are still blocked: {path}")
    df = pd.read_csv(path)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(kfold.split(df))
    test_idx = splits[fold][1]
    valid_idx = splits[fold - 1][1]
    excluded = set(test_idx) | set(valid_idx)
    train_idx = sorted(set(range(len(df))) - excluded)
    split = {"train": train_idx, "validation": sorted(valid_idx), "test": sorted(test_idx)}
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(df)
    return df, split


def tokenize(seq: str, device: str) -> torch.Tensor:
    return encode(seq.upper().replace("T", "U"), device=device)


def extract_embeddings(df: pd.DataFrame, cache: Path, device: str, batch_size: int) -> dict:
    sequences = df["utr"].astype(str).str.replace("<pad>", "", regex=False).tolist()
    fingerprint = sequence_sha256(sequences)
    weights_sha256 = sha256(Path(WEIGHTS) / "model.safetensors")
    weights_config_sha256 = sha256(Path(WEIGHTS) / "config.json")
    preflight_sha256 = sha256(Path("/home/rna_jev/tools/rinalmo_preflight.py"))
    extractor_sha256 = sha256(Path(__file__).resolve())
    embs_path = cache.with_suffix(".embs.npy")
    mask_path = cache.with_suffix(".mask.npy")
    meta_path = cache.with_suffix(".json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (meta.get("sequence_sha256") != fingerprint or meta.get("n_sequences") != len(df)
                or meta.get("pretrained_sha256") != weights_sha256
                or meta.get("pretrained_config_sha256") != weights_config_sha256
                or meta.get("preflight_sha256") != preflight_sha256
                or meta.get("extractor_sha256") != extractor_sha256
                or meta.get("embedding_content_sha256") != sha256(embs_path)
                or meta.get("padding_mask_sha256") != sha256(mask_path)):
            raise RuntimeError(f"cache identity mismatch: {meta_path}")
        embs = np.load(embs_path, mmap_mode="r", allow_pickle=False)
        padding_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if (embs.shape != tuple(meta["embedding_shape"]) or padding_mask.shape != embs.shape[:2]
                or embs.dtype != np.dtype("float16") or padding_mask.dtype != np.dtype("bool")):
            raise RuntimeError(f"cache shape/dtype mismatch: {meta_path}")
        for start in range(0, len(df), 32):
            if not np.isfinite(embs[start:start + 32]).all():
                raise RuntimeError(f"non-finite embedding cache rows: {start}")
        return {"embs": embs, "padding_mask": padding_mask}

    encoder, _ = load_encoder(device)
    max_len = max(len(seq) for seq in sequences) + 2
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_embs = cache.with_suffix(".embs.tmp.npy")
    temporary_mask = cache.with_suffix(".mask.tmp.npy")
    embs = np.lib.format.open_memmap(temporary_embs, mode="w+", dtype=np.float16,
                                     shape=(len(sequences), max_len, 1280))
    padding_mask = np.lib.format.open_memmap(temporary_mask, mode="w+", dtype=bool,
                                             shape=(len(sequences), max_len))
    padding_mask[:] = True
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            rows = [tokenize(seq, device) for seq in sequences[start:start + batch_size]]
            ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
            mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for index, row in enumerate(rows):
                values = row[0]
                ids[index, :len(values)] = values
                mask[index, :len(values)] = True
            out = encoder(input_ids=ids, attention_mask=mask)
            hidden = (out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state).float()
            embs[start:start + len(rows)] = hidden.cpu().numpy().astype(np.float16)
            padding_mask[start:start + len(rows)] = ~mask.cpu().numpy()
            if start % (batch_size * 250) == 0:
                print(f"[emb] {start + len(rows)}/{len(sequences)}", flush=True)
    embs.flush()
    padding_mask.flush()
    for start in range(0, len(df), 32):
        if not np.isfinite(embs[start:start + 32]).all():
            raise RuntimeError(f"non-finite extracted embedding rows: {start}")
    del embs, padding_mask, encoder
    meta = {"sequence_sha256": fingerprint, "n_sequences": len(df),
            "pretrained_sha256": weights_sha256, "pretrained_config_sha256": weights_config_sha256,
            "preflight_sha256": preflight_sha256, "extractor_sha256": extractor_sha256,
            "embedding_content_sha256": sha256(temporary_embs),
            "padding_mask_sha256": sha256(temporary_mask),
            "embedding_shape": [len(sequences), max_len, 1280], "dtype": "float16"}
    temporary_embs.replace(embs_path)
    temporary_mask.replace(mask_path)
    temporary_meta = cache.with_suffix(".tmp.json")
    temporary_meta.write_text(json.dumps(meta, indent=1) + "\n")
    temporary_meta.replace(meta_path)
    return {"embs": np.load(embs_path, mmap_mode="r", allow_pickle=False),
            "padding_mask": np.load(mask_path, mmap_mode="r", allow_pickle=False)}


def train_one(arrays: dict, labels: np.ndarray, split: dict, head_cls, seed: int, device: str,
              scale: bool, huber: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_idx = np.array(split["train"], dtype=int)
    valid_idx = np.array(split["validation"], dtype=int)
    test_idx = np.array(split["test"], dtype=int)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    train_y = y[torch.as_tensor(train_idx, device=device)]
    mean = train_y.mean() if scale else torch.tensor(0.0, device=device)
    std = train_y.std(unbiased=False).clamp(min=1e-8) if scale else torch.tensor(1.0, device=device)
    model = head_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss() if huber else nn.MSELoss()

    def evaluate(indices):
        model.eval()
        predictions, loss_sum = [], 0.0
        with torch.no_grad():
            for start in range(0, len(indices), 512):
                idx = indices[start:start + 512]
                xb = torch.as_tensor(arrays["embs"][idx], dtype=torch.float32, device=device)
                mb = torch.as_tensor(arrays["padding_mask"][idx], dtype=torch.bool, device=device)
                raw = model(xb, mb)
                target = (y[torch.as_tensor(idx, device=device)] - mean) / std
                loss_sum += float(loss_fn(raw, target)) * len(idx)
                predictions.append(raw.cpu().numpy())
        return np.concatenate(predictions), loss_sum / len(indices)

    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    validation_history = []
    max_epochs, patience, min_delta = 2000, 32, 1e-5
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = np.random.permutation(train_idx)
        for start in range(0, len(permutation), 32):
            idx = permutation[start:start + 32]
            xb = torch.as_tensor(arrays["embs"][idx], dtype=torch.float32, device=device)
            mb = torch.as_tensor(arrays["padding_mask"][idx], dtype=torch.bool, device=device)
            target = (y[torch.as_tensor(idx, device=device)] - mean) / std
            optimizer.zero_grad()
            loss = loss_fn(model(xb, mb), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
        _, valid_loss = evaluate(valid_idx)
        validation_history.append({"epoch": epoch, "validation_loss": valid_loss})
        if valid_loss < best_loss - min_delta:
            best_loss, best_epoch, bad = valid_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        print(f"[seed={seed}/epoch={epoch}] validation_loss={valid_loss:.6f} best={best_loss:.6f}", flush=True)
    if best_state is None or bad < patience:
        raise RuntimeError(f"not converged: seed={seed} best_epoch={best_epoch} bad={bad}")
    model.load_state_dict(best_state)
    raw_valid, _ = evaluate(valid_idx)
    raw_test, _ = evaluate(test_idx)
    label_mean, label_std = float(mean), float(std)
    pred_valid = raw_valid * label_std + label_mean
    pred_test = raw_test * label_std + label_mean
    if scale:
        pred_valid = np.maximum(pred_valid, 0.0)
        pred_test = np.maximum(pred_test, 0.0)
    truth_valid, truth_test = labels[valid_idx], labels[test_idx]
    r2 = lambda truth, pred: float(1 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2))
    return {"seed": seed, "device": device, "converged": True, "early_stopped": True,
            "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_loss": best_loss,
            "validation_history": validation_history,
            "validation_r2": r2(truth_valid, pred_valid), "validation_pcc": float(stats.pearsonr(pred_valid, truth_valid)[0]),
            "validation_spearman": float(stats.spearmanr(pred_valid, truth_valid)[0]),
            "test_r2": r2(truth_test, pred_test), "test_pcc": float(stats.pearsonr(pred_test, truth_test)[0]),
            "test_spearman": float(stats.spearmanr(pred_test, truth_test)[0]),
            "n_train": len(train_idx), "n_validation": len(valid_idx), "n_test": len(test_idx),
            "label_mean": label_mean, "label_std": label_std,
            "pred_test": pred_test, "truth_test": truth_test, "best_state": best_state}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["mrl", "te", "el"])
    parser.add_argument("--cell-line", default="HEK", choices=["Muscle", "pc3", "HEK"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if args.task == "mrl":
        df, split = load_mrl()
        labels = df["rl"].to_numpy(dtype=float)
        scale, huber, cache = True, False, ROOT / "data/rinalmo/mrl_official_tokens_v3.npz"
        source_data = TAR
    else:
        df, split = load_cell(args.task, args.cell_line, args.fold)
        labels = df["te_log" if args.task == "te" else "rnaseq_log"].to_numpy(dtype=float)
        scale, huber = False, True
        cache = ROOT / f"data/rinalmo/{args.cell_line}_official_tokens.npz"
        source_data = ROOT / f"data/rinalmo/te_el_official/{args.cell_line}_sequence.csv"
    if args.limit:
        for key in split:
            split[key] = split[key][:args.limit]
        used = sorted(split["train"] + split["validation"] + split["test"])
        remap = {old: new for new, old in enumerate(used)}
        df = df.iloc[used].copy().reset_index(drop=True)
        labels = labels[used]
        for key in split:
            split[key] = [remap[index] for index in split[key]]
        cache = cache.with_name(cache.stem + f"_smoke{args.limit}.npz")
    arrays = extract_embeddings(df, cache, args.device, args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for head_name, head_cls in [("official_head", OfficialRegressionHead), ("jev_head", JevRegressionHead)]:
        runs = []
        for seed in SEEDS:
            run = train_one(arrays, labels, split, head_cls, seed, args.device, scale, huber)
            pred, truth, best_state = run.pop("pred_test"), run.pop("truth_test"), run.pop("best_state")
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            torch.save({"model_state_dict": best_state, "head": head_name, "seed": seed,
                        "best_epoch": run["best_epoch"], "best_validation_loss": run["best_validation_loss"]}, checkpoint)
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            np.savez_compressed(prediction, predictions=pred, truth=truth)
            run.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "prediction_file": str(prediction), "prediction_sha256": sha256(prediction)})
            runs.append(run)
            print(f"[{args.task}/{head_name}/s{seed}] R2={run['test_r2']:.5f} PCC={run['test_pcc']:.5f} best={run['best_epoch']}", flush=True)
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
    p_value = float(test.pvalue)
    spearman_p_value = float(spearman_test.pvalue)
    primary_test, primary_p = (test, p_value) if args.task == "mrl" else (spearman_test, spearman_p_value)
    primary_delta = (jev - official) if args.task == "mrl" else (jev_spearman - official_spearman)
    primary_delta_sd = float(np.std(primary_delta, ddof=1))
    statistics_ok = bool(primary_delta_sd > 0 and np.isfinite(primary_test.statistic) and np.isfinite(primary_p))
    comparison = {"primary_metric": "r2" if args.task == "mrl" else "spearman",
                  "delta_r2_mean": float(np.mean(jev - official)),
                  "delta_spearman_mean": float(np.mean(jev_spearman - official_spearman)),
                  "inference_scope": "paired training-seed effects; no test-set population inference",
                  "one_sided_p_value": p_value if np.isfinite(p_value) else None,
                  "one_sided_spearman_p_value": spearman_p_value if np.isfinite(spearman_p_value) else None,
                  "primary_delta_mean": float(np.mean(primary_delta)),
                  "primary_delta_sample_sd": primary_delta_sd,
                  "primary_one_sided_p_value": float(primary_test.pvalue) if np.isfinite(primary_p) else None,
                  "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
                  "significant_positive": bool(statistics_ok and np.mean(primary_delta) > 0 and primary_p < 0.05)}
    payload = {"schema_version": 1, "status": "PASS",
               "protocol": {"task": args.task, "cell_line": args.cell_line, "fold": args.fold,
                            "sample_scope": "full_official_protocol" if not args.limit else "engineering_smoke_subset_not_scientific_result",
                            "device": args.device, "single_variable": "head architecture on frozen RiNALMo token embeddings",
                            "convergence": "validation loss; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
                            "label_scaling": "train mean/std" if scale else "none",
                            "source_data": str(source_data), "source_data_sha256": sha256(source_data),
                            "pretrained_weights": WEIGHTS,
                            "pretrained_sha256": sha256(Path(WEIGHTS) / "model.safetensors"),
                            "pretrained_config": str(Path(WEIGHTS) / "config.json"),
                            "pretrained_config_sha256": sha256(Path(WEIGHTS) / "config.json"),
                            "preflight_sha256": sha256(Path("/home/rna_jev/tools/rinalmo_preflight.py")),
                            "embedding_cache_meta": str(cache.with_suffix(".json")),
                            "embedding_cache_meta_sha256": sha256(cache.with_suffix(".json")),
                            "seeds": list(SEEDS), "optimizer": "Adam(lr=1e-3)", "train_batch_size": 32,
                            "embedding_batch_size": args.batch_size,
                            "code_sha256": sha256(Path(__file__).resolve()),
                            "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()},
               "heads": outputs, "comparison": comparison}
    for head_name, head_cls in [("official_head", OfficialRegressionHead), ("jev_head", JevRegressionHead)]:
        for run in outputs[head_name]["runs"]:
            saved = np.load(run["prediction_file"], allow_pickle=False)
            expected = float(1 - np.sum((saved["truth"] - saved["predictions"]) ** 2) /
                             np.sum((saved["truth"] - saved["truth"].mean()) ** 2))
            assert np.isclose(expected, run["test_r2"], atol=1e-10)
            assert sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            model = head_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert ckpt["seed"] == run["seed"] and run["device"] == "cuda" and run["early_stopped"]
            assert len(run["validation_history"]) == run["epochs_run"]
            best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
            assert best_point["validation_loss"] == run["best_validation_loss"]
            run["independent_recheck"] = "PASS"
    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(out_path.read_text())["status"] == "PASS"
    print(f"RINALMO_REGRESSION_DUAL_PASS task={args.task} delta={comparison['delta_r2_mean']:+.6f} p={comparison['one_sided_p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
