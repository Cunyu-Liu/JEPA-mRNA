#!/usr/bin/env python3
"""NucleicBERT CPEB3 fitness: official linear probe versus JEV head, five seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

NB_REPO = Path("/root/NucleicBERT-1.0.0")
sys.path.insert(0, str(NB_REPO))
from nucleicbert.models.bert import BERT, NB_CONFIG
from transformers import PreTrainedTokenizerFast

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/nucleicbert/cpeb3"
PRETRAINED = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt")
SEEDS = (0, 1, 2, 3, 4)


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class OfficialFitnessHead(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class JevFitnessHead(nn.Module):
    def __init__(self, d=1024, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def tokenize(tokenizer, seq: str) -> list[int]:
    seq = seq.upper().replace("U", "T").replace("X", "N").replace("I", "N")
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq))
    return [tokenizer.vocab["[CLS]"], *ids, tokenizer.vocab["[SEP]"]]


def extract_embeddings(df: pd.DataFrame, cache: Path, device: str, batch_size: int) -> dict:
    fingerprint = sequence_sha256(df["sequences"])
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if ("sequence_sha256" in saved and str(saved["sequence_sha256"]) == fingerprint
                and len(saved["embs"]) == len(df)
                and np.array_equal(saved["fit_mean"], df["fit_mean"].to_numpy())
                and np.array_equal(saved["number_mutations"], df["number_mutations"].to_numpy())):
            return {key: saved[key] for key in ["embs", "fit_mean", "number_mutations"]}
        raise RuntimeError(f"cache identity mismatch: {cache}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(NB_REPO / "nucleicbert/tokenizers/noncoding_seqs.json"))
    model = BERT(**NB_CONFIG)
    model.load_state_dict(torch.load(PRETRAINED, map_location="cpu", weights_only=False))
    model = model.to(device).eval()
    embs = np.zeros((len(df), 1024), dtype=np.float16)
    sequences = df["sequences"].tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            rows = [tokenize(tokenizer, seq) for seq in sequences[start:start + batch_size]]
            max_len = max(map(len, rows))
            ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
            for index, row in enumerate(rows):
                ids[index, :len(row)] = torch.tensor(row, device=device)
            hidden, _, _ = model.encoder(ids, need_weights=False)
            embs[start:start + len(rows)] = hidden[:, 0].float().cpu().numpy().astype(np.float16)
            if start % (batch_size * 250) == 0:
                print(f"[emb] {start + len(rows)}/{len(sequences)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embs=embs, fit_mean=df["fit_mean"].to_numpy(),
                        number_mutations=df["number_mutations"].to_numpy(), sequence_sha256=fingerprint)
    temporary.replace(cache)
    return {"embs": embs, "fit_mean": df["fit_mean"].to_numpy(), "number_mutations": df["number_mutations"].to_numpy()}


def train_one(arrays: dict, split: dict, head_cls, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_idx = np.array(split["train"], dtype=int)
    valid_idx = np.array(split["validation"], dtype=int)
    test_idx = np.array(split["test"], dtype=int)
    X = torch.tensor(arrays["embs"], dtype=torch.float32, device=device)
    y = torch.tensor(arrays["fit_mean"], dtype=torch.float32, device=device)
    model = head_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
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
        validation_history.append({"epoch": epoch, "validation_mse": valid_loss})
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
        pred_valid = model(X[valid_idx]).cpu().numpy()
        pred_test = model(X[test_idx]).cpu().numpy()
    truth_valid = arrays["fit_mean"][valid_idx]
    truth_test = arrays["fit_mean"][test_idx]
    return {"seed": seed, "device": device, "converged": True, "early_stopped": True,
            "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_mse": best_loss,
            "validation_history": validation_history,
            "validation_r2": float(1 - np.sum((truth_valid - pred_valid) ** 2) /
                                   np.sum((truth_valid - truth_valid.mean()) ** 2)),
            "test_r2": float(1 - np.sum((truth_test - pred_test) ** 2) / np.sum((truth_test - truth_test.mean()) ** 2)),
            "test_pcc": float(stats.pearsonr(pred_test, truth_test)[0]),
            "n_train": len(train_idx), "n_validation": len(valid_idx), "n_test": len(test_idx),
            "pred_test": pred_test, "truth_test": truth_test, "best_state": best_state}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Engineering smoke only; not a scientific result")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval_decision/nb_cpeb3_round33")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    df = pd.read_csv(DATA / "cpeb3_all.csv")
    split = json.loads((DATA / "split.json").read_text())
    if args.limit:
        for key in ["train", "validation", "test"]:
            split[key] = split[key][:args.limit]
        df = df.iloc[sorted(split["train"] + split["validation"] + split["test"])].copy()
        remap = {old: new for new, old in enumerate(sorted(split["train"] + split["validation"] + split["test"]))}
        for key in ["train", "validation", "test"]:
            split[key] = [remap[index] for index in split[key]]
    cache = DATA / ("cpeb3_cls_v2.npz" if not args.limit else f"cpeb3_cls_smoke{args.limit}_v2.npz")
    arrays = extract_embeddings(df, cache, args.device, args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for head_name, head_cls in [("official_probe", OfficialFitnessHead), ("jev_head", JevFitnessHead)]:
        runs = []
        for seed in SEEDS:
            run = train_one(arrays, split, head_cls, seed, args.device)
            pred = run.pop("pred_test")
            truth = run.pop("truth_test")
            best_state = run.pop("best_state")
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            torch.save({"model_state_dict": best_state, "head": head_name, "seed": seed,
                        "best_epoch": run["best_epoch"], "best_validation_mse": run["best_validation_mse"]}, checkpoint)
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            np.savez_compressed(prediction, predictions=pred, truth=truth)
            run.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "prediction_file": str(prediction), "prediction_sha256": sha256(prediction)})
            runs.append(run)
            print(f"[{head_name}/s{seed}] R2={run['test_r2']:.5f} PCC={run['test_pcc']:.5f} best={run['best_epoch']}", flush=True)
        outputs[head_name] = {"runs": runs,
                              "r2_mean": float(np.mean([run["test_r2"] for run in runs])),
                              "r2_sample_std": float(np.std([run["test_r2"] for run in runs], ddof=1)),
                              "pcc_mean": float(np.mean([run["test_pcc"] for run in runs]))}
    official = np.array([run["test_r2"] for run in outputs["official_probe"]["runs"]])
    jev = np.array([run["test_r2"] for run in outputs["jev_head"]["runs"]])
    test = stats.ttest_rel(jev, official, alternative="greater")
    p_value = float(test.pvalue)
    comparison = {"delta_r2_mean": float(np.mean(jev - official)),
                  "paired_t": {"statistic": float(test.statistic) if np.isfinite(test.statistic) else None,
                               "pvalue": p_value if np.isfinite(p_value) else None,
                               "df": int(test.df)},
                  "statistics_status": "ok" if np.isfinite(p_value) else "undefined_zero_variance"}
    comparison["significant_positive"] = bool(comparison["delta_r2_mean"] > 0 and np.isfinite(p_value) and p_value < 0.05)
    payload = {"schema_version": 1, "status": "PASS",
               "protocol": {"task": "NucleicBERT CPEB3 fitness regression", "device": args.device,
                            "sample_scope": "full_dataset" if not args.limit else "engineering_smoke_subset_not_scientific_result",
                            "split": "frozen 64% train, 16% validation, 20% test; same split for both heads",
                            "convergence": "validation MSE; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
                            "single_variable": "head architecture on frozen NucleicBERT CLS embeddings",
                            "data_card": str(DATA / "data_card.json"), "data_card_sha256": sha256(DATA / "data_card.json"),
                            "pretrained_sha256": sha256(PRETRAINED),
                            "seeds": list(SEEDS), "optimizer": "Adam(lr=1e-3)", "train_batch_size": 512,
                            "embedding_batch_size": args.batch_size,
                            "code_sha256": sha256(Path(__file__).resolve()),
                            "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()},
               "heads": outputs, "comparison": comparison}
    # Independent saved-output checks.
    for head_name, head_cls in [("official_probe", OfficialFitnessHead), ("jev_head", JevFitnessHead)]:
        for run in outputs[head_name]["runs"]:
            saved = np.load(run["prediction_file"], allow_pickle=False)
            assert np.isclose(1 - np.sum((saved["truth"] - saved["predictions"]) ** 2) /
                              np.sum((saved["truth"] - saved["truth"].mean()) ** 2), run["test_r2"], atol=1e-10)
            assert sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            model = head_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert ckpt["seed"] == run["seed"] and run["device"] == "cuda" and run["early_stopped"]
            assert len(run["validation_history"]) == run["epochs_run"]
            best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
            assert best_point["validation_mse"] == run["best_validation_mse"]
            run["independent_recheck"] = "PASS"
    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
    checked = json.loads(out_path.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and len(checked["heads"]) == 2
    print(f"NB_CPEB3_DUAL_PASS delta={comparison['delta_r2_mean']:+.6f} p={comparison['paired_t']['pvalue']} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
