#!/usr/bin/env python3
"""RiNALMo Spliceator：官方 128 维 CLS 头与 JEV 头的五种子对照。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, "/home/rna_jev/tools")
import nb_splice_dual_round33 as base
from rinalmo_preflight import WEIGHTS, load_encoder, encode

DATA = base.DATA
SEEDS = base.SEEDS
SPECIES = base.SPECIES


class OfficialSpliceHead(nn.Module):
    def __init__(self, d=1280, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class JevSpliceHead(nn.Module):
    def __init__(self, d=1280, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_embeddings(df, cache: Path, device: str, batch_size: int) -> dict:
    fingerprint = base.sequence_sha256(df["sequences"])
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if ("sequence_sha256" in saved and str(saved["sequence_sha256"]) == fingerprint
                and len(saved["embs"]) == len(df)
                and np.array_equal(saved["labels"], df["labels"].to_numpy())):
            return {"embs": saved["embs"], "labels": saved["labels"]}
        raise RuntimeError(f"cache identity mismatch: {cache}")

    encoder, _ = load_encoder(device)
    embs = np.zeros((len(df), 1280), dtype=np.float16)
    sequences = df["sequences"].astype(str).str.upper().str.replace("T", "U", regex=False).tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            rows = [encode(seq, device=device) for seq in sequences[start:start + batch_size]]
            max_len = max(row.shape[1] for row in rows)
            ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
            mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for index, row in enumerate(rows):
                values = row[0]
                ids[index, :len(values)] = values
                mask[index, :len(values)] = True
            out = encoder(input_ids=ids, attention_mask=mask)
            hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            embs[start:start + len(rows)] = hidden[:, 0].float().cpu().numpy().astype(np.float16)
            if start % (batch_size * 250) == 0:
                print(f"[emb] {start + len(rows)}/{len(sequences)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embs=embs, labels=df["labels"].to_numpy(), sequence_sha256=fingerprint)
    temporary.replace(cache)
    return {"embs": embs, "labels": df["labels"].to_numpy()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ss-type", required=True, choices=["acceptor", "donor"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    df = base.load_frame(args.ss_type, args.limit)
    cache = DATA / f"{args.ss_type}_rinalmo_cls_v1{'_smoke' + str(args.limit) if args.limit else ''}.npz"
    arrays = extract_embeddings(df, cache, args.device, args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for head_name, head_cls in [("official_head", OfficialSpliceHead), ("jev_head", JevSpliceHead)]:
        runs = []
        for seed in SEEDS:
            run = base.train_one(arrays, df, head_cls, seed, args.device)
            pred = run.pop("pred_test")
            truth = run.pop("truth_test")
            best_state = run.pop("best_state")
            checkpoint = args.out_dir / f"{head_name}_s{seed}_best.pt"
            torch.save({"model_state_dict": best_state, "head": head_name, "seed": seed,
                        "best_epoch": run["best_epoch"], "best_validation_bce": run["best_validation_bce"]}, checkpoint)
            prediction = args.out_dir / f"{head_name}_s{seed}_predictions.npz"
            np.savez_compressed(prediction, predictions=pred, truth=truth)
            run.update({"checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256(checkpoint),
                        "prediction_file": str(prediction), "prediction_sha256": base.sha256(prediction)})
            runs.append(run)
            print(f"[{args.ss_type}/{head_name}/s{seed}] acc={run['mean_accuracy']:.5f} best={run['best_epoch']}", flush=True)
        outputs[head_name] = {"runs": runs,
                              "accuracy_mean": float(np.mean([run["mean_accuracy"] for run in runs])),
                              "accuracy_sample_std": float(np.std([run["mean_accuracy"] for run in runs], ddof=1))}
    official = np.array([run["mean_accuracy"] for run in outputs["official_head"]["runs"]])
    jev = np.array([run["mean_accuracy"] for run in outputs["jev_head"]["runs"]])
    test = stats.ttest_rel(jev, official, alternative="greater")
    p_value = float(test.pvalue)
    comparison = {"delta_accuracy_mean": float(np.mean(jev - official)),
                  "one_sided_p_value": p_value if np.isfinite(p_value) else None,
                  "statistics_status": "ok" if np.isfinite(p_value) else "undefined_zero_variance",
                  "significant_positive": bool(np.mean(jev - official) > 0 and np.isfinite(p_value) and p_value < 0.05)}
    payload = {"schema_version": 1, "status": "PASS",
               "protocol": {"task": f"RiNALMo Spliceator {args.ss_type}", "device": args.device,
                            "sample_scope": "full_dataset" if not args.limit else "engineering_smoke_subset_not_scientific_result",
                            "split": "Spliceator GS_1/db_1; four-species test sets; same split for both heads",
                            "convergence": "validation BCE; patience=32; min_delta=1e-5; max_epochs=2000 safety cap",
                            "single_variable": "head architecture on frozen RiNALMo CLS embeddings; official head embed_dim=128",
                            "data_card": str(DATA / "data_card.json"), "data_card_sha256": base.sha256(DATA / "data_card.json"),
                            "pretrained_weights": WEIGHTS,
                            "pretrained_sha256": base.sha256(Path(WEIGHTS) / "model.safetensors"),
                            "seeds": list(SEEDS), "optimizer": "Adam(lr=1e-3)", "train_batch_size": 512,
                            "embedding_batch_size": args.batch_size,
                            "code_sha256": base.sha256(Path(__file__).resolve()),
                            "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip()},
               "heads": outputs, "comparison": comparison}
    for head_name, head_cls in [("official_head", OfficialSpliceHead), ("jev_head", JevSpliceHead)]:
        for run in outputs[head_name]["runs"]:
            saved = np.load(run["prediction_file"], allow_pickle=False)
            test_species = df.iloc[df.index[df["split"] == "test"].to_numpy()]["species"].to_numpy()
            for species in SPECIES:
                mask = test_species == species
                accuracy = float(np.mean((saved["predictions"][mask] >= 0.5) == saved["truth"][mask]))
                assert np.isclose(accuracy, run["cells"][species]["accuracy"], atol=1e-10)
            assert base.sha256(Path(run["prediction_file"])) == run["prediction_sha256"]
            ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            model = head_cls()
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert ckpt["seed"] == run["seed"] and run["device"] == "cuda" and run["early_stopped"]
            assert len(run["validation_history"]) == run["epochs_run"]
            best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
            assert best_point["validation_bce"] == run["best_validation_bce"]
            run["independent_recheck"] = "PASS"
    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=base.json_default, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(out_path.read_text())["status"] == "PASS"
    print(f"RINALMO_SPLICE_DUAL_PASS type={args.ss_type} delta={comparison['delta_accuracy_mean']:+.6f} p={comparison['one_sided_p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
