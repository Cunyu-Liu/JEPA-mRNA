#!/usr/bin/env python3
"""NucleicBERT original SSP: official head versus JEV head on identical frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_nucleicbert import NB_CODE  # noqa: E402
sys.path.insert(0, NB_CODE)
from nucleicbert.downstream.secstrmodule import SecStruct2DPredictionHead, prob_mat_to_sec_struct  # noqa: E402
from rnajepa.decision_head import FlatDecisionHead, pair_mask_from_ids, turner_phys_scores  # noqa: E402
from rnajepa.harness import nussinov_map  # noqa: E402
from rnajepa.train_decision import EmbeddingStore  # noqa: E402

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = ROOT / "data/nucleicbert_secondary_structure_round34"
EMBEDDINGS = DATA / "embeddings"
PRETRAINED = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt")
SPLIT_FILES = {"train": "train_rnalign_tr0", "validation": "vl0", "ts0": "ts0", "archiveii600": "archiveii600"}
MAX_CORE = 598
THRESHOLDS = tuple(i / 100 for i in range(10, 60, 5))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cache_split(role: str, limit: int) -> str:
    return f"smoke{limit}_{role}" if limit else SPLIT_FILES[role]


def load_records(role: str, limit: int) -> tuple[list[dict], dict]:
    path = DATA / "jsonl" / f"{SPLIT_FILES[role]}.jsonl"
    all_records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = all_records[:limit] if limit else all_records
    unique = list(dict.fromkeys(str(r["seq"]).upper().replace("T", "U") for r in all_records))
    if limit:
        unique = unique[:limit]
    unique.sort(key=lambda s: (len(s), s))
    jsonl_sha256 = sha256(path)
    expected_fingerprint = {
        "jsonl": str(path), "jsonl_sha256": jsonl_sha256,
        "sequence_sha256": hashlib.sha256("".join(s + "\n" for s in unique).encode()).hexdigest(),
        "n_input_records": len(all_records), "n_unique_sequences": len(unique), "max_length": 600,
        "pretrained": str(PRETRAINED), "pretrained_sha256": sha256(PRETRAINED),
        "code_sha256": sha256(Path("/home/rna_jev/tools/extract_nb_secstr_round34.py")), "limit": limit,
    }
    manifest = json.loads((EMBEDDINGS / "manifest.json").read_text())
    entries = [e for e in manifest["entries"] if e["split"] == cache_split(role, limit)]
    if not entries:
        raise RuntimeError(f"no manifest entries for {cache_split(role, limit)}")
    cached_sequences = 0
    for entry in entries:
        shard = Path(entry["path"])
        metadata = json.loads(shard.with_suffix(".json").read_text())
        if metadata.get("fingerprint") != expected_fingerprint:
            raise RuntimeError(f"embedding fingerprint mismatch: {shard}")
        if metadata.get("sha256") != entry.get("sha256") or sha256(shard) != entry["sha256"]:
            raise RuntimeError(f"embedding shard hash mismatch: {shard}")
        cached_sequences += int(entry["n_sequences"])
    if cached_sequences != len(unique):
        raise RuntimeError(f"embedding sequence count mismatch for {cache_split(role, limit)}")
    required = {r["seq"] for r in records}
    store = EmbeddingStore.from_dir(str(EMBEDDINGS), split=cache_split(role, limit), required_sequences=required)
    for record in records:
        record["embedding"] = store.get(record["seq"])
        if len(record["embedding"]) != len(record["seq"]):
            raise RuntimeError(f"embedding length mismatch: {record['name']}")
    return records, {"jsonl": str(path), "jsonl_sha256": jsonl_sha256, "cache_entries": entries}


def build_head(name: str, device: str):
    if name == "official":
        return SecStruct2DPredictionHead(1024, num_blocks=1, dropout=0.1).to(device)
    return FlatDecisionHead(
        d_model=1024, d_z=128, hidden=64, scorer="resnet2d", resnet_blocks=4,
        resnet_kernel_size=3, chunk_size=64, grad_checkpoint=True,
    ).to(device)


def ids_for(sequence: str, length: int, device: str) -> torch.Tensor:
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    ids = [mapping.get(char, 4) for char in sequence[:length]]
    ids.extend([4] * (length - len(ids)))
    return torch.tensor([ids], dtype=torch.long, device=device)


def jev_scores(head, embeddings: torch.Tensor, seq_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    chunks = []
    width = embeddings.shape[1]
    for start in range(0, width, 128):
        chunks.append(head.pair_repr.cross(embeddings, embeddings[:, start:start + 128]))
    pair = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    scores = head.resnet2d(pair)
    scores = scores + head.prior_weight.to(scores.dtype) * turner_phys_scores(seq_ids).to(embeddings.device)
    return head.calibration(scores, lengths)


def forward(head_name: str, head, embeddings: torch.Tensor, seq_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if head_name == "official":
        return head(embeddings)
    return jev_scores(head, embeddings, seq_ids, lengths)


def collate(records: list[dict], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    embeddings = torch.zeros((batch, MAX_CORE, 1024), dtype=torch.float32)
    targets = torch.zeros((batch, MAX_CORE, MAX_CORE), dtype=torch.float32)
    seq_ids = torch.full((batch, MAX_CORE), 4, dtype=torch.long)
    lengths = torch.tensor([len(r["seq"]) for r in records], dtype=torch.long)
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    for row, record in enumerate(records):
        length = len(record["seq"])
        embeddings[row, :length] = torch.from_numpy(record["embedding"]).float()
        seq_ids[row, :length] = torch.tensor([mapping.get(c, 4) for c in record["seq"]], dtype=torch.long)
        for i, j in record["pairs"]:
            targets[row, i, j] = targets[row, j, i] = 1.0
    return embeddings.to(device), targets.to(device), seq_ids.to(device), lengths.to(device)


def loss_train(head_name, head, records, device):
    embeddings, targets, seq_ids, lengths = collate(records, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = forward(head_name, head, embeddings, seq_ids, lengths)
    upper = torch.triu(torch.ones(MAX_CORE, MAX_CORE, dtype=torch.bool, device=device), diagonal=1)
    loss = F.binary_cross_entropy_with_logits(logits[upper], targets[upper])
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    return loss


def validation_loss(head_name, head, records, device, batch_size):
    total = 0.0
    count = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            embeddings, targets, seq_ids, lengths = collate(records[start:start + batch_size], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward(head_name, head, embeddings, seq_ids, lengths)
            upper = torch.triu(torch.ones(MAX_CORE, MAX_CORE, dtype=torch.bool, device=device), diagonal=1)
            for row, length in enumerate(lengths.tolist()):
                valid = torch.zeros((MAX_CORE, MAX_CORE), dtype=torch.bool, device=device)
                valid[:length, :length] = True
                mask = upper & valid
                total += float(F.binary_cross_entropy_with_logits(logits[row][mask], targets[row][mask], reduction="sum"))
                count += int(mask.sum())
    return total / count


def pair_metrics(pred: list[list[int]], truth: list[list[int]]) -> dict:
    predicted = {tuple(p) for p in pred}
    expected = {tuple(p) for p in truth}
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall >= 1e-5 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
            "fp": len(predicted) - tp, "fn": len(expected) - tp}


def score_one(head_name, head, record, device):
    length = len(record["seq"])
    embeddings = torch.from_numpy(record["embedding"]).float().unsqueeze(0).to(device)
    seq_ids = ids_for(record["seq"], length, device)
    lengths = torch.tensor([length], device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = forward(head_name, head, embeddings, seq_ids, lengths)
    return scores[0].float().cpu().numpy(), seq_ids


def predict(head_name, head, records, device, threshold=None, scored=None):
    if scored is None:
        scored = [score_one(head_name, head, record, device) for record in records]
    if len(scored) != len(records):
        raise ValueError("scored/record length mismatch")
    rows = []
    for index, (record, (scores, seq_ids)) in enumerate(zip(records, scored)):
        if head_name == "official":
            matrix = prob_mat_to_sec_struct(expit(scores), record["seq"], threshold=float(threshold))
            predicted = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(matrix, k=1)))]
        else:
            mask = pair_mask_from_ids(seq_ids).numpy()[0]
            predicted = [[int(i), int(j)] for i, j in nussinov_map(np.where(mask, scores, -np.inf), mask)]
        metrics = pair_metrics(predicted, record["pairs"])
        rows.append({"name": record["name"], "length": len(record["seq"]),
                     "pred_pairs": predicted, "gt_pairs": record["pairs"], **metrics})
        if (index + 1) % 250 == 0:
            print(f"[predict/{head_name}] {index+1}/{len(records)}", flush=True)
    return rows


def save_predictions(path: Path, split: str, rows: list[dict]) -> dict:
    atomic_text(path, "".join(json.dumps({"split": split, **row}, separators=(",", ":")) + "\n" for row in rows))
    loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    recomputed = [pair_metrics(row["pred_pairs"], row["gt_pairs"]) for row in loaded]
    for original, check in zip(loaded, recomputed):
        for key in ("precision", "recall", "f1"):
            if not math.isclose(original[key], check[key], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"prediction metric mismatch: {path}")
    return {"path": str(path), "sha256": sha256(path), "rows": len(loaded),
            "macro_f1": float(np.mean([r["f1"] for r in loaded])),
            "macro_precision": float(np.mean([r["precision"] for r in loaded])),
            "macro_recall": float(np.mean([r["recall"] for r in loaded]))}


def train(args) -> int:
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "run.json"
    if result_path.is_file():
        old = json.loads(result_path.read_text())
        if old.get("status") == "PASS":
            print(f"SKIP_PASS {result_path}", flush=True)
            return 0
        raise RuntimeError(f"previous non-PASS run requires manual audit: {result_path}")
    train_records, train_evidence = load_records("train", args.limit)
    valid_records, valid_evidence = load_records("validation", args.limit)
    run_identity = {
        "code_sha256": sha256(Path(__file__).resolve()), "data_card_sha256": sha256(DATA / "data_card.json"),
        "pretrained_sha256": sha256(PRETRAINED), "batch_size": args.batch_size, "lr": args.lr,
        "patience": args.patience, "min_delta": args.min_delta, "max_epochs": args.max_epochs,
    }
    head = build_head(args.head, args.device)
    trainable = [(n, p) for n, p in head.named_parameters() if args.head == "official" or not n.startswith("type_head")]
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    resume_path = args.out_dir / "resume.pt"
    start_epoch, best_loss, best_epoch, best_state, bad, history = 0, math.inf, 0, None, 0, []
    coverage = None
    early_stopped = False
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["head"] != args.head or saved["seed"] != args.seed or saved["limit"] != args.limit:
            raise RuntimeError("resume identity mismatch")
        if saved.get("run_identity") != run_identity:
            raise RuntimeError("resume code/data/hyperparameter identity mismatch")
        if saved.get("train_evidence") != train_evidence or saved.get("valid_evidence") != valid_evidence:
            raise RuntimeError("resume train/validation cache evidence mismatch")
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"]
        best_loss, best_epoch, best_state, bad, history = saved["best_loss"], saved["best_epoch"], saved["best_state"], saved["bad"], saved["history"]
        coverage = saved.get("coverage")
        early_stopped = bool(saved["early_stopped"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        print(f"RESUME epoch={start_epoch} best={best_loss:.8f} early_stopped={early_stopped}", flush=True)
    elif any(args.out_dir.iterdir()):
        raise RuntimeError(f"non-empty run directory has no resumable checkpoint: {args.out_dir}")
    max_epochs = args.max_epochs
    for epoch in range(start_epoch + 1, max_epochs + 1):
        if early_stopped:
            break
        head.train()
        generator = torch.Generator().manual_seed(args.seed + epoch * 100003)
        order = torch.randperm(len(train_records), generator=generator).tolist()
        epoch_loss, batches = 0.0, 0
        started = time.time()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_train(args.head, head, [train_records[i] for i in indices], args.device)
            loss.backward()
            if coverage is None:
                missing = [n for n, p in trainable if p.grad is None]
                nonfinite = [n for n, p in trainable if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                if missing or nonfinite:
                    raise RuntimeError(f"gradient coverage failure missing={missing} nonfinite={nonfinite}")
                coverage = {"covered": [n for n, _ in trainable], "missing": [], "nonfinite": [],
                            "excluded": [] if args.head == "official" else ["type_head (not used by pair-existence objective)"]}
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
            if batches % args.log_every == 0:
                print(f"[{args.head}/s{args.seed}/epoch{epoch}] step={batches} train_bce={epoch_loss/batches:.7f} elapsed={time.time()-started:.0f}s", flush=True)
        current = validation_loss(args.head, head, valid_records, args.device, args.batch_size)
        history.append({"epoch": epoch, "train_bce": epoch_loss / batches, "validation_bce": current})
        print(f"[{args.head}/s{args.seed}/epoch{epoch}] validation_bce={current:.8f} best={min(best_loss,current):.8f}", flush=True)
        if current < best_loss - args.min_delta:
            best_loss, best_epoch, bad = current, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            bad += 1
        if args.limit == 0 and bad >= args.patience:
            early_stopped = True
        atomic_torch(resume_path, {
            "model": head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best_loss": best_loss, "best_epoch": best_epoch, "best_state": best_state, "bad": bad,
            "history": history, "coverage": coverage, "head": args.head, "seed": args.seed,
            "limit": args.limit, "early_stopped": early_stopped, "run_identity": run_identity,
            "train_evidence": train_evidence, "valid_evidence": valid_evidence,
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        })
        atomic_text(args.out_dir / "validation_history.jsonl", "".join(json.dumps(row) + "\n" for row in history))
        if early_stopped:
            break
    if best_state is None or (args.limit == 0 and not early_stopped):
        payload = {"status": "NOT_CONVERGED", "head": args.head, "seed": args.seed,
                   "epochs_run": len(history), "best_epoch": best_epoch, "best_validation_bce": best_loss}
        atomic_json(result_path, payload)
        raise RuntimeError(f"training did not reach validation plateau: {payload}")
    head.load_state_dict(best_state)
    head.eval()
    checkpoint = args.out_dir / "best.pt"
    protocol = {
        "task": "NucleicBERT original secondary-structure prediction",
        "single_variable": "head architecture and its native decoder on identical frozen NucleicBERT embeddings, labels, BCE, optimizer and splits",
        "train": "RNAStrAlign 37,149 + TR0 10,814; all labels including pseudoknots",
        "validation": "VL0 1,300; full-set BCE for early stopping; official threshold selected on full VL0 macro-F1",
        "tests": ["TS0 1,305", "ArchiveII600 3,911"],
        "preprocessing": "official 598 nt truncation; X/I->N; U->T only during tokenization; fixed-598 training BCE as official implementation; valid-position validation BCE",
        "optimizer": f"AdamW(lr={args.lr}, weight_decay=0); no scheduler; batch_size={args.batch_size}; bf16 autocast; TF32 disabled",
        "convergence": f"full VL0 BCE; patience={args.patience}; min_delta={args.min_delta}; safety_cap={max_epochs}; cap is not convergence",
        "decoder": "official canonical/no-sharp-loop greedy conflict cleanup" if args.head == "official" else "JEV canonical/min-loop non-crossing max-product Nussinov DP",
        "metric": "per-structure upper-triangle precision/recall/F1, macro-averaged exactly as NucleicBERT ss_metrics",
        "seeds": args.seed, "device": args.device,
        "scope": "engineering_smoke_not_scientific" if args.limit else "full_dataset",
        "scientific_eligibility": bool(args.limit == 0),
    }
    atomic_torch(checkpoint, {"head": args.head, "seed": args.seed, "model_state": best_state,
                              "best_epoch": best_epoch, "best_validation_bce": best_loss,
                              "protocol": protocol, "code_sha256": sha256(Path(__file__).resolve())})
    predictions = {}
    if args.head == "official":
        valid_scored = [score_one(args.head, head, record, args.device) for record in valid_records]
        candidate_rows = {t: predict(args.head, head, valid_records, args.device, t, valid_scored) for t in THRESHOLDS}
        threshold_scores = {str(t): float(np.mean([r["f1"] for r in rows])) for t, rows in candidate_rows.items()}
        threshold = max(THRESHOLDS, key=lambda t: threshold_scores[str(t)])
        predictions["validation"] = save_predictions(args.out_dir / "validation_predictions.jsonl", "validation", candidate_rows[threshold])
    else:
        threshold = None
        threshold_scores = None
        predictions["validation"] = save_predictions(args.out_dir / "validation_predictions.jsonl", "validation",
                                                       predict(args.head, head, valid_records, args.device))
    test_evidence = {}
    for role in ("ts0", "archiveii600"):
        records, evidence = load_records(role, args.limit)
        test_evidence[role] = evidence
        predictions[role] = save_predictions(args.out_dir / f"{role}_predictions.jsonl", role,
                                               predict(args.head, head, records, args.device, threshold))
    payload = {
        "schema_version": 1, "status": "PASS", "head": args.head, "seed": args.seed,
        "converged": bool(args.limit == 0), "early_stopped": early_stopped,
        "scientific_eligibility": bool(args.limit == 0), "limit": args.limit,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_validation_bce": best_loss,
        "gradient_coverage": coverage, "threshold": threshold, "threshold_validation_f1": threshold_scores,
        "protocol": protocol, "data": {"train": train_evidence, "validation": valid_evidence, **test_evidence},
        "data_card": str(DATA / "data_card.json"), "data_card_sha256": run_identity["data_card_sha256"],
        "pretrained_sha256": run_identity["pretrained_sha256"], "code_sha256": run_identity["code_sha256"],
        "code_commit": subprocess.check_output(["git", "-C", "/home/rna_jev", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "predictions": predictions, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(result_path, payload)
    if resume_path.is_file():
        resume_path.replace(args.out_dir / "resume.completed.pt")
    print(f"NB_SECSTR_RUN_PASS {args.head} seed={args.seed} " + " ".join(f"{k}={v['macro_f1']:.6f}" for k, v in predictions.items()), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, choices=["official", "jev"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.limit and args.max_epochs == 1000:
        args.max_epochs = 2
    if args.batch_size <= 0 or args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, max epochs and patience must be positive")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
