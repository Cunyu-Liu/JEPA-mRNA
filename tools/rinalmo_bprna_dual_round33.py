#!/usr/bin/env python3
"""Frozen-RiNALMo bpRNA dual-head control: official head/decoder versus JEV."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/home/rna_jev/src")
sys.path.insert(0, "/home/rna_jev/tools")
sys.path.insert(0, "/home/rna_jev/tools/flash_shim")
sys.path.insert(0, "/home/rna_jev/third_party/RiNALMo")
from rinalmo_preflight import WEIGHTS, encode, load_encoder  # noqa: E402
from rinalmo.model.downstream import SecStructPredictionHead  # noqa: E402
from rinalmo.utils.sec_struct import (  # noqa: E402
    prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall,
)
from rnajepa.decision_head import (  # noqa: E402
    FlatDecisionHead, pair_mask_from_ids, turner_phys_scores,
)
from dp_decode_c_wrapper import nussinov_map_single  # noqa: E402
from rnajepa.harness import _max_dp_reference, _traceback_max  # noqa: E402

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/bpRNA_spot0")
OFFICIAL_REPO = Path("/home/rna_jev/third_party/RiNALMo")
EXPECTED_OFFICIAL_COMMIT = "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"
EXPECTED_DATA_CARD_SHA256 = "dc370656ebce5d4ed1941dae07c31fc387025ce5d0e38feb4c09f343c4f1c9d0"
EXPECTED_WEIGHTS_SHA256 = "0b3f73b5c80d267f0b7e92655b28f16088e3d5a595c6dae631b52142aa38bcda"
FAMILIES = ("TS0",)
SPLITS = ("train", "valid", "test")
SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = tuple(i / 100 for i in range(1, 30))
PATIENCE, MIN_DELTA, MAX_EPOCHS = 32, 1e-5, 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def model_sequence(row: dict) -> str:
    return row["sequence"].upper().replace("T", "U")


def sequence_ids(sequence: str, device: str | torch.device = "cpu") -> torch.Tensor:
    alphabet = {"A": 0, "C": 1, "G": 2, "U": 3}
    return torch.tensor([[alphabet.get(base, 4) for base in sequence]],
                        dtype=torch.long, device=device)


def load_fold(family: str, card: dict):
    rows, split_indices, paths, jsonl_sha256 = [], {}, {}, {}
    for split in SPLITS:
        path = ROOT / "jsonl" / family / f"{split}.jsonl"
        actual_sha = sha256(path)
        key = f"{family}/{split}"
        if actual_sha != card["jsonl_sha256"][key]:
            raise RuntimeError(f"JSONL does not match data card: {path}")
        paths[split], jsonl_sha256[split] = path, actual_sha
        start = len(rows)
        ids = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {"id", "family", "held_out_family", "split", "sequence", "pairs", "source_ct"}
                if not required.issubset(row):
                    raise RuntimeError(f"missing fields at {path}:{line_number}")
                if row["family"] != family or row["held_out_family"] != family or row["split"] != split:
                    raise RuntimeError(f"fold metadata mismatch at {path}:{line_number}")
                sequence = row["sequence"]
                if not sequence or len(sequence) > 512 or row["id"] in ids:
                    raise RuntimeError(f"invalid length or duplicate id at {path}:{line_number}")
                ids.add(row["id"])
                pairs = []
                for pair in row["pairs"]:
                    if len(pair) != 2:
                        raise RuntimeError(f"invalid pair at {path}:{line_number}")
                    i, j = int(pair[0]), int(pair[1])
                    if not 0 <= i < j < len(sequence):
                        raise RuntimeError(f"out-of-range pair at {path}:{line_number}")
                    pairs.append([i, j])
                if len({tuple(pair) for pair in pairs}) != len(pairs):
                    raise RuntimeError(f"duplicate pair at {path}:{line_number}")
                row["pairs"] = sorted(pairs)
                rows.append(row)
        if len(rows) == start:
            raise RuntimeError(f"empty official split: {path}")
        split_indices[split] = list(range(start, len(rows)))
    return rows, split_indices, paths, jsonl_sha256


def fold_sequence_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    digest.update(b"split\\0id\\0sequence\\n")
    for row in rows:
        digest.update(row["split"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sequence"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_embedding_finite(embs: np.ndarray) -> None:
    for start in range(0, embs.shape[0], 32):
        if not np.isfinite(np.asarray(embs[start:start + 32])).all():
            raise RuntimeError(f"non-finite embedding cache values in rows {start}:{start + 32}")


def extract_embeddings(rows: list[dict], cache: Path, device: str, batch_size: int,
                       weights_sha256: str, weights_config_sha256: str,
                       preflight_sha256: str, jsonl_sha256: dict) -> np.ndarray:
    sequences = [model_sequence(row) for row in rows]
    shape = [len(rows), max(map(len, sequences)) + 2, 1280]
    expected = {
        "cache_schema_version": 2,
        "sequence_sha256": fold_sequence_sha256(rows),
        "sequence_sha256_algorithm": "ordered split\\0id\\0sequence rows; full train/valid/test fold",
        "n_sequences": len(rows),
        "jsonl_sha256": jsonl_sha256,
        "pretrained_sha256": weights_sha256,
        "pretrained_config_sha256": weights_config_sha256,
        "preflight_sha256": preflight_sha256,
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "embedding_shape": shape,
        "dtype": "float16",
        "token_embeddings_include_bos_eos": True,
    }
    embs_path = cache.with_suffix(".embs.npy")
    meta_path = cache.with_suffix(".json")
    content_sha_key = "embedding_content_sha256"
    if meta_path.is_file():
        if not embs_path.is_file():
            raise RuntimeError(f"embedding cache metadata exists without embeddings: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or set(metadata) != set(expected) | {content_sha_key}
                or any(metadata.get(key) != value for key, value in expected.items())):
            raise RuntimeError(f"old, incomplete, or mismatched embedding cache: {meta_path}")
        if metadata[content_sha_key] != sha256(embs_path):
            raise RuntimeError(f"embedding cache content SHA256 mismatch: {embs_path}")
        embs = np.load(embs_path, mmap_mode="r", allow_pickle=False)
        if list(embs.shape) != shape or embs.dtype != np.dtype("float16"):
            raise RuntimeError(f"embedding cache shape/dtype mismatch: {embs_path}")
        validate_embedding_finite(embs)
        return embs
    if embs_path.exists():
        raise RuntimeError(f"cache embeddings exist without valid metadata: {embs_path}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".embs.tmp.npy")
    embs = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=tuple(shape))
    encoder, _ = load_encoder(device)
    max_len = shape[1]
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_sequences = sequences[start:start + batch_size]
            token_rows = [encode(sequence, device=device)[0] for sequence in batch_sequences]
            ids = torch.zeros((len(token_rows), max_len), dtype=torch.long, device=device)
            attention = torch.zeros((len(token_rows), max_len), dtype=torch.bool, device=device)
            for index, tokens in enumerate(token_rows):
                ids[index, :len(tokens)] = tokens
                attention[index, :len(tokens)] = True
            output = encoder(input_ids=ids, attention_mask=attention)
            hidden = (output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state).float()
            embs[start:start + len(token_rows)] = hidden.cpu().numpy().astype(np.float16)
            if start % (batch_size * 250) == 0:
                print(f"[embedding] {start + len(token_rows)}/{len(rows)}", flush=True)
    validate_embedding_finite(embs)
    embs.flush()
    del embs, encoder
    torch.cuda.empty_cache()
    expected[content_sha_key] = sha256(temporary)
    temporary.replace(embs_path)
    temporary_meta = cache.with_suffix(".tmp.json")
    temporary_meta.write_text(json.dumps(expected, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    temporary_meta.replace(meta_path)
    return np.load(embs_path, mmap_mode="r", allow_pickle=False)


def make_head(name: str) -> torch.nn.Module:
    if name == "official_head":
        return SecStructPredictionHead(embed_dim=1280, num_blocks=2, conv_dim=64)
    if name == "jev_head":
        return FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer="resnet2d",
                                resnet_blocks=4, chunk_size=64, grad_checkpoint=True)
    raise ValueError(name)


def jev_score_only(head: FlatDecisionHead, h: torch.Tensor, ids: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
    """Exactly the finite score branch of pair_scores_and_types/forward, without types/masking."""
    if head.resnet2d is None:
        raise RuntimeError("bpRNA JEV control requires scorer='resnet2d'")
    length = h.shape[1]
    chunk = 128 if length > 128 else length
    use_checkpoint = head.training and head.grad_checkpoint and torch.is_grad_enabled()
    chunks = []
    for start in range(0, length, chunk):
        hj = h[:, start:start + chunk]
        if use_checkpoint:
            chunks.append(checkpoint(head.pair_repr.cross, h, hj, use_reentrant=False))
        else:
            chunks.append(head.pair_repr.cross(h, hj))
    z = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    if not use_checkpoint:
        scores = head.resnet2d(z)
    else:
        batch_size = z.shape[0]
        idx = torch.arange(length, device=z.device, dtype=z.dtype)
        rel = (idx[:, None] - idx[None, :]).unsqueeze(0).unsqueeze(0)
        rel = rel.expand(batch_size, 1, length, length)
        x = head.resnet2d.stem(torch.cat([z.permute(0, 3, 1, 2), rel], dim=1))
        for block in head.resnet2d.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        scores = head.resnet2d.head(x).squeeze(1)
    if head.use_turner_prior:
        scores = scores + head.prior_weight.to(scores.dtype) * turner_phys_scores(ids).to(h.device)
    return head.calibration(scores, lengths)


def make_batches(indices: list[int], lengths: np.ndarray, batch_size: int,
                 rng: np.random.Generator | None) -> list[list[int]]:
    groups = {}
    for index in indices:
        groups.setdefault(int(lengths[index]), []).append(index)
    batches = []
    for group in groups.values():
        order = rng.permutation(group).tolist() if rng is not None else group
        batches.extend(order[start:start + batch_size] for start in range(0, len(order), batch_size))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def batch_tensors(embs: np.ndarray, rows: list[dict], indices: list[int], device: str):
    length = len(model_sequence(rows[indices[0]]))
    if any(len(model_sequence(rows[index])) != length for index in indices):
        raise RuntimeError("internal error: non-homogeneous length batch")
    # Cached states retain BOS/EOS; both heads receive residues only.
    array = np.asarray(embs[indices, 1:length + 1, :])
    h = torch.as_tensor(array, dtype=torch.float32, device=device)
    ids = torch.cat([sequence_ids(model_sequence(rows[index]), device) for index in indices], dim=0)
    target = torch.zeros((len(indices), length, length), dtype=torch.float32, device=device)
    for batch_index, index in enumerate(indices):
        for i, j in rows[index]["pairs"]:
            target[batch_index, i, j] = 1.0
            target[batch_index, j, i] = 1.0
    return h, ids, target


def forward_scores(name: str, model: torch.nn.Module, h: torch.Tensor,
                   ids: torch.Tensor) -> torch.Tensor:
    if name == "official_head":
        return model(h)
    lengths = torch.full((h.shape[0],), h.shape[1], dtype=torch.long, device=h.device)
    return jev_score_only(model, h, ids, lengths)


@torch.no_grad()
def verify_jev_score_only(embs: np.ndarray, rows: list[dict], index: int, device: str):
    model = make_head("jev_head").to(device).eval()
    h, ids, _ = batch_tensors(embs, rows, [index], device)
    lengths = torch.full((1,), h.shape[1], dtype=torch.long, device=h.device)
    expected, _ = model.pair_scores_and_types(h)
    if model.use_turner_prior:
        expected = expected + model.prior_weight.to(expected.dtype) * turner_phys_scores(ids).to(h.device)
    expected = model.calibration(expected, lengths)
    actual = jev_score_only(model, h, ids, lengths)
    if not torch.allclose(actual, expected, rtol=0, atol=0):
        raise RuntimeError("JEV score_only differs from the complete score branch")
    del model, h, ids, expected, actual
    torch.cuda.empty_cache()


def upper_bce(logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    upper = torch.triu(torch.ones(target.shape[1:], dtype=torch.bool, device=target.device), diagonal=1)
    return F.binary_cross_entropy_with_logits(logits[..., upper], target[..., upper], reduction=reduction)


@torch.no_grad()
def validation_bce(name: str, model: torch.nn.Module, embs: np.ndarray, rows: list[dict],
                   indices: list[int], lengths: np.ndarray, batch_size: int, device: str) -> float:
    model.eval()
    total = 0.0
    for batch in make_batches(indices, lengths, batch_size, None):
        h, ids, target = batch_tensors(embs, rows, batch, device)
        loss = upper_bce(forward_scores(name, model, h, ids), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite validation BCE: {name}")
        total += float(loss) * len(batch)
    return total / len(indices)


def pair_matrix(pairs, length: int) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float32)
    for i, j in pairs:
        matrix[i, j] = matrix[j, i] = 1.0
    return matrix


def matrix_pairs(matrix: np.ndarray) -> list[list[int]]:
    return [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(matrix, k=1)))]


def metrics_from_pairs(true_pairs, predicted_pairs, length: int) -> dict:
    truth = pair_matrix(true_pairs, length)
    prediction = pair_matrix(predicted_pairs, length)
    return {
        "precision": float(ss_precision(truth, prediction)),
        "recall": float(ss_recall(truth, prediction)),
        "f1": float(ss_f1(truth, prediction)),
    }


def prediction_record(rows: list[dict], index: int, predicted_pairs, head: str, seed: int,
                      threshold: float | None) -> dict:
    row = rows[index]
    record = {
        "id": row["id"], "family": row["family"], "held_out_family": row["held_out_family"],
        "split": row["split"], "cache_index": int(index), "length": len(model_sequence(row)),
        "true_pairs": row["pairs"], "predicted_pairs": [list(map(int, pair)) for pair in predicted_pairs],
        "head": head, "seed": int(seed), "official_threshold": threshold,
    }
    record.update(metrics_from_pairs(record["true_pairs"], record["predicted_pairs"], record["length"]))
    return record


def macro_metrics(records: list[dict]) -> dict:
    return {key: float(np.mean([record[key] for record in records]))
            for key in ("precision", "recall", "f1")}


def python_reference_dp(scores: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    _, dec, kp = _max_dp_reference(scores, mask)
    return sorted(_traceback_max(dec, kp, scores.shape[0]))


def verify_dp_decoder() -> None:
    length = 9
    positions = np.arange(length)
    mask = np.abs(positions[:, None] - positions[None, :]) > 3
    rng = np.random.default_rng(20261028)
    cases = {
        "all_negative": -np.ones((length, length), dtype=np.float64),
        "zero_ties": np.zeros((length, length), dtype=np.float64),
        "integer_ties": rng.integers(0, 2, size=(length, length)).astype(np.float64),
    }
    for label, scores in cases.items():
        decode_scores = np.where(mask, scores, -np.inf)
        expected = python_reference_dp(decode_scores, mask)
        actual = nussinov_map_single(decode_scores, mask)
        if actual != expected:
            raise RuntimeError(f"C/Python DP decoder mismatch at startup: {label}")


@torch.no_grad()
def tune_official(model: torch.nn.Module, embs: np.ndarray, rows: list[dict], indices: list[int],
                  lengths: np.ndarray, batch_size: int, device: str, seed: int):
    model.eval()
    sums = np.zeros(len(THRESHOLDS), dtype=np.float64)
    candidates = [[] for _ in THRESHOLDS]
    for batch in make_batches(indices, lengths, batch_size, None):
        h, ids, _ = batch_tensors(embs, rows, batch, device)
        probabilities = torch.sigmoid(model(h)).float().cpu().numpy()
        for batch_index, index in enumerate(batch):
            row = rows[index]
            sequence = model_sequence(row)
            truth = pair_matrix(row["pairs"], len(sequence))
            for threshold_index, threshold in enumerate(THRESHOLDS):
                decoded = prob_mat_to_sec_struct(probabilities[batch_index].copy(), sequence, threshold)
                predicted_pairs = matrix_pairs(decoded)
                candidates[threshold_index].append((index, predicted_pairs))
                sums[threshold_index] += float(ss_f1(truth, decoded))
    macro = sums / len(indices)
    best = 0
    for index in range(1, len(THRESHOLDS)):
        if macro[index] > macro[best]:  # strict comparison preserves lowest-threshold tie breaking
            best = index
    threshold = THRESHOLDS[best]
    records = [prediction_record(rows, index, pairs, "official_head", seed, threshold)
               for index, pairs in candidates[best]]
    report = [{"threshold": threshold, "validation_macro_f1": float(score)}
              for threshold, score in zip(THRESHOLDS, macro)]
    return threshold, report, records


@torch.no_grad()
def decode_records(name: str, model: torch.nn.Module, embs: np.ndarray, rows: list[dict],
                   indices: list[int], lengths: np.ndarray, batch_size: int, device: str,
                   seed: int, threshold: float | None = None) -> list[dict]:
    model.eval()
    records = []
    for batch in make_batches(indices, lengths, batch_size, None):
        h, ids, _ = batch_tensors(embs, rows, batch, device)
        scores = forward_scores(name, model, h, ids)
        expected_shape = (len(batch), h.shape[1], h.shape[1])
        if tuple(scores.shape) != expected_shape:
            raise RuntimeError(f"score shape mismatch: got {tuple(scores.shape)}, expected {expected_shape}")
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"non-finite scores before decode: {name}")
        if name == "official_head":
            if threshold is None:
                raise RuntimeError("official test decode requires its validation-selected threshold")
            scores = torch.sigmoid(scores)
        scores = scores.float().cpu().numpy()
        for batch_index, index in enumerate(batch):
            sequence = model_sequence(rows[index])
            if name == "official_head":
                decoded = prob_mat_to_sec_struct(scores[batch_index].copy(), sequence, threshold)
                predicted_pairs = matrix_pairs(decoded)
            else:
                mask = pair_mask_from_ids(ids[batch_index:batch_index + 1].detach().cpu(),
                                          model.min_loop)[0].numpy()
                if mask.shape != scores[batch_index].shape or not np.array_equal(mask, mask.T):
                    raise RuntimeError("JEV decode mask shape/symmetry mismatch")
                decode_scores = np.where(mask, scores[batch_index], -np.inf)
                predicted_pairs = nussinov_map_single(decode_scores, mask)
            records.append(prediction_record(rows, index, predicted_pairs, name, seed, threshold))
    return records


def train_one(name: str, seed: int, embs: np.ndarray, rows: list[dict], selected: dict,
              train_batch_size: int, device: str, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_head(name).to(device)
    parameters = [parameter for parameter_name, parameter in model.named_parameters()
                  if name != "jev_head" or not parameter_name.startswith("type_head.")]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    lengths = np.asarray([len(model_sequence(row)) for row in rows], dtype=np.int32)
    train_indices, valid_indices = selected["train"], selected["valid"]
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        for batch in make_batches(train_indices, lengths, train_batch_size, rng):
            h, ids, target = batch_tensors(embs, rows, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = upper_bce(forward_scores(name, model, h, ids), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training BCE: {name} seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
        valid_loss = validation_bce(name, model, embs, rows, valid_indices, lengths,
                                    train_batch_size, device)
        history.append({"epoch": epoch, "validation_bce": valid_loss})
        if valid_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, bad = valid_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        print(f"[{name}/seed={seed}/epoch={epoch}] valid_bce={valid_loss:.7f} best={best_loss:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"no finite best checkpoint: {name} seed={seed}")
    converged = bad >= PATIENCE
    model.load_state_dict(best_state, strict=True)

    threshold, threshold_report = None, None
    if name == "official_head":
        threshold, threshold_report, valid_records = tune_official(
            model, embs, rows, valid_indices, lengths, train_batch_size, device, seed)
    else:
        valid_records = decode_records(name, model, embs, rows, valid_indices, lengths,
                                       train_batch_size, device, seed)
    test_records = decode_records(name, model, embs, rows, selected["test"], lengths,
                                  train_batch_size, device, seed, threshold)
    valid_metrics, test_metrics = macro_metrics(valid_records), macro_metrics(test_records)

    held_out_family = rows[train_indices[0]]["held_out_family"]
    checkpoint = out_dir / f"{name}_s{seed}_best.pt"
    temporary_checkpoint = checkpoint.with_suffix(".tmp.pt")
    torch.save({
        "model_state_dict": best_state, "head": name, "seed": seed,
        "held_out_family": held_out_family,
        "best_epoch": best_epoch, "best_validation_bce": best_loss,
        "official_threshold": threshold, "converged": converged,
        "architecture": ("SecStructPredictionHead(embed_dim=1280,num_blocks=2,conv_dim=64)"
                         if name == "official_head" else
                         "FlatDecisionHead(d_model=1280,d_z=128,hidden=64,scorer=resnet2d,resnet_blocks=4,chunk_size=64,grad_checkpoint=True)"),
    }, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint)
    prediction_path = out_dir / f"{name}_s{seed}_predictions.jsonl"
    temporary_prediction = prediction_path.with_suffix(".tmp.jsonl")
    with temporary_prediction.open("w", encoding="utf-8") as handle:
        for record in sorted(valid_records + test_records, key=lambda item: item["cache_index"]):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                    default=json_default, allow_nan=False) + "\n")
    temporary_prediction.replace(prediction_path)
    run = {
        "seed": seed, "device": device, "held_out_family": held_out_family,
        "converged": converged, "early_stopped": converged,
        "stop_reason": "validation_plateau" if converged else "max_epochs_safety_cap",
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_bce": best_loss,
        "validation_history": history, "official_threshold": threshold,
        "official_threshold_validation_history": threshold_report,
        "validation_macro_precision": valid_metrics["precision"],
        "validation_macro_recall": valid_metrics["recall"],
        "validation_macro_f1": valid_metrics["f1"],
        "test_macro_precision": test_metrics["precision"],
        "test_macro_recall": test_metrics["recall"],
        "test_macro_f1": test_metrics["f1"],
        "n_train": len(train_indices), "n_valid": len(valid_indices), "n_test": len(selected["test"]),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "prediction_file": str(prediction_path), "prediction_sha256": sha256(prediction_path),
    }
    print(f"[{name}/seed={seed}] test_macro_f1={test_metrics['f1']:.6f} "
          f"best_epoch={best_epoch} converged={converged}", flush=True)
    return run


def validate_official_threshold_report(report: list[dict], threshold: float) -> None:
    if not isinstance(report, list) or len(report) != len(THRESHOLDS):
        raise RuntimeError("official threshold report must contain exactly 29 grid points")
    scores = []
    for expected_threshold, item in zip(THRESHOLDS, report):
        if item["threshold"] != expected_threshold or not np.isfinite(item["validation_macro_f1"]):
            raise RuntimeError("official threshold report grid/value mismatch")
        scores.append(float(item["validation_macro_f1"]))
    best = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best]:
            best = index
    if threshold != THRESHOLDS[best]:
        raise RuntimeError("official threshold violates strict lowest-threshold tie breaking")


@torch.no_grad()
def independent_recheck(name: str, run: dict, embs: np.ndarray, rows: list[dict], selected: dict,
                        train_batch_size: int, device: str) -> None:
    if sha256(Path(run["prediction_file"])) != run["prediction_sha256"]:
        raise RuntimeError("prediction SHA256 changed before independent recheck")
    expected = {}
    for split in ("valid", "test"):
        for index in selected[split]:
            row = rows[index]
            key = (split, index, row["id"])
            expected[key] = (index, row)
    stored = {}
    with Path(run["prediction_file"]).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("split"), record.get("cache_index"), record.get("id"))
            if key in stored:
                raise RuntimeError(f"duplicate prediction at line {line_number}: {key}")
            if key not in expected:
                raise RuntimeError(f"unexpected prediction at line {line_number}: {key}")
            stored[key] = record
    if set(stored) != set(expected):
        raise RuntimeError("prediction JSONL has missing valid/test records")
    required = {"family", "held_out_family", "head", "seed", "official_threshold",
                "length", "true_pairs", "predicted_pairs"}
    for split in ("valid", "test"):
        expected_ids = {rows[index]["id"] for index in selected[split]}
        actual_ids = [record["id"] for record in stored.values() if record["split"] == split]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise RuntimeError(f"independent {split} ID set mismatch")
    for key, record in stored.items():
        index, row = expected[key]
        if (not required.issubset(record) or record["family"] != row["family"]
                or record["held_out_family"] != row["held_out_family"]
                or record["head"] != name or record["seed"] != run["seed"]
                or record["official_threshold"] != run["official_threshold"]
                or record["length"] != len(model_sequence(row))
                or record["true_pairs"] != row["pairs"]):
            raise RuntimeError(f"prediction metadata/truth mismatch: {key}")

    if sha256(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 changed before independent recheck")
    checkpoint_data = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    model = make_head(name).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    if (checkpoint_data["head"] != name or checkpoint_data["seed"] != run["seed"]
            or checkpoint_data["held_out_family"] != run["held_out_family"]
            or checkpoint_data["best_epoch"] != run["best_epoch"]
            or checkpoint_data["official_threshold"] != run["official_threshold"]
            or checkpoint_data["converged"] != run["converged"]
            or len(run["validation_history"]) != run["epochs_run"]):
        raise RuntimeError("checkpoint/history metadata mismatch")
    if any(row["held_out_family"] != run["held_out_family"] for _, row in expected.values()):
        raise RuntimeError("held-out family differs across prediction rows")
    best_point = next(item for item in run["validation_history"] if item["epoch"] == run["best_epoch"])
    if not np.isclose(best_point["validation_bce"], run["best_validation_bce"], rtol=0, atol=1e-12):
        raise RuntimeError("best checkpoint is not bound to validation history")
    if run["converged"]:
        if (run["epochs_run"] != run["best_epoch"] + PATIENCE or not run["early_stopped"]
                or not str(run["device"]).startswith("cuda") or not str(device).startswith("cuda")):
            raise RuntimeError("converged run fails early-stopping/GPU audit")
    if run["n_valid"] != len(selected["valid"]) or run["n_test"] != len(selected["test"]):
        raise RuntimeError("independent valid/test count mismatch")

    lengths = np.asarray([len(model_sequence(row)) for row in rows], dtype=np.int32)
    if name == "official_head":
        fresh_threshold, fresh_report, fresh_valid = tune_official(
            model, embs, rows, selected["valid"], lengths, train_batch_size, device, run["seed"])
        validate_official_threshold_report(run["official_threshold_validation_history"],
                                           run["official_threshold"])
        validate_official_threshold_report(fresh_report, fresh_threshold)
        if fresh_threshold != run["official_threshold"]:
            raise RuntimeError("independent official threshold mismatch")
        for old_item, new_item in zip(run["official_threshold_validation_history"], fresh_report):
            if not np.isclose(old_item["validation_macro_f1"], new_item["validation_macro_f1"],
                              rtol=0, atol=1e-12):
                raise RuntimeError("stored official threshold report differs from GPU recheck")
    else:
        fresh_valid = decode_records(name, model, embs, rows, selected["valid"], lengths,
                                     train_batch_size, device, run["seed"])
    fresh_test = decode_records(name, model, embs, rows, selected["test"], lengths,
                                train_batch_size, device, run["seed"], run["official_threshold"])

    for split, fresh_records in (("valid", fresh_valid), ("test", fresh_test)):
        fresh_by_key = {}
        for record in fresh_records:
            key = (record["split"], record["cache_index"], record["id"])
            if key in fresh_by_key:
                raise RuntimeError(f"duplicate fresh prediction: {key}")
            fresh_by_key[key] = record
        expected_keys = {key for key in expected if key[0] == split}
        if set(fresh_by_key) != expected_keys:
            raise RuntimeError(f"fresh {split} prediction set mismatch")
        recomputed_records = []
        for key, fresh in fresh_by_key.items():
            stored_record = stored[key]
            if fresh["predicted_pairs"] != stored_record["predicted_pairs"]:
                raise RuntimeError(f"independent predicted_pairs mismatch: {key}")
            _, row = expected[key]
            recomputed = metrics_from_pairs(row["pairs"], fresh["predicted_pairs"],
                                            len(model_sequence(row)))
            for metric, value in recomputed.items():
                if not np.isclose(value, stored_record[metric], rtol=0, atol=1e-12):
                    raise RuntimeError(f"stored per-structure {metric} mismatch: {key}")
            recomputed_records.append(recomputed)
        macro = macro_metrics(recomputed_records)
        prefix = "validation" if split == "valid" else "test"
        for metric in ("precision", "recall", "f1"):
            if not np.isclose(macro[metric], run[f"{prefix}_macro_{metric}"], rtol=0, atol=1e-12):
                raise RuntimeError(f"independent {split} macro {metric} mismatch: {name}")

    if name == "jev_head":
        shortest = min(selected["test"], key=lambda index: (len(model_sequence(rows[index])), index))
        h, ids, _ = batch_tensors(embs, rows, [shortest], device)
        model.eval()
        scores = forward_scores(name, model, h, ids)
        if (tuple(scores.shape) != (1, h.shape[1], h.shape[1])
                or not torch.isfinite(scores).all()):
            raise RuntimeError("JEV independent DP input shape/finite check failed")
        mask = pair_mask_from_ids(ids.detach().cpu(), model.min_loop)[0].numpy()
        score_matrix = scores[0].float().cpu().numpy()
        if mask.shape != score_matrix.shape or not np.array_equal(mask, mask.T):
            raise RuntimeError("JEV independent DP mask shape/symmetry mismatch")
        decode_scores = np.where(mask, score_matrix, -np.inf)
        c_pairs = nussinov_map_single(decode_scores, mask)
        reference_pairs = python_reference_dp(decode_scores, mask)
        if c_pairs != reference_pairs:
            raise RuntimeError("JEV checkpoint C/Python DP pairwise mismatch")
    run["independent_recheck"] = "PASS"
    del model
    torch.cuda.empty_cache()


def selected_indices(rows: list[dict], full: dict, limit: int) -> dict:
    selected = {}
    for split, indices in full.items():
        if limit:
            longest = sorted(indices, key=lambda index: len(model_sequence(rows[index])), reverse=True)[:limit]
            chosen = sorted(longest)  # preserve original relative order after selection
            if max(len(model_sequence(rows[index])) for index in chosen) != max(
                    len(model_sequence(rows[index])) for index in indices):
                raise RuntimeError(f"smoke selection lost the longest sequence: {split}")
            selected[split] = chosen
        else:
            selected[split] = list(indices)
    return selected


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def require_official_tracked_clean(path: Path) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True)
    unexpected = []
    for line in status.splitlines():
        if line.startswith("?? ") and "__pycache__" in Path(line[3:]).parts:
            continue
        unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"official RiNALMo tracked files are not clean: {unexpected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2, help="frozen embedding extraction batch size")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="longest N per split; 0 means full official protocol")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing any CPU fallback")
    if args.batch_size < 1 or args.train_batch_size < 1 or args.limit < 0:
        raise ValueError("batch sizes must be positive and limit must be nonnegative")
    if args.out_dir.exists():
        if not args.out_dir.is_dir() or any(args.out_dir.iterdir()):
            raise RuntimeError("out_dir must not exist or must be an empty directory")
    else:
        args.out_dir.mkdir(parents=True)
    verify_dp_decoder()

    card_path = ROOT / "jsonl" / "data_card.json"
    if sha256(card_path) != EXPECTED_DATA_CARD_SHA256:
        raise RuntimeError("bpRNA data card SHA256 mismatch")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("status") != "PASS":
        raise RuntimeError("bpRNA data card is not PASS")
    weights_path = Path(WEIGHTS) / "model.safetensors"
    weights_config_path = Path(WEIGHTS) / "config.json"
    weights_sha256 = sha256(weights_path)
    weights_config_sha256 = sha256(weights_config_path)
    if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("RiNALMo-giga weights SHA256 mismatch")
    implementation_paths = {
        "preflight": Path("/home/rna_jev/tools/rinalmo_preflight.py"),
        "decision_head": Path("/home/rna_jev/src/rnajepa/decision_head.py"),
        "official_downstream": OFFICIAL_REPO / "rinalmo/model/downstream.py",
        "official_sec_struct": OFFICIAL_REPO / "rinalmo/utils/sec_struct.py",
        "dp_decode_c_wrapper": Path("/home/rna_jev/tools/dp_decode_c_wrapper.py"),
        "dp_decode_c_so": Path("/home/rna_jev/tools/dp_decode_c.so"),
    }
    implementation_sha256 = {key: sha256(path) for key, path in implementation_paths.items()}
    official_commit = git_commit(OFFICIAL_REPO)
    if official_commit != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("official RiNALMo commit mismatch")
    require_official_tracked_clean(OFFICIAL_REPO)

    rows, full_indices, split_paths, jsonl_sha256 = load_fold(args.family, card)
    selected = selected_indices(rows, full_indices, args.limit)
    cache = ROOT / "cache" / f"{args.family}_bprna_rinalmo_giga_v1"
    embs = extract_embeddings(rows, cache, args.device, args.batch_size, weights_sha256,
                              weights_config_sha256, implementation_sha256["preflight"], jsonl_sha256)
    verify_jev_score_only(embs, rows, selected["train"][0], args.device)

    outputs = {"official_head": {"runs": []}, "jev_head": {"runs": []}}
    for seed in SEEDS:
        for name in ("official_head", "jev_head"):
            run = train_one(name, seed, embs, rows, selected, args.train_batch_size,
                            args.device, args.out_dir)
            independent_recheck(name, run, embs, rows, selected, args.train_batch_size, args.device)
            outputs[name]["runs"].append(run)
    for name, output in outputs.items():
        values = [run["test_macro_f1"] for run in output["runs"]]
        output["test_macro_f1_mean"] = float(np.mean(values))
        output["test_macro_f1_sample_sd"] = float(np.std(values, ddof=1))

    official = np.asarray([run["test_macro_f1"] for run in outputs["official_head"]["runs"]])
    jev = np.asarray([run["test_macro_f1"] for run in outputs["jev_head"]["runs"]])
    test = stats.ttest_rel(jev, official, alternative="greater")
    p_value, t_statistic = float(test.pvalue), float(test.statistic)
    paired_delta = jev - official
    paired_delta_mean = float(np.mean(paired_delta))
    paired_delta_sd = float(np.std(paired_delta, ddof=1))
    statistics_ok = bool(paired_delta_sd > 0 and np.isfinite(paired_delta_sd)
                         and np.isfinite(p_value) and np.isfinite(t_statistic))
    comparison = {
        "paired_by_seed": [{"seed": seed, "official_macro_f1": float(official[index]),
                            "jev_macro_f1": float(jev[index]), "delta_jev_minus_official": float(jev[index] - official[index])}
                           for index, seed in enumerate(SEEDS)],
        "official_macro_f1_mean": float(np.mean(official)),
        "official_macro_f1_sample_sd": float(np.std(official, ddof=1)),
        "jev_macro_f1_mean": float(np.mean(jev)),
        "jev_macro_f1_sample_sd": float(np.std(jev, ddof=1)),
        "paired_delta_mean": paired_delta_mean,
        "paired_delta_sample_sd": paired_delta_sd,
        "one_sided_ttest_rel_greater_t_statistic": t_statistic if np.isfinite(t_statistic) else None,
        "one_sided_ttest_rel_greater_p_value": p_value if np.isfinite(p_value) else None,
        "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
        "significant_positive": bool(statistics_ok and paired_delta_mean > 0 and p_value < 0.05),
        "inference_scope": "paired training-seed effects; no test-set population inference",
    }
    all_converged = all(run["converged"] for output in outputs.values() for run in output["runs"])
    independent_rechecks_complete = all(run.get("independent_recheck") == "PASS"
                                        for output in outputs.values() for run in output["runs"])
    scientific_eligibility = bool(args.limit == 0 and all_converged and independent_rechecks_complete)
    if all_converged:
        status = "SMOKE_PASS" if args.limit > 0 else "PASS"
    else:
        status = "NOT_CONVERGED"
    cache_meta = cache.with_suffix(".json")
    protocol = {
        "task": "RiNALMo bpRNA intra-family secondary structure",
        "family": args.family,
        "sample_scope": "engineering_smoke_subset_not_scientific_result" if args.limit else "full_official_protocol",
        "selection": "longest limit per split, original selected order retained" if args.limit else "no filtering",
        "device": args.device,
        "data_card": str(card_path), "data_card_sha256": sha256(card_path),
        "split": {split: {"path": str(split_paths[split]), "sha256": jsonl_sha256[split],
                          "n_full": len(full_indices[split]), "n_used": len(selected[split])}
                  for split in SPLITS},
        "pretrained_weights": str(weights_path), "pretrained_sha256": weights_sha256,
        "pretrained_config": str(weights_config_path), "pretrained_config_sha256": weights_config_sha256,
        "cache_meta": str(cache_meta), "cache_meta_sha256": sha256(cache_meta),
        "code_sha256": sha256(Path(__file__).resolve()),
        "code_commit": git_commit(Path("/home/rna_jev")),
        "implementation_paths": {key: str(path) for key, path in implementation_paths.items()},
        "implementation_sha256": implementation_sha256,
        "software_versions": {"torch": torch.__version__,
                              "multimolecule": importlib.metadata.version("multimolecule")},
        "official_repo": str(OFFICIAL_REPO), "official_commit": official_commit,
        "official_repo_tracked_files_clean": True,
        "batch": {"embedding_batch_size": args.batch_size, "train_batch_size": args.train_batch_size,
                  "length_grouped_same_length_batches": True,
                  "epoch_batch_order": "SeedSequence([seed, epoch]); identical for both heads"},
        "seeds": list(SEEDS), "optimizer": "Adam(lr=1e-3)",
        "convergence": {"criterion": "held-out validation upper-triangle BCE best checkpoint",
                        "validation_aggregation": "mean of per-structure upper-triangle BCE",
                        "patience": PATIENCE, "min_delta": MIN_DELTA,
                        "max_epochs_safety_cap": MAX_EPOCHS,
                        "not_plateaued_is_not_converged": True},
        "single_variable": "head architecture and native decoder on identical frozen RiNALMo embeddings; all-upper-triangle BCE fixed",
        "official_head": "SecStructPredictionHead(embed_dim=1280,num_blocks=2,conv_dim=64)",
        "jev_head": "FlatDecisionHead(d_model=1280,d_z=128,hidden=64,scorer=resnet2d,resnet_blocks=4,chunk_size=64,grad_checkpoint=True), score_only",
        "training_truth": "all upper-triangle pairs, including pseudoknots and noncanonical pairs; no legality mask or -inf in BCE",
        "official_decoder": "prob_mat_to_sec_struct; threshold selected once on best-checkpoint validation macro F1",
        "official_threshold_grid": list(THRESHOLDS),
        "jev_decoder": "pair_mask_from_ids + nussinov_map_single noncrossing MAP; no threshold",
        "metrics": "per-structure tolerant ss_precision/ss_recall/ss_f1, then macro average; no strict micro metric",
    }
    payload = {"schema_version": 1, "status": status,
               "scientific_eligibility": scientific_eligibility,
               "protocol": protocol, "heads": outputs, "comparison": comparison}
    result_path = args.out_dir / "result.json"
    temporary = args.out_dir / "result.tmp"
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=json_default,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    checked = json.loads(result_path.read_text(encoding="utf-8"))
    if (checked["status"] != payload["status"]
            or checked["scientific_eligibility"] != payload["scientific_eligibility"]):
        raise RuntimeError("result JSON reread failed")
    print(f"RINALMO_BPRNA_DUAL_{payload['status']} family={args.family} "
          f"delta={comparison['paired_delta_mean']:+.6f} "
          f"scientific_eligibility={payload['scientific_eligibility']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
