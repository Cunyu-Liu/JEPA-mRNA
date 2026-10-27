#!/usr/bin/env python3
"""Round-33 raw-sequence end-to-end speed benchmark on one RTX 4090.

Scope: tokenization, encoder forward, task head, required decoding and scoring input.
No cached embeddings are used. CUDA is mandatory and predictions are checked against
frozen strict micro-F1 values before a result can be accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rna_jev")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "tools" / "flash_shim"))

from rinalmo_preflight import load_encoder, encode, VOCAB
from rnajepa.train_decision import TrainConfig, build_decision_model
from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.harness import valid_pair_mask
from tools.dp_decode_c_wrapper import nussinov_map_batch

W = Path("/teams/HoulabData_1788502323/liucunyu")
DEFAULT_DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl")
R2D_CKPT = W / "rna_ssp_20261024/weights/main_ablations/r2d.pt"
PLANA_S0 = W / "rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt"
PLANA_S1 = W / "rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt"
RINALMO_FT = Path("/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt")
EXPECTED_F1 = {"r2d": 0.667971, "plana": 0.7866, "xens2": 0.8039, "rinalmo_ft": 0.7214,
               "mxfold2": 0.565134, "rnaformer": 0.7578}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_r2d_head(device: torch.device):
    state = torch.load(R2D_CKPT, map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in fields})
    model = build_decision_model(cfg)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def load_plana_head(path: Path, device: torch.device):
    from rnajepa.decision_head import FlatDecisionHead
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder, _ = load_encoder(device=str(device), dtype=torch.float32)
    encoder.load_state_dict(ckpt["model"]["encoder"], strict=True)
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer="resnet2d", chunk_size=0)
    head.load_state_dict(ckpt["model"]["head"], strict=True)
    return encoder.to(device).eval(), head.to(device).eval()


def load_models(name: str, device: torch.device) -> dict:
    if name == "r2d":
        encoder, report = load_encoder(device=str(device), dtype=torch.bfloat16)
        return {"name": name, "encoder": encoder, "head": load_r2d_head(device), "report": report}
    if name == "plana":
        encoder, head = load_plana_head(PLANA_S0, device)
        return {"name": name, "encoder": encoder, "head": head}
    if name == "xens2":
        enc0, head0 = load_plana_head(PLANA_S0, device)
        enc1, head1 = load_plana_head(PLANA_S1, device)
        base_encoder, report = load_encoder(device=str(device), dtype=torch.bfloat16)
        return {"name": name, "enc0": enc0, "enc1": enc1, "head0": head0, "head1": head1,
                "base_encoder": base_encoder, "r2d_head": load_r2d_head(device), "report": report}
    if name == "rinalmo_ft":
        from eval_rinalmo_ft import load_ft_model
        lm, head, alphabet, threshold = load_ft_model(str(RINALMO_FT), str(device))
        return {"name": name, "encoder": lm, "head": head, "alphabet": alphabet, "threshold": threshold}
    if name == "mxfold2":
        from types import SimpleNamespace
        from mxfold2.predict import Predict
        package = Path(__import__("mxfold2").__path__[0])
        weights = package / "models/TrainSetAB.pth"
        args = SimpleNamespace(
            model="MixC", max_helix_length=30, embed_size=64,
            num_filters=[64] * 8, filter_size=[5, 3] * 4, pool_size=[1], dilation=0,
            num_lstm_layers=2, num_lstm_units=32, num_transformer_layers=0,
            num_transformer_hidden_units=2048, num_transformer_att=8,
            num_hidden_units=[32], num_paired_filters=[64] * 8,
            paired_filter_size=[5, 3] * 4, dropout_rate=0.5, fc_dropout_rate=0.5,
            num_att=8, pair_join="cat", no_split_lr=False)
        model, config = Predict().build_model(args)
        state = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        report = {"weights": str(weights), "weights_sha256": sha256_file(weights),
                  "strict_load": True, "config": config, "n_parameters": sum(p.numel() for p in model.parameters())}
        return {"name": name, "model": model.to(device).eval(), "report": report}
    if name == "rnaformer":
        from types import SimpleNamespace
        sys.path.insert(0, str(REPO / "third_party/RNAformer"))
        from ss.run_rnaformer import build_model
        weights = W / "rna_ssp_20261024/weights/baselines/rnaformer/RNAformer_32M_state_dict_bprna.pth"
        config = W / "rna_ssp_20261024/weights/baselines/rnaformer/RNAformer_32M_config_bprna.yml"
        args = SimpleNamespace(state_dict=str(weights), config=str(config), cycling=0)
        model, report = build_model(args)
        report.update({"weights": str(weights), "weights_sha256": sha256_file(weights),
                       "config": str(config), "config_sha256": sha256_file(config),
                       "n_parameters": sum(p.numel() for p in model.parameters())})
        return {"name": name, "model": model.to(device).eval(), "report": report}
    raise ValueError(name)


@torch.no_grad()
def encoder_hidden(encoder, seqs: list[str], device: torch.device, autocast_bf16: bool) -> torch.Tensor:
    ids = torch.cat([encode(seq, device=str(device)) for seq in seqs], dim=0)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if autocast_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(input_ids=ids, attention_mask=mask)
    else:
        out = encoder(input_ids=ids, attention_mask=mask)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return hidden[:, 1:-1, :]


@torch.no_grad()
def decision_scores(head, hidden, seqs: list[str], device: torch.device) -> torch.Tensor:
    length = len(seqs[0])
    ids = torch.tensor([[BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq] for seq in seqs],
                       dtype=torch.long, device=device)
    lengths = torch.full((len(seqs),), length, dtype=torch.long, device=device)
    return head(hidden.float(), ids, lengths=lengths).scores


@torch.no_grad()
def forward_our(models: dict, seqs: list[str], device: torch.device) -> np.ndarray:
    name = models["name"]
    if name == "r2d":
        hidden = encoder_hidden(models["encoder"], seqs, device, False)
        scores = decision_scores(models["head"], hidden, seqs, device)
    elif name == "plana":
        hidden = encoder_hidden(models["encoder"], seqs, device, True)
        scores = decision_scores(models["head"], hidden, seqs, device)
    elif name == "xens2":
        h0 = encoder_hidden(models["enc0"], seqs, device, True)
        h1 = encoder_hidden(models["enc1"], seqs, device, True)
        hr = encoder_hidden(models["base_encoder"], seqs, device, False)
        s0 = decision_scores(models["head0"], h0, seqs, device)
        s1 = decision_scores(models["head1"], h1, seqs, device)
        sr = decision_scores(models["r2d_head"], hr, seqs, device)
        scores = 0.7 * (0.5 * s0 + 0.5 * s1) + 0.3 * sr
    else:
        raise ValueError(name)
    return scores.float().cpu().numpy()


@torch.no_grad()
def forward_rinalmo_ft(models: dict, seqs: list[str], device: torch.device) -> list[np.ndarray]:
    alphabet = models["alphabet"]
    tokens = torch.tensor([alphabet.encode(seq) for seq in seqs], dtype=torch.int64, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        representation = models["encoder"](tokens)["representation"]
        logits = models["head"](representation[..., 1:-1, :]).squeeze(-1)
    probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    output = []
    for prob in probabilities:
        prob = np.triu(prob, k=1)
        output.append(prob + prob.T)
    return output


@torch.no_grad()
def forward_mxfold2(models: dict, seqs: list[str]) -> list[list[tuple[int, int]]]:
    _, _, partner_lists = models["model"](seqs)
    predictions = []
    for partners in partner_lists:
        pairs = [(index - 1, int(partner) - 1)
                 for index, partner in enumerate(partners, start=1)
                 if int(partner) > index]
        predictions.append(pairs)
    return predictions


@torch.no_grad()
def forward_rnaformer(models: dict, seqs: list[str], device: torch.device) -> list[list[tuple[int, int]]]:
    from ss.run_rnaformer import SEQ_STOI, decode_pairs
    predictions = []
    for seq in seqs:
        ids = torch.LongTensor([SEQ_STOI.get(base, 4) for base in seq])[None, :].to(device)
        src_len = torch.LongTensor([len(seq)]).to(device)
        pdb_sample = torch.ones((1, 1), dtype=torch.float32, device=device)
        logits, _ = models["model"](ids, src_len, pdb_sample)
        probability = torch.sigmoid(logits[0, :, :, -1]).float().cpu().numpy().astype(np.float64)
        probability = 0.5 * (probability + probability.T)
        np.fill_diagonal(probability, 0.0)
        predictions.append(decode_pairs(np.clip(probability, 0.0, 1.0), seq, threshold=0.5))
    return predictions


def decode_our(scores: np.ndarray, seqs: list[str]) -> list[list[tuple[int, int]]]:
    masks = np.stack([valid_pair_mask(seq).astype(np.int8) for seq in seqs])
    for index in range(len(seqs)):
        scores[index][masks[index] == 0] = -np.inf
    return [list(map(tuple, pairs)) for pairs in nussinov_map_batch(scores, masks)]


def decode_rinalmo_ft(models: dict, probabilities: list[np.ndarray], seqs: list[str]) -> list[list[tuple[int, int]]]:
    from rinalmo.utils.sec_struct import prob_mat_to_sec_struct
    predictions = []
    for prob, seq in zip(probabilities, seqs):
        matrix = prob_mat_to_sec_struct(probs=np.copy(prob), seq=seq, threshold=models["threshold"])
        predictions.append([(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(matrix, k=1)))])
    return predictions


def read_records(path: Path, max_sequences: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            gt = {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}
            records.append({"seq": seq, "gt": gt, "name": str(row.get("name") or row.get("id") or len(records))})
    records = [row for row in records if len(row["seq"]) <= 600]
    if max_sequences:
        records = records[:max_sequences]
    if not records:
        raise RuntimeError("no records")
    return records


def run_once(models: dict, records: list[dict], batch_size: int, device: torch.device) -> dict:
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        buckets[len(row["seq"])].append(index)
    tp = fp = fn = 0
    predictions = [None] * len(records)
    timed_s = 0.0
    n_timed = 0
    batch_index = 0
    for _, indices in sorted(buckets.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            seqs = [records[index]["seq"] for index in selected]
            torch.cuda.synchronize()
            begin = time.perf_counter()
            if models["name"] == "rinalmo_ft":
                probabilities = forward_rinalmo_ft(models, seqs, device)
                batch_predictions = decode_rinalmo_ft(models, probabilities, seqs)
            elif models["name"] == "mxfold2":
                batch_predictions = forward_mxfold2(models, seqs)
            elif models["name"] == "rnaformer":
                batch_predictions = forward_rnaformer(models, seqs, device)
            else:
                scores = forward_our(models, seqs, device)
                batch_predictions = decode_our(scores, seqs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if batch_index > 0:
                timed_s += elapsed
                n_timed += len(selected)
            for local, index in enumerate(selected):
                pred = set(batch_predictions[local])
                gt = records[index]["gt"]
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                predictions[index] = sorted(batch_predictions[local])
            batch_index += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    macro = []
    for row, pred_list in zip(records, predictions):
        pred, gt = set(pred_list), row["gt"]
        macro.append(2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred)))
    canonical = json.dumps([{"name": row["name"], "pairs": pred} for row, pred in zip(records, predictions)],
                           separators=(",", ":")).encode()
    return {"n_sequences": len(records), "n_timed": n_timed, "timed_wall_s": timed_s,
            "throughput_seq_per_s": n_timed / timed_s if timed_s else 0.0,
            "strict_micro_f1": f1, "strict_macro_f1": float(np.mean(macro)),
            "tp": tp, "fp": fp, "fn": fn,
            "predictions_sha256": hashlib.sha256(canonical).hexdigest(),
            "predictions": predictions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(EXPECTED_F1))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--f1-tolerance", type=float, default=0.002)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device("cuda:0")
    records = read_records(args.data, args.max_sequences)
    models = load_models(args.model, device)
    runs = []
    predictions = None
    for repeat in range(args.repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run_once(models, records, args.batch_size, device)
        predictions = result.pop("predictions")
        result["repeat"] = repeat
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        runs.append(result)
        print(f"RUN model={args.model} B={args.batch_size} repeat={repeat} "
              f"F1={result['strict_micro_f1']:.6f} seq/s={result['throughput_seq_per_s']:.3f} "
              f"peakMiB={result['peak_memory_mib']:.0f}", flush=True)
    expected = EXPECTED_F1[args.model]
    if not args.max_sequences:
        for run in runs:
            if abs(run["strict_micro_f1"] - expected) > args.f1_tolerance:
                raise RuntimeError(f"F1 mismatch for {args.model}: got {run['strict_micro_f1']} expected {expected}")
    if len({run["predictions_sha256"] for run in runs}) != 1:
        raise RuntimeError("predictions are not invariant across repeated runs")
    throughput = statistics.median(run["throughput_seq_per_s"] for run in runs)
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema_version": 1, "status": "PASS", "model": args.model,
        "protocol": {
            "device": torch.cuda.get_device_name(device), "device_type": device.type,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "data": str(args.data), "data_sha256": sha256_file(args.data),
            "n_sequences": len(records), "batch_size": args.batch_size,
            "sample_scope": "full_TS0" if not args.max_sequences else "engineering_smoke_subset_not_scientific_result",
            "repeats": args.repeats, "warmup_batches_per_repeat": 1,
            "scope": "raw sequence -> tokenization -> encoder -> task head -> required decode -> prediction scoring input",
            "cached_embeddings": False, "expected_f1": expected, "f1_tolerance": args.f1_tolerance,
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "model_report": models.get("report"),
            "code_commit": commit,
        },
        "throughput_median_seq_per_s": throughput,
        "throughput_runs_seq_per_s": [run["throughput_seq_per_s"] for run in runs],
        "f1_runs": [run["strict_micro_f1"] for run in runs],
        "runs": runs,
        "predictions": [{"name": row["name"], "pairs": pred} for row, pred in zip(records, predictions)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = json.loads(args.out.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and checked["protocol"]["device_type"] == "cuda"
    assert checked["protocol"]["cached_embeddings"] is False
    print(f"SPEED_E2E_PASS model={args.model} B={args.batch_size} median_seq_per_s={throughput:.3f} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
