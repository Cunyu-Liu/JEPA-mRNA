#!/usr/bin/env python3
"""Adapter: run the **published** RNAformer checkpoints over this project's splits.

``evaluate_RNAformer.py`` in the RNAformer repo hard-codes ``datasets/test_sets.plk``
(relative to its own cwd) and scores element-wise on the raw ``L x L`` matrix, so it
cannot be pointed at this project's corpora.  Rather than edit upstream code (the
task forbids it), this driver imports their model class + config and re-implements
*only* the data plumbing:

* reads the project JSONL corpus (``ss_data/jsonl/<split>.jsonl``) so the output
  order and sequences match what ``eval/ss/run_baselines.py`` validates against;
* runs the checkpoint, emits a dot-bracket FASTA (for the project scorer) and an
  ``.npz`` of per-sequence probability matrices (same layout
  ``rnajepa.distill.load_teacher_shard`` reads: ``sequences`` + ``probs`` object
  arrays of ``str`` / ``L x L`` float64);
* also reports the project metrics against the **release's full pair set** (the
  ``test_sets.plk`` ``pos1id``/``pos2id`` columns, including pseudoknots and
  non-canonical pairs), so the GT-restricted number cannot be mistaken for it.

Probability convention
----------------------
Copying ``evaluate_RNAformer.py`` / ``infer_RNAformer.py`` exactly: the model emits
``logits`` of shape ``(1, L, L, C)`` and the pair probability is
``sigmoid(logits[0, :, :, -1])`` (last channel; ``C = 2`` for the bprna/biophysical
configs, ``C = 1`` for the binary-output inter-family config).  We save that
**post-sigmoid** quantity symmetrised as ``(P + P.T) / 2`` with a zero diagonal, as
float64.  Entries outside the canonical mask are the model's own outputs (they are
*not* forced to zero); the ViennaRNA teacher shards are 0 off-canonical, so apply
``rnajepa.harness.valid_pair_mask(seq)`` before a like-for-like ECE comparison.

Point prediction / dot-bracket
------------------------------
RNAformer originally thresholds the raw matrix without enforcing nesting.  A valid
dot-bracket cannot be crossing, so the emitted structure is a greedy non-crossing
selection over the candidate pairs ``valid_pair_mask(seq) & (P > 0.5)``, taken in
descending probability.  This is stated because it is a decode choice, not the
model's raw output.

Usage
-----
    PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval TMPDIR=/mnt/cunyuliu/tmp \
    /home/cunyuliu/miniconda3/envs/lucaone/bin/python eval/ss/run_rnaformer.py \
        --state-dict  /mnt/cunyuliu/rna-jepa/refmodels/models/RNAformer_32M_state_dict_bprna.pth \
        --config      /mnt/cunyuliu/rna-jepa/refmodels/models/RNAformer_32M_config_bprna.yml \
        --split ref_bprna_ts0 \
        --out-dbn     /mnt/cunyuliu/rna-jepa/eval_decision/rnaformer_ref_bprna_ts0.dbn \
        --out-npz     /mnt/cunyuliu/rna-jepa/eval_decision/rnaformer_probs_ref_bprna_ts0.npz \
        --out-release-json /mnt/cunyuliu/rna-jepa/eval_decision/rnaformer_release_gt_ref_bprna_ts0.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "eval"),
           os.path.join(_REPO, "data", "ss")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RNAFORMER_SRC = "/mnt/cunyuliu/rna_baselines_src/RNAformer-code"
if RNAFORMER_SRC not in sys.path:
    sys.path.insert(0, RNAFORMER_SRC)

# project helpers reused verbatim (never rewritten): the pandas-1.x index shim, the
# .plk frame loader, the raw pair extractor and the dot-bracket renderer.
_pdd_spec = importlib.util.spec_from_file_location(
    "prepare_decision_data", os.path.join(_REPO, "data", "ss", "prepare_decision_data.py"))
pdd = importlib.util.module_from_spec(_pdd_spec)
_pdd_spec.loader.exec_module(pdd)

from rnajepa.harness import valid_pair_mask  # noqa: E402
from ss.metrics import PairLevelMetrics, StructureLevelMetrics  # noqa: E402

SEQ_VOCAB = ["A", "C", "G", "U", "N"]
SEQ_STOI = {c: i for i, c in enumerate(SEQ_VOCAB)}
DEFAULT_PLK = "/mnt/cunyuliu/rna-jepa/refmodels/datasets/test_sets.plk"
JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"


def read_jsonl_records(split: str, limit: int = 0) -> List[Tuple[str, str, List[Tuple[int, int]]]]:
    """``(name, seq, pairs)`` in file order -- the order the scorer validates against."""
    out = []
    with open(os.path.join(JSONL_DIR, f"{split}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = str(rec["seq"]).upper().replace("T", "U")
            pairs = [tuple(int(v) for v in p) for p in rec["pairs"]]
            out.append((rec["name"], seq, sorted(pairs)))
            if limit and len(out) >= limit:
                break
    return out


def load_release_frames(plk: str) -> Dict[str, object]:
    return {k: v for k, v in pdd._plk_frames(plk)}


def release_row(frames: Dict[str, object], split: str, name: str, seq: str):
    """Find the release ``.plk`` row a JSONL record came from, by name then by seq."""
    key = split[4:] if split.startswith("ref_") else split            # ref_pdb_ts3 -> pdb_ts3
    frame = frames.get(key)
    if frame is None:
        raise SystemExit(f"split {split!r}: no {key!r} frame in the .plk")
    prefix = key + "#"
    if name.startswith(prefix):
        idx = int(name[len(prefix):])
        if idx in frame.index:
            return frame.loc[idx]
    # fallback: unique sequence match
    hits = [r for _, r in frame.iterrows()
            if "".join(r["sequence"]).upper().replace("T", "U") == seq]
    if len(hits) != 1:
        raise SystemExit(f"split {split!r}: cannot locate release row for {name!r} "
                         f"({len(hits)} sequence matches)")
    return hits[0]


def build_model(args):
    import torch
    from RNAformer.model.RNAformer import RiboFormer
    from RNAformer.utils.configuration import Config
    from evaluate_RNAformer import insert_lora_layer

    sd = torch.load(args.state_dict, map_location="cpu")
    cfg = Config(config_file=args.config)

    # The inter-family finetuned checkpoint was trained with recycling (its
    # state_dict carries ``recycle_pair_norm.*``) even though its shipped config
    # says ``cycling: false``; without enabling it strict load fails on those keys.
    need_cycling = any(k.startswith("recycle_pair_norm.") for k in sd)
    if need_cycling:
        cfg.RNAformer.cycling = args.cycling if args.cycling > 0 else 6
    elif args.cycling > 0:
        cfg.RNAformer.cycling = args.cycling

    model = RiboFormer(cfg.RNAformer)
    if getattr(cfg, "lora", False):
        model = insert_lora_layer(model, cfg)

    mk, sk = set(model.state_dict()), set(sd)
    load_info = {"n_model_keys": len(mk), "n_state_dict_keys": len(sk),
                 "missing": sorted(mk - sk), "unexpected": sorted(sk - mk),
                 "cycling": int(getattr(cfg.RNAformer, "cycling", 0) or 0),
                 "lora": bool(getattr(cfg, "lora", False))}
    model.load_state_dict(sd, strict=True)
    load_info["strict_load"] = True
    return model, load_info


def decode_pairs(prob: np.ndarray, seq: str, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """Greedy non-crossing selection over canonical candidates above ``threshold``."""
    mask = valid_pair_mask(seq)
    L = len(seq)
    cand = [(i, j) for i in range(L) for j in range(i + 1, L)
            if mask[i, j] and prob[i, j] > threshold]
    cand.sort(key=lambda p: -prob[p[0], p[1]])
    chosen: List[Tuple[int, int]] = []
    for i, j in cand:
        if any((i < k < j < l) or (k < i < l < j) or k == i or k == j or l == i or l == j
               for k, l in chosen):
            continue
        chosen.append((i, j))
    return sorted(chosen)


def aggregate(records: Sequence[Tuple[Sequence[Tuple[int, int]],
                                     Sequence[Tuple[int, int]], str]]) -> Dict[str, object]:
    """micro/macro F1 + INF via ``ss.metrics`` -- the same aggregation as score_all."""
    tp = fp = fn = 0
    per: List[Tuple[float, float]] = []
    for pred, gt, seq in records:
        mask = valid_pair_mask(seq)
        m = PairLevelMetrics.from_pairs(pred, gt, L=len(seq), mask=mask)
        s = StructureLevelMetrics.from_pairs(pred, gt)
        tp += m.tp
        fp += m.fp
        fn += m.fn
        per.append((m.f1, s.inf))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": micro,
            "macro_f1": float(statistics.fmean([p[0] for p in per])) if per else 0.0,
            "inf": float(statistics.fmean([p[1] for p in per])) if per else 0.0,
            "n_sequences": len(records), "tp": tp, "fp": fp, "fn": fn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dict", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="project split, e.g. ref_bprna_ts0")
    ap.add_argument("--plk", default=DEFAULT_PLK)
    ap.add_argument("--out-dbn", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-release-json", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="", help="cuda / cuda:6 / cpu; default auto")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--cycling", type=int, default=0, help="override recycle steps (0=config/auto)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    import torch

    model, load_info = build_model(args)
    print(f"[rnaf] load: {load_info}", flush=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if dtype is not torch.float32 and device.startswith("cuda"):
        model = model.to(dtype)
    model.eval()

    records = read_jsonl_records(args.split, args.limit)
    print(f"[rnaf] {args.split}: {len(records)} project records", flush=True)
    frames = load_release_frames(args.plk)

    seqs: List[str] = []
    probs: List[np.ndarray] = []
    pred_pairs: List[List[Tuple[int, int]]] = []
    release_gt: List[List[Tuple[int, int]]] = []
    jsonl_gt: List[List[Tuple[int, int]]] = []
    raw_pair_counts = {"release": 0, "project": 0}

    t0 = time.time()
    with torch.no_grad():
        for k, (name, seq, pairs) in enumerate(records, 1):
            row = release_row(frames, args.split, name, seq)
            rseq = "".join(row["sequence"]).upper().replace("T", "U")
            if rseq != seq:
                raise SystemExit(f"{name}: release seq != project seq (len {len(rseq)} vs {len(seq)})")
            rpairs, _labels = pdd._row_pairs(dict(row), len(seq))
            raw_pair_counts["release"] += len(rpairs)
            raw_pair_counts["project"] += len(pairs)

            ids = torch.LongTensor([SEQ_STOI.get(c, 4) for c in seq])[None, :].to(device)
            src_len = torch.LongTensor([len(seq)]).to(device)
            pdb_sample = torch.ones((1, 1), dtype=dtype if device.startswith("cuda") else torch.float32,
                                    device=device)
            logits, _pair_mask = model(ids, src_len, pdb_sample)
            p = torch.sigmoid(logits[0, :, :, -1]).float().cpu().numpy().astype(np.float64)
            p = 0.5 * (p + p.T)
            np.fill_diagonal(p, 0.0)
            p = np.clip(p, 0.0, 1.0)

            seqs.append(seq)
            probs.append(p)
            pred_pairs.append(decode_pairs(p, seq, args.threshold))
            release_gt.append(sorted(rpairs))
            jsonl_gt.append(pairs)
            if k % 100 == 0:
                print(f"[rnaf]   {k}/{len(records)} ({(time.time() - t0) / k:.2f}s/seq)", flush=True)

    # ---- dbn FASTA (project scorer reads this; order == project JSONL order) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_dbn)), exist_ok=True)
    with open(args.out_dbn, "w", encoding="utf-8") as fh:
        for (name, seq, _pairs), pp in zip(records, pred_pairs):
            fh.write(f">{name}\n{seq}\n{pdd.pairs_to_dotbracket(len(seq), pp)}\n")
    print(f"[rnaf] wrote {args.out_dbn}", flush=True)

    # ---- probability shards (load_teacher_shard layout) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez(args.out_npz, sequences=np.array(seqs, dtype=object),
             probs=np.array(probs, dtype=object))
    print(f"[rnaf] wrote {args.out_npz}", flush=True)

    result = {
        "split": args.split, "state_dict": args.state_dict, "config": args.config,
        "precision": args.precision, "device": device, "load_info": load_info,
        "prob_convention": "sigmoid(logits[...,-1]), symmetrised, zero diagonal, float64 (post-sigmoid)",
        "decode": f"greedy non-crossing over valid_pair_mask(seq) & (P>{args.threshold}), desc prob",
        "n_sequences": len(records),
        "n_predicted_pairs": int(sum(len(p) for p in pred_pairs)),
        "raw_pair_counts": raw_pair_counts,
        "metrics_release_full_gt": aggregate(list(zip(pred_pairs, release_gt, seqs))),
        "metrics_project_gt": aggregate(list(zip(pred_pairs, jsonl_gt, seqs))),
    }
    print(f"[rnaf] release-full GT: {result['metrics_release_full_gt']}", flush=True)
    print(f"[rnaf] project   GT  : {result['metrics_project_gt']}", flush=True)

    if args.out_release_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_release_json)), exist_ok=True)
        with open(args.out_release_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        print(f"[rnaf] wrote {args.out_release_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())