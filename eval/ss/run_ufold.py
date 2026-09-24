#!/usr/bin/env python3
"""Self-contained UFold baseline adapter.

UFold's own scripts (``ufold_test.py`` / ``ufold_predict.py``) cannot run here:
they need ``data/TS0.cPickle`` and ``data/ArchiveII.pickle`` (absent -- ``data/``
holds only ``Readme.md`` and ``input.txt``) and ``models/ufold_train.pt`` (a
different filename).  The weights we *do* have are ``models/ufold_train_alldata.pt``.

This adapter reuses UFold's own code for the parts that define the model --

* ``Network.U_Net``                      -- the network,
* ``ufold.config.process_config``        -- the config loader,
* ``ufold.data_generator``               -- ``perm`` / ``get_cut_len`` / ``creatmat``,
* ``ufold.postprocess.postprocess_new``  -- the one-norm decode/refinement,

and bypasses only the missing *data* pipeline by encoding this project's JSONL
corpus with UFold's own encoding recipe (see ``Dataset_Cut_concat_new.__getitem__``).

Per split it emits, in split order:

* ``<out-dir>/ufold_<split>.dbn``        -- FASTA (``>name``/``seq``/dot-bracket``)
  that ``eval/ss/run_baselines.py --external-dbn`` scores directly;
* ``<out-dir>/ufold_<split>_probs.npz``  -- ``sequences`` + ``probs`` object arrays,
  exactly the format ``src/rnajepa/distill.load_teacher_shard`` reads.  ``probs[i]``
  is the ``L x L`` ``sigmoid`` of the raw network output kept on the strict upper
  triangle (diagonal and below zeroed), the same convention as ViennaTeacher;
* ``<out-dir>/ufold_<split>.json``       -- micro/macro F1 + a probability-matrix
  validity audit + the ``state_dict`` load report.

Usage
-----
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/cunyuliu/pylibs:src TMPDIR=/mnt/cunyuliu/tmp \
      python eval/ss/run_ufold.py --split ref_bprna_ts0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# --- UFold's own code lives outside this repo; reuse it, do not fork it. -----
UFOLD_SRC = "/home/cunyuliu/rna_baselines_src/UFold-main"
UFOLD_WEIGHTS = os.path.join(UFOLD_SRC, "models", "ufold_train_alldata.pt")
UFOLD_CONFIG = os.path.join(UFOLD_SRC, "ufold", "config.json")
if UFOLD_SRC not in sys.path:
    sys.path.insert(0, UFOLD_SRC)

# ufold/config.py imports `munch`, which is not installed in this cluster env and
# is not needed for anything except attribute access on the parsed config dict.
# Ship a minimal in-process shim (a dict with attribute access) so UFold's own
# process_config runs unmodified.  This touches no file outside this adapter.
try:                                               # pragma: no cover - env probe
    import munch  # noqa: F401
except ModuleNotFoundError:
    import types as _types

    class _Munch(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc

        def __setattr__(self, k, v):
            self[k] = v

    _shim = _types.ModuleType("munch")
    _shim.Munch = _Munch
    sys.modules["munch"] = _shim

from Network import U_Net                          # noqa: E402  UFold's network
from ufold.config import process_config            # noqa: E402  UFold's config
from ufold.postprocess import postprocess_new      # noqa: E402  UFold's decode
from ufold.data_generator import (                 # noqa: E402  UFold's encoding
    creatmat,
    get_cut_len,
    perm,
)

JSONL_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"
OUT_DIR = "/mnt/cunyuliu/rna-jepa/eval_decision"
BASES = "AUCG"

# UFold's own inference call (ufold_predict.py / ufold_test.py), verbatim:
#   postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
#   map_no_train = (u_no_train > 0.5)
POSTPROCESS_ARGS = dict(lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6,
                        with_l1=True, s=1.5)
POSTPROCESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# corpus I/O (this project's JSONL format)
# ---------------------------------------------------------------------------
def read_records(path: str, limit: int = 0) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = str(rec["seq"]).upper().replace("T", "U")
            pairs = sorted((int(i), int(j)) for i, j in rec["pairs"])
            out.append({"name": str(rec.get("name", f"seq{len(out)}")),
                        "seq": seq, "pairs": pairs})
            if limit and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# UFold's input encoding (mirrors Dataset_Cut_concat_new.__getitem__)
# ---------------------------------------------------------------------------
def one_hot(seq: str) -> np.ndarray:
    """``L x 4`` one-hot over ``AUCG`` (unknown -> all -1), as UFold's one_hot_600."""
    feat = np.zeros((len(seq), 4), dtype=np.float64)
    for k, ch in enumerate(seq):
        idx = BASES.find(ch)
        if idx < 0:
            feat[k, :] = -1.0
        else:
            feat[k, idx] = 1.0
    return feat


def build_input(seq: str, device: torch.device):
    """Return UFold's 17-channel input ``(1, 17, l, l)`` and the padded one-hot."""
    length = len(seq)
    if any(ch not in BASES for ch in seq):
        bad = sorted(set(seq) - set(BASES))
        raise ValueError(f"sequence {seq[:20]!r}... contains non-ACGU chars {bad}")
    data_seq = one_hot(seq)
    l = get_cut_len(length, 80)                       # UFold's own padding rule

    data_fcn = np.zeros((16, l, l), dtype=np.float64)
    for n, (i, j) in enumerate(perm):                 # 16 = 4 x 4 base-pair channels
        data_fcn[n, :length, :length] = np.outer(data_seq[:, i], data_seq[:, j])
    data_fcn_1 = np.zeros((1, l, l), dtype=np.float64)
    data_fcn_1[0, :length, :length] = creatmat(data_seq, device=device).cpu().numpy()
    x = np.concatenate([data_fcn, data_fcn_1], axis=0)  # (17, l, l), 17 = img_ch

    seq_pad = np.zeros((l, 4), dtype=np.float64)
    seq_pad[:length] = data_seq
    return (torch.from_numpy(x).unsqueeze(0).float().to(device),
            torch.from_numpy(seq_pad).unsqueeze(0).float().to(device), length, l)


# ---------------------------------------------------------------------------
# UFold's decode (get_ct_dict_fast + seq2dot, verbatim rule)
# ---------------------------------------------------------------------------
def seq2dot(seq: np.ndarray) -> str:
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(["_"] * len(seq))
    dot_file[seq > idx] = "("
    dot_file[seq < idx] = ")"
    dot_file[seq == 0] = "."
    return "".join(dot_file)


def _is_balanced(dot: str) -> bool:
    depth = 0
    for ch in dot:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def decode_dot(predict_matrix: torch.Tensor):
    """Binary ``L x L`` contact map -> dot-bracket using UFold's own rule.

    Returns ``(raw_dot, consistent_dot, n_directed, n_mutual)``.

    ``raw_dot`` is UFold's ``seq2dot(partner + 1)`` verbatim.  UFold's decode picks,
    per position, the ``argmax`` partner, which is *not* guaranteed to be symmetric,
    so the raw string is unbalanced for a small fraction of sequences (~2.6% on
    ref_bprna_ts0) and the project scorer's strict ``parse_pairs`` rejects it.  The
    ``consistent_dot`` keeps a decoded pair only when it is *mutual*
    (``partner[j] == i``) -- UFold's own decision with the minimal repair that makes
    the string a valid structure.  Both counts are reported for transparency.
    """
    pm = predict_matrix.detach().cpu().float()
    length = pm.shape[0]
    m = pm.view(1, length, length)                    # match UFold's (1, L, L) shape
    row_sum = m.sum(axis=1)
    partner = torch.mul(m.argmax(axis=1), row_sum.clamp_max(1)).squeeze(0).numpy().astype(int)
    partner[row_sum.squeeze(0).numpy() == 0] = -1
    raw_dot = seq2dot(np.atleast_1d(partner + 1))

    pairs = [(int(i), int(j)) for i, j in enumerate(partner)
             if 0 <= j < length and j != i and i < j and partner[j] == i]
    chars = ["."] * length
    for i, j in pairs:
        chars[i] = "("
        chars[j] = ")"
    return raw_dot, "".join(chars), int((partner >= 0).sum()), len(pairs)


def parse_dot_pairs(dot: str) -> List[Tuple[int, int]]:
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(dot):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                pairs.append((stack.pop(), i))
    return pairs


# ---------------------------------------------------------------------------
# scoring (exact base-pair micro/macro F1; the authoritative number comes from
# eval/ss/run_baselines.py over the emitted .dbn)
# ---------------------------------------------------------------------------
def micro_macro(preds: Sequence[List[Tuple[int, int]]],
                gts: Sequence[List[Tuple[int, int]]]) -> Dict[str, float]:
    tp = fp = fn = 0
    per_seq = []
    for pred, gt in zip(preds, gts):
        p, g = set(pred), set(gt)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
        f1 = 2 * len(p & g) / (len(p) + len(g)) if (p or g) else 1.0
        per_seq.append(f1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"micro_precision": precision, "micro_recall": recall,
            "micro_f1": micro_f1,
            "macro_f1": float(np.mean(per_seq)) if per_seq else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--jsonl-dir", default=JSONL_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-mode", action="store_true",
                    help="use model.eval() instead of UFold's own model.train()")
    args = ap.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_path = os.path.join(args.jsonl_dir, f"{args.split}.jsonl")
    records = read_records(jsonl_path, args.limit)
    print(f"[ufold] {args.split}: {len(records)} sequences from {jsonl_path}", flush=True)

    config = process_config(UFOLD_CONFIG)
    print(f"[ufold] UFold config loaded (gpu={config.gpu}, "
          f"batch_size_stage_1={config.batch_size_stage_1})", flush=True)

    model = U_Net(img_ch=17)
    state = torch.load(UFOLD_WEIGHTS, map_location="cpu", weights_only=True)
    model_keys, ckpt_keys = set(model.state_dict()), set(state)
    load_report = {
        "weights": UFOLD_WEIGHTS,
        "n_ckpt_keys": len(ckpt_keys),
        "n_model_keys": len(model_keys),
        "missing_in_ckpt": sorted(model_keys - ckpt_keys),
        "unexpected_in_ckpt": sorted(ckpt_keys - model_keys),
    }
    try:
        model.load_state_dict(state, strict=True)
        load_report["load"] = "strict=True OK (all keys matched)"
    except RuntimeError as exc:                       # pragma: no cover - safety net
        load_report["load"] = f"strict=True FAILED: {exc}"
        model.load_state_dict(state, strict=False)
        load_report["load"] += " -> retried with strict=False"
    print(f"[ufold] weights: {load_report['load']} "
          f"(ckpt={load_report['n_ckpt_keys']}, model={load_report['n_model_keys']}, "
          f"missing={load_report['missing_in_ckpt']}, "
          f"unexpected={load_report['unexpected_in_ckpt']})", flush=True)
    model.to(device)
    # UFold's own inference scripts run the net in train() mode; replicate it.
    model.train() if not args.eval_mode else model.eval()

    dots: List[str] = []
    raw_dots: List[str] = []
    seqs: List[str] = []
    probs_list: List[np.ndarray] = []
    pred_pairs: List[List[Tuple[int, int]]] = []
    gt_pairs: List[List[Tuple[int, int]]] = []

    # probability-matrix audit accumulators
    p_min, p_max, max_asym, max_diag = 1.0, 0.0, 0.0, 0.0
    sum_upper, n_gt_pairs = [], []
    # decode audit accumulators
    raw_unbalanced = 0
    directed_pairs = 0
    mutual_pairs = 0
    t0 = time.time()

    for k, rec in enumerate(records, 1):
        seq = rec["seq"]
        with torch.no_grad():
            x, seq_pad, length, l = build_input(seq, device)
            logits = model(x)                          # (1, l, l), UFold "utility"
            u = postprocess_new(logits, seq_pad, **POSTPROCESS_ARGS)  # (1, l, l)
        contact = (u[0, :length, :length] > POSTPROCESS_THRESHOLD).float()
        raw_dot, dot, n_directed, n_mutual = decode_dot(contact)
        raw_unbalanced += int(not _is_balanced(raw_dot))
        directed_pairs += n_directed
        mutual_pairs += n_mutual

        probs_full = torch.sigmoid(logits[0, :length, :length]).double().cpu().numpy()
        p_np = probs_full
        p_min = min(p_min, float(p_np.min()))
        p_max = max(p_max, float(p_np.max()))
        max_asym = max(max_asym, float(np.abs(p_np - p_np.T).max()))
        max_diag = max(max_diag, float(np.abs(np.diag(p_np)).max()))
        upper = np.triu(p_np, k=1)                     # teacher-shard convention
        sum_upper.append(float(upper.sum()))
        n_gt_pairs.append(len(rec["pairs"]))

        dots.append(dot)
        raw_dots.append(raw_dot)
        seqs.append(seq)
        probs_list.append(upper)
        pred_pairs.append(parse_dot_pairs(dot))
        gt_pairs.append(rec["pairs"])

        if k % 100 == 0 or k == len(records):
            print(f"[ufold] {k}/{len(records)} "
                  f"({(time.time() - t0) / k:.3f}s/seq)", flush=True)

    # --- emit dot-bracket (FASTA: >name / seq / dot) for run_baselines.py -----
    # Primary file is the mutual-consistent decode (valid structure, scorable).
    dbn_path = os.path.join(args.out_dir, f"ufold_{args.split}.dbn")
    with open(dbn_path, "w", encoding="utf-8") as fh:
        for rec, dot in zip(records, dots):
            fh.write(f">{rec['name']}\n{rec['seq']}\n{dot}\n")
    # Raw UFold decode kept for audit (may be unbalanced; NOT the scored file).
    raw_dbn_path = os.path.join(args.out_dir, f"ufold_{args.split}.raw_ufold_decode.dbn")
    with open(raw_dbn_path, "w", encoding="utf-8") as fh:
        for rec, dot in zip(records, raw_dots):
            fh.write(f">{rec['name']}\n{rec['seq']}\n{dot}\n")

    # --- emit probability .npz in load_teacher_shard's format ----------------
    npz_path = os.path.join(args.out_dir, f"ufold_{args.split}_probs.npz")
    np.savez(npz_path,
             sequences=np.array(seqs, dtype=object),
             probs=np.array(probs_list, dtype=object))

    metrics = micro_macro(pred_pairs, gt_pairs)
    su = np.asarray(sum_upper)
    gp = np.asarray(n_gt_pairs)
    summary = {
        "split": args.split,
        "adapter": "eval/ss/run_ufold.py",
        "ufold_src": UFOLD_SRC,
        "weights": load_report,
        "postprocess": {**POSTPROCESS_ARGS, "threshold": POSTPROCESS_THRESHOLD},
        "model_mode": "eval" if args.eval_mode else "train (as in UFold's own scripts)",
        "n_sequences_in_split": len(records),
        "n_sequences_predicted": len(dots),
        "metrics": metrics,
        "prob_matrix_audit": {
            "convention": "sigmoid(network_output), strict upper triangle (diag+below=0)",
            "min": p_min, "max": p_max,
            "max_asymmetry_max_abs": max_asym,
            "max_abs_value_on_diagonal": max_diag,
            "mean_sum_upper": float(su.mean()),
            "mean_gt_n_pairs": float(gp.mean()),
            "corr_sum_upper_vs_gt_pairs": float(np.corrcoef(su, gp)[0, 1])
            if len(su) > 1 and su.std() > 0 and gp.std() > 0 else None,
        },
        "decode_audit": {
            "rule": "ufold.get_ct_dict_fast argmax partner + seq2dot",
            "raw_unbalanced_sequences": raw_unbalanced,
            "directed_pairs_total": directed_pairs,
            "mutual_pairs_total": mutual_pairs,
            "note": ("raw dot-bracket is unbalanced for some sequences (asymmetric "
                     "argmax); the scored .dbn uses the mutual-consistent decode"),
        },
        "artifacts": {"dbn": dbn_path, "raw_dbn": raw_dbn_path, "npz": npz_path},
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(args.out_dir, f"ufold_{args.split}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    print(f"[ufold] micro_f1={metrics['micro_f1']:.4f} ext_prec={metrics['micro_precision']:.4f} "
          f"ext_rec={metrics['micro_recall']:.4f} macro_f1={metrics['macro_f1']:.4f}",
          flush=True)
    print(f"[ufold] wrote {dbn_path}\n[ufold] wrote {npz_path}\n[ufold] wrote {json_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())