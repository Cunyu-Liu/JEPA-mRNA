"""Evaluate structRFM SSP (bpRNA1m fine-tuned) on our benchmarks.

Uses the model's own inference path (structRFM_SSP.py logic):
  structRFMForSsp (frozen LM embeddings) -> MixedFold (CNN+LSTM pair scorer
  + Turner-parameters + predict_mxfold DP decode) -> bpseq connects.

Scoring: BOTH conventions reported --
  ours:    strict micro-F1 (pooled TP/FP/FN), like every other row in the draft
  theirs:  their SspMetrics semantics (strict per-seq F1, macro-averaged) and,
           for comparability with RiNALMo-paper numbers, the Mathews-tolerant
           macro-F1 from tools/rescore_mathews.py applied to their .dbn output.
Input: jsonl with {"seq"/"sequence", "pairs", "id"}; output: per-seq pairs +
summary json.

Run inside structRFM repo root with tasks/seqcls_ssp on sys.path, python 3.8,
torch <= 1.13 (interface .so is cpython-38).
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent

import transformers.data.data_collator as _dc
if not hasattr(_dc, "pad_without_fast_tokenizer_warning"):
    def pad_without_fast_tokenizer_warning(features, tokenizer, **kw):
        from transformers.data.data_collator import DataCollatorWithPadding
        keep = {k: v for k, v in kw.items()
                if k in ("padding", "max_length", "pad_to_multiple_of", "return_tensors")}
        return DataCollatorWithPadding(tokenizer, **keep)(features)
    _dc.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "structRFM-master" / "src"))
    sys.path.insert(0, str(REPO / "structRFM-master" / "tasks" / "seqcls_ssp"))

    from structRFM.model import get_structRFM, get_model_scale
    from structRFM.data import get_mlm_tokenizer
    from ss_pred import structRFMForSsp, MixedFold
    import param_turner2004

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = get_mlm_tokenizer(max_length=514)
    model_paras = get_model_scale("base")
    LM = get_structRFM(from_pretrained=None, output_hidden_states=True,
                       tokenizer=tokenizer, **model_paras)
    pretrained_model = structRFMForSsp(LM)

    config = {
        "max_helix_length": 30,
        "embed_size": 64,
        "num_filters": (64, 64, 64, 64, 64, 64, 64, 64),
        "filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
        "pool_size": (1,),
        "dilation": 0,
        "num_lstm_layers": 2,
        "num_lstm_units": 32,
        "num_transformer_layers": 0,
        "num_hidden_units": (32,),
        "num_paired_filters": (64, 64, 64, 64, 64, 64, 64, 64),
        "paired_filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
        "dropout_rate": 0.5,
        "fc_dropout_rate": 0.5,
        "num_att": 8,
        "pair_join": "cat",
        "no_split_lr": False,
        "n_out_paired_layers": 3,
        "n_out_unpaired_layers": 0,
        "exclude_diag": True,
        "embed_dim": 768,
    }
    model = MixedFold(init_param=param_turner2004, **config)

    data = torch.load(args.ckpt, map_location="cpu")
    assert "model" in data, "unexpected checkpoint layout"
    model.load_state_dict(data["model"])
    if "pretrained_model" in data:
        pretrained_model.load_state_dict(data["pretrained_model"])
    print(f"loaded {args.ckpt}")

    model.to(device).eval()
    pretrained_model.to(device).eval()

    records = [json.loads(l) for l in open(args.data)]
    if args.limit:
        records = records[: args.limit]
    n_long = sum(1 for r in records if len(r.get("seq") or r.get("sequence")) > args.max_len - 1)
    print(f"n={len(records)} (over {args.max_len}: {n_long}), device={device}")

    strict_tp = strict_fp = strict_fn = 0
    per_seq = []
    seqs_out = []

    with torch.no_grad():
        for k, r in enumerate(records):
            seq = (r.get("seq") or r.get("sequence")).upper().replace("T", "U")
            L = len(seq)
            seq_trunc = seq[: args.max_len - 1]

            kmer_text = "[CLS]" + seq_trunc
            input_ids = tokenizer(kmer_text)["input_ids"][: args.max_len]
            input_ids = [0 if x is None else x for x in input_ids]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            embeddings = pretrained_model(input_tensor)
            scs, preds, bps = model((seq_trunc,), embeddings)
            connects = bps[0][1:]  # 1-based: connects[i-1] = 1-based partner of 1-based pos i

            # project jsonl pairs are 0-based -> shift to 1-based for comparison
            gt_pairs = {(i + 1, j + 1) for i, j in r.get("pairs", []) if i < j}
            pred_pairs = set()
            for i, j in enumerate(connects):
                if j > 0:
                    a, b = i + 1, int(j)
                    pred_pairs.add((min(a, b), max(a, b)))
            if L > len(seq_trunc):
                pass

            tp = len(gt_pairs & pred_pairs)
            fp = len(pred_pairs - gt_pairs)
            fn = len(gt_pairs - pred_pairs)
            strict_tp += tp; strict_fp += fp; strict_fn += fn
            per_seq.append({"id": r.get("id") or k, "len": L, "tp": tp, "fp": fp, "fn": fn})
            seqs_out.append({"id": r.get("id") or k, "seq": seq, "connects": [int(x) for x in connects]})

            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(records)}", flush=True)

    prec = strict_tp / max(1, strict_tp + strict_fp)
    rec = strict_tp / max(1, strict_tp + strict_fn)
    f1m = 2 * prec * rec / max(1e-9, prec + rec)

    macro = []
    for p in per_seq:
        pr = p["tp"] / max(1, p["tp"] + p["fp"])
        rc = p["tp"] / max(1, p["tp"] + p["fn"])
        macro.append(2 * pr * rc / max(1e-9, pr + rc))
    macro_f = float(np.mean(macro)) if macro else 0.0

    out = {
        "n": len(records),
        "n_over_maxlen": n_long,
        "max_len": args.max_len,
        "our_protocol": {"strict_micro_f1": f1m, "precision": prec, "recall": rec},
        "their_protocol": {"strict_macro_f1": macro_f},
        "per_seq": per_seq,
        "seqs": seqs_out,
    }
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_seq", "seqs")}, indent=1))


if __name__ == "__main__":
    main()
