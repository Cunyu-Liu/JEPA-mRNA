"""Plan-A evaluation: TS0 / bpRNA-new micro F1 with the in-loop giga backbone.

The stock evaluator (eval/ss/evaluate_decision.py) rebuilds models via
``build_decision_model`` and looks embeddings up in the cache -- it has no path
for a checkpoint whose 650M backbone is part of the model. This script mirrors
its scoring protocol exactly and swaps only the representation source:

* decode: ``nussinov_map`` over the head's raw scores with ``-inf`` on illegal
  pairs (identical to the r2d arm's evaluation, whose ``--prior-weight -1``
  leaves the scores untouched: the trained ``prior_weight`` and calibration
  temperature are already inside the head);
* forward: ``<cls>+seq+<eos>`` + attention_mask, fp16-quantised residue slice
  ``h[:, 1:L+1]`` -- the exact frame the arm trained on (and the one the frozen
  control arm's cached features were extracted with);
* metrics: PairLevelMetrics pooled micro precision/recall/F1, the numbers the
  draft's leaderboards quote;
* sequences longer than RiNALMo's rotary limit (1024) are skipped and logged,
  the same explicit-exclusion convention as the embedding extractor.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/cunyuliu/rna-jepa")
sys.path.insert(0, str(REPO))        # for eval.ss.metrics (package root)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX  # noqa: E402
from rnajepa.decision_head import FlatDecisionHead  # noqa: E402
from rnajepa.harness import nussinov_map, valid_pair_mask  # noqa: E402
from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402
from eval.ss.metrics import PairLevelMetrics  # noqa: E402


def load_head_and_encoder(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    encoder.load_state_dict(ck["model"]["encoder"])
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    head.load_state_dict(ck["model"]["head"])
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    return encoder, head, ck.get("meta", {}), report


@torch.no_grad()
def scores_for(encoder, head, seq, device):
    L = len(seq)
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] \
        + [VOCAB["<eos>"]]
    enc_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(enc_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(input_ids=enc_ids, attention_mask=attn)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h[:, 1:L + 1, :].float().to(torch.float16).to(torch.float32)
    head_ids = torch.tensor(
        [[BASE_TO_INDEX.get(c, UNK_INDEX) for c in seq]], dtype=torch.long,
        device=device)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    s = head(h, head_ids, lengths=lengths).scores
    return s[0, :L, :L].float()


DUMP_PER_SEQ = False


def evaluate_split(encoder, head, data_path, device, max_pos=1024):
    records = []
    names = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = str(r["seq"]).upper().replace("T", "U")
            pairs = [tuple(p) for p in r["pairs"]] if "pairs" in r else []
            records.append((seq, [tuple(p) for p in pairs]))
            names.append(str(r.get("name", "")))
    skipped = [(s, len(s)) for s, _ in records if len(s) > max_pos]
    keep_idx = [i for i, (s, _) in enumerate(records) if len(s) <= max_pos]
    records = [records[i] for i in keep_idx]
    names = [names[i] for i in keep_idx]

    agg = {"tp": 0, "fp": 0, "fn": 0}
    per_seq = []
    t0 = time.time()
    for i, (seq, gt_pairs) in enumerate(records):
        s = scores_for(encoder, head, seq, device)
        mask = valid_pair_mask(seq)
        s_np = s.cpu().numpy()
        s_np = np.where(mask, s_np, -np.inf)
        pred_pairs = [tuple(p) for p in nussinov_map(s_np, mask)]
        pm = PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq),
                                         mask=mask)
        agg["tp"] += pm.tp
        agg["fp"] += pm.fp
        agg["fn"] += pm.fn
        per_seq.append({"length": len(seq), "f1": pm.f1,
                        "n_pred": len(pred_pairs), "n_gt": len(gt_pairs)})
        if (i + 1) % 200 == 0:
            print(f"[eval-plan-a] {i+1}/{len(records)} "
                  f"({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    macro = float(np.mean([p["f1"] for p in per_seq])) if per_seq else 0.0
    out = {"n_sequences": len(records), "n_skipped_over_pos": len(skipped),
           "skipped_lengths": [l for _, l in skipped],
           "micro": {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn},
           "macro_f1": macro}
    if DUMP_PER_SEQ:
        out["per_sequence"] = [
            {"name": n, **row}
            for n, row in zip(names, per_seq)
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="ts0,new")
    ap.add_argument("--dump-per-seq", action="store_true",
                    help="serialise the per-sequence rows already computed "
                         "(name from the jsonl, f1/length/n_pred/n_gt); no "
                         "protocol change, only extra output")
    args = ap.parse_args()

    device = args.device
    encoder, head, meta, report = load_head_and_encoder(args.checkpoint, device)
    print(f"[eval-plan-a] checkpoint {args.checkpoint} step={meta.get('schedule', {}).get('steps')}",
          flush=True)

    split_files = {"ts0": "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl",
                   "new": "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_new.jsonl"}
    global DUMP_PER_SEQ
    DUMP_PER_SEQ = bool(args.dump_per_seq)
    results = {}
    for name in args.splits.split(","):
        path = split_files[name]
        print(f"[eval-plan-a] evaluating {name} <- {path}", flush=True)
        results[name] = evaluate_split(encoder, head, path, device)
        print(f"[eval-plan-a] {name}: micro F1 "
              f"{results[name]['micro']['f1']:.4f} "
              f"(P {results[name]['micro']['precision']:.4f} / "
              f"R {results[name]['micro']['recall']:.4f}, "
              f"{results[name]['n_sequences']} seqs)", flush=True)

    stem = Path(args.checkpoint).stem
    out = {"tag": stem,
           "checkpoint": args.checkpoint,
           "arm": stem,
           "control": "rinalmo_r2d_b4_s0 (TS0 0.6629 / new 0.5010)",
           "protocol": "nussinov_map decode, prior_weight=-1 (model's own), "
                       "fp16-quantised in-loop frame, same as training",
           "splits": results, "meta": meta}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[eval-plan-a] written {args.out}", flush=True)


if __name__ == "__main__":
    main()
