import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "RiNALMo-main"))

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct, ss_precision, ss_recall, ss_f1


def load_ft_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    threshold = state.pop("threshold", None)
    if threshold is not None:
        threshold = float(threshold)
    else:
        threshold = 0.5

    config = model_config("giga")
    if not HAS_FLASH:
        config["model"]["transformer"]["use_flash_attn"] = False
        print("flash_attn unavailable -> use_flash_attn=False")
    lm = RiNALMo(config)
    alphabet = Alphabet(**config["alphabet"])
    head = SecStructPredictionHead(
        config["model"]["transformer"].embed_dim, num_blocks=2
    )

    lm_keys = {k[len("lm."):]: v for k, v in state.items() if k.startswith("lm.")}
    head_keys = {k[len("pred_head."):]: v for k, v in state.items() if k.startswith("pred_head.")}
    if not lm_keys:  # bare RiNALMo checkpoint (no wrapper prefixes)
        lm_keys = {k: v for k, v in state.items() if not k.startswith("pred_head.")}
        head_keys = {k: v for k, v in state.items() if k.startswith("pred_head.")}
    missing_lm, unexpected_lm = lm.load_state_dict(lm_keys, strict=False)
    missing_h, unexpected_h = head.load_state_dict(head_keys, strict=False)
    print(f"load lm: {len(lm_keys)} keys, missing={len(missing_lm)}, unexpected={len(unexpected_lm)}")
    print(f"load head: {len(head_keys)} keys, missing={len(missing_h)}, unexpected={len(unexpected_h)}")
    print(f"threshold from ckpt: {threshold}")

    lm = lm.to(device).eval()
    head = head.to(device).eval()
    return lm, head, alphabet, threshold


def strict_prf_counts(gt_pairs, pred_pairs):
    tp = len(gt_pairs & pred_pairs)
    return tp, len(pred_pairs) - tp, len(gt_pairs) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).parent / "RiNALMo-main"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm, head, alphabet, threshold = load_ft_model(args.ckpt, device)

    sys.path.insert(0, str(Path(args.repo)))
    from rinalmo.utils.sec_struct import _relax_ss  # noqa

    records = [json.loads(l) for l in open(args.data)]
    if args.limit:
        records = records[: args.limit]
    print(f"n = {len(records)}, device = {device}")

    tol_f1s, tol_ps, tol_rs = [], [], []
    strict_tp = strict_fp = strict_fn = 0
    per_seq = []

    with torch.no_grad():
        for k, r in enumerate(records):
            seq = r["sequence"]
            L = len(seq)
            tokens = torch.tensor([alphabet.encode(seq)], dtype=torch.int64, device=device)
            x = lm(tokens)["representation"]
            logits = head(x[..., 1:-1, :]).squeeze(-1)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()

            # symmetrise upper triangle (their training uses upper-tri loss)
            probs = np.triu(probs, k=1)
            probs = probs + probs.T

            pred_mat = prob_mat_to_sec_struct(
                probs=np.copy(probs), seq=seq, threshold=threshold
            )

            gt_mat = np.zeros((L, L), dtype=int)
            for i, j in r.get("pairs", []):
                gt_mat[i, j] = gt_mat[j, i] = 1

            tol_p = ss_precision(gt_mat, pred_mat)
            tol_r = ss_recall(gt_mat, pred_mat)
            tol_f = ss_f1(gt_mat, pred_mat)
            tol_ps.append(tol_p)
            tol_rs.append(tol_r)
            tol_f1s.append(tol_f)

            gt_set = {(i, j) for i, j in r.get("pairs", []) if i < j}
            pred_set = {(i, j) for (i, j) in zip(*np.nonzero(np.triu(pred_mat, k=1)))}
            tp, fp, fn = strict_prf_counts(gt_set, pred_set)
            strict_tp += tp
            strict_fp += fp
            strict_fn += fn

            per_seq.append({
                "id": r.get("id") or r.get("name") or k,
                "len": L,
                "tol_f1": tol_f,
                "strict_tp": tp, "strict_fp": fp, "strict_fn": fn,
            })
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(records)}  tol-macro-so-far={np.mean(tol_f1s):.4f}")

    prec = strict_tp / max(1, strict_tp + strict_fp)
    rec = strict_tp / max(1, strict_tp + strict_fn)
    strict_micro = 2 * prec * rec / max(1e-9, prec + rec)

    out = {
        "n": len(records),
        "threshold": threshold,
        "rinalmo_protocol": {
            "tolerant_macro_f1": float(np.mean(tol_f1s)),
            "tolerant_macro_precision": float(np.mean(tol_ps)),
            "tolerant_macro_recall": float(np.mean(tol_rs)),
        },
        "our_protocol": {
            "strict_micro_f1": float(strict_micro),
            "strict_micro_precision": float(prec),
            "strict_micro_recall": float(rec),
        },
        "per_seq": per_seq,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_seq"}, indent=1))


if __name__ == "__main__":
    main()
