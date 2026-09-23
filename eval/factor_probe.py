"""Factor semantics probe for a pre-trained RNA-JEPA checkpoint (Fig.5 material).

Question the probe answers: **do the K orthogonal factors of the OPF basis carry
interpretable RNA semantics, or are they an arbitrary decomposition?**  The answer
decides whether claim C3 survives or whether Fig.5 degrades to a training-dynamics
panel, as the proposal's fallback path allows.

Method
------
1. Encode a pool of sequences drawn from the three mRNA regions (5'UTR-only, CDS-only,
   3'UTR-only tasks) with the checkpoint's encoder.
2. For every sequence build the factor coordinates: pool the region-aware summary,
   apply the region projection, then project through the learned basis ``P_k``.
3. Probe those coordinates for (a) the factor's own region identity and (b) the task
   family, using both a linear classifier (accuracy, balanced accuracy) and clustering
   (ARI, FMI).
4. **Random-subspace control.**  Repeat (3) with K random orthonormal projections of the
   same rank.  A linear probe is invariant to rotation, so comparing against a *rotated*
   basis would be vacuous; what actually needs controlling for is whether the learned
   r-dimensional subspaces are better than arbitrary r-dimensional subspaces.
5. Significance: permutation test (label shuffling) per probe, then Benjamini-Hochberg
   across all probes.  Nothing is reported without the control and the correction.

Outputs: ``factor_probe.json`` (all numbers), ``factor_coords.csv`` (coordinates plus
labels for t-SNE/UMAP figures), ``probe_report.md`` (human-readable summary).

Usage:
  python eval/factor_probe.py --ckpt <run>/hf/step_N --weights <weights_dir> \
      --data_root <extracted> --out <dir> --n_per_task 400
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# Tasks whose sequences are dominated by one region, used to give the probe a
# region label that is true by construction of the task, not by our annotation.
REGION_TASKS = {
    "5UTR": ["5UTR/Rank/U1", "5UTR/Rank/U2"],
    "CDS": ["CDS/mRFP", "CDS/Fungal"],
    "3UTR": ["3UTR/RNA_protein_interaction/22_eCLIP/0"],
}

# A *non-trivial* label set.  All three tasks are CDS with identical codon
# tokenisation, so tokenisation granularity cannot separate them: any probe accuracy
# above chance must come from sequence content.  The region pool above is kept as a
# sanity check on the machinery, but it is confounded by token length (codons vs
# single nucleotides) and must never be quoted as evidence for C3.
WITHIN_REGION_TASKS = {
    "CDS-mRFP": ["CDS/mRFP"],
    "CDS-Fungal": ["CDS/Fungal"],
    "CDS-Cov": ["CDS/Cov"],
    "CDS-ecoli": ["CDS/ecoli"],
}


def _preselect_device(dev: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)


def load_sequences(data_root: str, relpaths: Sequence[str], n_per_task: int,
                   seed: int) -> List[Tuple[List[int], List[int]]]:
    """Return ``(space_separated_tokens, per-token region ids)`` for each sequence.

    Region ids are recovered from the tokenisation itself: a three-character token is
    a codon (CDS), a one-character token is a nucleotide (UTR).  That is exact for the
    `codon` and `utr` tasks used here; the ambiguity that exists for `complete`
    sequences (a short trailing codon) does not arise because none of the pool tasks
    uses it.
    """
    from rnajepa.tokenization import REGION_3UTR, REGION_CDS, REGION_5UTR
    rng = random.Random(seed)
    out: List[Tuple[List[int], List[int]]] = []
    for rel in relpaths:
        path = os.path.join(data_root, rel, "train.csv")
        if not os.path.isfile(path):
            print(f"  WARN missing {path}")
            continue
        rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
        rows = [r for r in rows if len(r) >= 2 and r[0].strip()]
        rng.shuffle(rows)
        for row in rows[:n_per_task]:
            toks = row[0].split()
            regions = []
            seen_codon = False
            for t in toks:
                if len(t) >= 3:
                    regions.append(REGION_CDS)
                    seen_codon = True
                else:
                    regions.append(REGION_3UTR if seen_codon else REGION_5UTR)
            out.append((toks, regions))
    return out


def build_features(model, tokenizer, specials, samples, device, max_len: int):
    """Return factor coordinates ``[N, K, r_dim]`` and the sequence-level labels."""
    import numpy as np
    import torch
    from rnajepa.tokenization import REGION_NONE

    tok_map = tokenizer.get_vocab()
    feats, regions = [], []
    model.eval()
    with torch.no_grad():
        for toks, regs in samples:
            ids = [specials["cls"]] + [tok_map.get(t, specials["unk"]) for t in toks] \
                  + [specials["sep"]]
            rids = [REGION_NONE] + regs + [REGION_NONE]
            if len(ids) > max_len:
                ids = ids[:max_len - 1] + [specials["sep"]]
                rids = rids[:max_len - 1] + [REGION_NONE]
            ids_t = torch.tensor([ids], device=device)
            attn = torch.ones_like(ids_t)
            rids_t = torch.tensor([rids], device=device)
            h = model.student.encode(ids_t, attn)

            # region summary in the same way the objective forms one
            present = [r for r in (0, 1, 2) if ((rids_t == r) & attn.bool()).any()]
            if present:
                r = max(present, key=lambda rr: ((rids_t == rr) & attn.bool()).sum().item())
                sel = (rids_t == r) & attn.bool()
                w = sel.to(h.dtype).unsqueeze(-1)
                pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
                pooled = model.region_proj[str(r)](pooled)
                label_region = r
            else:
                pooled = h[:, 0]
                label_region = -1
            coords = model.factor_basis.project(pooled).squeeze(0)   # K, r_dim
            feats.append(coords.cpu().numpy())
            regions.append(label_region)
    return np.stack(feats), np.array(regions)


def probes(coords, labels, seed: int, n_perm: int = 200) -> Dict[str, float]:
    """Linear probe accuracy + clustering agreement + permutation p-value."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                                 balanced_accuracy_score, fowlkes_mallows_score)
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    x = coords.reshape(len(coords), -1)
    y = np.asarray(labels)
    uniq, counts = np.unique(y, return_counts=True)
    if uniq.size < 2:
        return {"status": "single_class", "n_classes": int(uniq.size)}

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed,
                                              stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = float(accuracy_score(y_te, pred))
    bal = float(balanced_accuracy_score(y_te, pred))

    km = KMeans(n_clusters=uniq.size, n_init=10, random_state=seed).fit_predict(x)
    ari = float(adjusted_rand_score(y, km))
    fmi = float(fowlkes_mallows_score(y, km))

    null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        xtr, xte, ytr, yte = train_test_split(x, y_perm, test_size=0.3,
                                              random_state=seed, stratify=y_perm)
        null.append(accuracy_score(yte, LogisticRegression(max_iter=1000).fit(xtr, ytr)
                                   .predict(xte)))
    null = np.asarray(null)
    p = float((null >= acc).mean() * (n_perm + 1) / n_perm)
    return {"status": "ok", "n": int(len(coords)), "n_classes": int(uniq.size),
            "majority": float(counts.max() / counts.sum()), "acc": acc,
            "balanced_acc": bal, "ari": ari, "fmi": fmi,
            "chance": float(null.mean()), "p_perm": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir holding rnajepa_modules.pt + config")
    ap.add_argument("--weights", required=True, help="pristine weights dir (tokenizer/config)")
    ap.add_argument("--data_root", default="/mnt/cunyuliu/rna-jepa/data/downstream_extracted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_task", type=int, default=400)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()

    _preselect_device(args.device)
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from rnajepa.metrics import benjamini_hochberg
    from rnajepa.model import JEPAConfig, RNARJEPA

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable; the probe must run on GPU")
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.weights, use_fast=True,
                                              trust_remote_code=True)
    specials = {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id,
                "pad": tokenizer.pad_token_id, "mask": tokenizer.mask_token_id,
                "unk": tokenizer.unk_token_id}
    cfg = JEPAConfig(model_path=args.weights, mask_token_id=specials["mask"])
    model = RNARJEPA(cfg).cuda()

    modules = os.path.join(args.ckpt, "rnajepa_modules.pt")
    if os.path.isfile(modules):
        state = torch.load(modules, map_location="cpu")
        model.predictor.load_state_dict(state["predictor"])
        model.factor_basis.load_state_dict(state["factor_basis"])
        model.region_proj.load_state_dict(state["region_proj"])
        model.teacher_region_proj.load_state_dict(state["teacher_region_proj"])
        with torch.no_grad():
            model.remask_embed.copy_(state["remask_embed"])
        print(f"loaded RNA-JEPA modules from {modules}")
    else:
        print("WARN: no rnajepa_modules.pt; factors will be at their initial values")

    ckpt_model = os.path.join(args.ckpt, "pytorch_model.bin")
    if os.path.isfile(ckpt_model):
        sd = torch.load(ckpt_model, map_location="cpu")
        missing, unexpected = model.student.load_state_dict(sd, strict=False)
        print(f"loaded encoder from checkpoint; missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    else:
        print("WARN: no encoder checkpoint next to the modules; using the original weights")

    results: Dict[str, dict] = {}
    pvals, keys = [], []
    k = None
    per_pool: Dict[str, tuple] = {}

    def run_pool(pool: Dict[str, Sequence[str]], pool_name: str, tag: str,
                 control: Dict[str, dict]) -> None:
        nonlocal k
        samples, labels = [], []
        for label, rels in pool.items():
            got = load_sequences(args.data_root, rels, args.n_per_task, args.seed)
            print(f"  pool {label}: {len(got)} sequences")
            samples.extend(got)
            labels.extend([label] * len(got))
        coords, _ = build_features(model, tokenizer, specials, samples, "cuda",
                                   args.max_len)
        k = coords.shape[1]
        y = np.array(labels)
        per_pool[tag] = (coords, y)
        print(f"  {pool_name}: coordinates {coords.shape}")
        for kk in range(k):
            key = f"{tag}_factor{kk}"
            results[key] = probes(coords[:, kk, :], y, args.seed, args.n_perm)
            if results[key].get("status") == "ok":
                pvals.append(results[key]["p_perm"])
                keys.append(key)
        key = f"{tag}_all"
        results[key] = probes(coords.reshape(len(coords), -1), y, args.seed, args.n_perm)
        if results[key].get("status") == "ok":
            pvals.append(results[key]["p_perm"])
            keys.append(key)

        # Specificity: does factor k beat the *union of the others* for this label?
        # If the factors are interchangeable partitions of one representation, this
        # difference is ~0, which is the honest outcome to report.
        for kk in range(k):
            others = [j for j in range(k) if j != kk]
            key_o = f"{tag}_factor{kk}_vs_others"
            a = probes(coords[:, kk, :], y, args.seed, args.n_perm).get("balanced_acc")
            b = probes(coords[:, others, :].reshape(len(coords), -1), y, args.seed,
                       args.n_perm).get("balanced_acc")
            if a is not None and b is not None:
                results[key_o] = {"status": "specificity", "factor": kk,
                                  "bal_acc_factor": a, "bal_acc_others": b,
                                  "delta": a - b}
                print(f"  {key_o}: factor {kk} {a:.3f} vs others {b:.3f} "
                      f"(delta {a - b:+.3f})")

        # ---- random-subspace control -------------------------------------
        # The K learned blocks tile the hidden space, so concatenating all K is just an
        # orthogonal rotation of the whole vector -- and a linear probe is
        # rotation-invariant, which makes that version of the control vacuous (an
        # earlier version of this file did exactly that and reported identical numbers
        # for learned and control, which is a property of the probe, not evidence about
        # the factors).  The meaningful control keeps the *rank* fixed: one random
        # r_dim-dimensional subspace, probed exactly like a single learned factor.
        with torch.no_grad():
            d = model.hidden_size
            r_dim = d // k
            orig = model.factor_basis.P.data.clone()
            for trial in range(2):
                torch.manual_seed(1000 + trial)
                q, _ = torch.linalg.qr(torch.randn(d, d, device="cuda"))
                rand_sub = q[:, :r_dim]                       # single random r_dim subspace
                model.factor_basis.P.data.copy_(
                    rand_sub.unsqueeze(0).expand(k, -1, -1).clone())
                c2, _ = build_features(model, tokenizer, specials, samples, "cuda",
                                       args.max_len)
                model.factor_basis.P.data.copy_(orig)
                control[f"{tag}_random_{r_dim}d_subspace_trial{trial}"] = probes(
                    c2[:, 0, :].reshape(len(c2), -1), y, args.seed, args.n_perm)

    control: Dict[str, dict] = {}
    print("pool A: region identity (sanity check, confounded by token length)")
    run_pool(REGION_TASKS, "region", "region", control)
    print("pool B: task identity within one region (the interpretable question)")
    run_pool(WITHIN_REGION_TASKS, "within-region task", "task", control)


    # ---- FDR across every probe reported ----------------------------------
    if pvals:
        adj = benjamini_hochberg(pvals, alpha=0.05)
        for key, pa, rej in zip(keys, adj["p_adj"], adj["reject"]):
            results[key]["p_adj_bh"] = pa
            results[key]["significant_bh"] = bool(rej)

    # ---- write ------------------------------------------------------------
    n_total = sum(int(c.shape[0]) for c, _ in per_pool.values())
    with open(os.path.join(args.out, "factor_probe.json"), "w", encoding="utf-8") as fh:
        json.dump({"ckpt": args.ckpt, "n_sequences": n_total,
                   "K": k, "r_dim": int(next(iter(per_pool.values()))[0].shape[2]) if per_pool else None,
                   "probes": results, "random_subspace_control": control,
                   "pool_sizes": {tag: int(c.shape[0]) for tag, (c, _) in per_pool.items()},
                   "protocol_note": (
                       "The region pool is a sanity check on the machinery and is "
                       "confounded by tokenisation (codons are three characters, UTR "
                       "tokens are one), so it must not be quoted as evidence for C3. "
                       "The within-region task pool is the interpretable question: all "
                       "four tasks are CDS with identical codon tokenisation. Controls "
                       "are random orthonormal subspaces of equal rank, because a linear "
                       "probe is rotation-invariant and rotating the learned basis would "
                       "prove nothing. If no probe survives BH-FDR, claim C3 is withdrawn "
                       "and Fig.5 becomes a training-dynamics panel.")},
                  fh, indent=1, default=str)

    for tag, (coords, y) in per_pool.items():
        path = os.path.join(args.out, f"factor_coords_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "label"] + [f"f{kk}_{j}" for kk in range(coords.shape[1])
                                             for j in range(coords.shape[2])])
            flat = coords.reshape(len(coords), -1)
            for i in range(len(coords)):
                w.writerow([i, y[i]] + [f"{v:.5f}" for v in flat[i]])

    lines = ["# Factor semantics probe", "",
             f"checkpoint: `{args.ckpt}`",
             f"sequences: {n_total} across {len(per_pool)} pool(s)",
             f"K = {k}, r_dim = {next(iter(per_pool.values()))[0].shape[2] if per_pool else '?'}",
             "",
             "| probe | n | classes | chance | balanced acc | ARI | FMI | p(perm) | p(BH) | significant |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    spec_lines = ["", "### Factor specificity (factor k vs the union of the others)", "",
                  "| pool | factor | bal_acc(factor) | bal_acc(others) | delta |",
                  "|---|---|---|---|---|"]
    for name, r in results.items():
        if r.get("status") == "specificity":
            spec_lines.append(f"| {name} | {r['factor']} | {r['bal_acc_factor']:.3f} | "
                              f"{r['bal_acc_others']:.3f} | {r['delta']:+.3f} |")
    for name, r in list(results.items()) + list(control.items()):
        if r.get("status") == "specificity":
            continue
        if r.get("status") != "ok":
            lines.append(f"| {name} | - | - | - | - | - | - | - | - | {r.get('status')} |")
            continue
        lines.append(
            f"| {name} | {r['n']} | {r['n_classes']} | {r['chance']:.3f} | "
            f"{r['balanced_acc']:.3f} | {r['ari']:.3f} | {r['fmi']:.3f} | "
            f"{r['p_perm']:.4f} | {r.get('p_adj_bh', float('nan')):.4f} | "
            f"{r.get('significant_bh', '')} |")
    lines += spec_lines + ["",
              "`region_*` is a sanity check confounded by tokenisation; `task_*` is the",
              "interpretable pool (four CDS tasks, identical codon tokenisation).",
              "Controls use random orthonormal subspaces of equal rank: a linear probe is",
              "rotation-invariant, so rotating the learned basis would prove nothing.",
              "",
              "This is an interpretability probe, not a performance result. If nothing",
              "survives FDR, claim C3 is withdrawn and Fig.5 becomes a dynamics panel."]
    with open(os.path.join(args.out, "probe_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    for name, r in results.items():
        if r.get("status") == "ok":
            print(f"  {name}: bal_acc={r['balanced_acc']:.3f} (chance {r['chance']:.3f}) "
                  f"ARI={r['ari']:.3f} p={r['p_perm']:.4f} p_bh={r.get('p_adj_bh')}")
    print(f"wrote {args.out}/factor_probe.json, factor_coords_*.csv, probe_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())