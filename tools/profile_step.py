"""Diagnostic: where does a real training step spend its time, and is the
hierarchical cascade head compatible with the Gibbs/Nussinov objective?

Run on the cluster:
    PYTHONPATH=/mnt/cunyuliu/pylibs:src TMPDIR=/mnt/cunyuliu/tmp \
        ~/miniconda3/envs/lucaone/bin/python /home/cunyuliu/rna-jepa/tools/profile_step.py

Answers three questions with measurements, not assumptions:

1. Per-segment wall time of one optimiser step at the *real* length distribution
   (encoder forward / head / inside-outside DP / backward / optimizer), plus the
   CPU-numpy share of the DP.
2. Whether ``FlatDecisionHead`` or ``HierarchicalCascade`` fits in a 5 GB MIG
   slice at the corpus' longest sequences (max L = 498), for several batch sizes.
3. Whether the cascade's ``-inf`` outside active blocks is compatible with
   ``negative_log_likelihood`` (it is *not* expected to be: the objective is a
   partition function over the whole non-crossing space, so masking most of the
   matrix changes what ``log Z`` means).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import torch

from rnajepa.decision_head import (
    DecisionModel,
    FlatDecisionHead,
    HierarchicalCascade,
    l0_helix_recall,
)
from rnajepa.encoder import build_encoder
from rnajepa.harness import negative_log_likelihood, inside_outside
from rnajepa.rlcd import ObjectiveWeights, combined_loss
from rnajepa.train_decision import (
    NEG_BIG,
    DecisionExample,
    TeacherLabelStore,
    collate,
    load_dataset_from_jsonl,
)

DATA = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl"
TEACHER = "/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr0"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_examples(path: str, teacher_dir: str, n: int) -> List[DecisionExample]:
    """``n`` examples spread evenly across the length range (not the shortest ``n``)."""
    store = TeacherLabelStore.from_dir(teacher_dir)
    ds = load_dataset_from_jsonl(path, store, need_teacher_probs=True,
                                 strict_teacher=True)
    ordered = sorted(ds.examples, key=lambda e: e.length)
    if n >= len(ordered):
        return list(ordered)
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[int(i)] for i in idx]


def profile_flat(examples: List[DecisionExample], device: str, steps: int) -> Dict[str, object]:
    """Time each segment of the flat-head training step at the real length mix."""
    encoder = build_encoder(size="35M")
    head = FlatDecisionHead(d_model=int(encoder.d_model))
    model = DecisionModel(encoder, head).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    weights = ObjectiveWeights()

    # length-bucketed batches, exactly as the driver builds them
    order = sorted(range(len(examples)), key=lambda i: examples[i].length)
    batches = [collate([examples[i] for i in order[k:k + 2]]) for k in range(0, len(order) - 1, 2)]

    seg: Dict[str, List[float]] = {k: [] for k in
                                   ("to_device", "forward", "dp_inside", "backward", "step")}
    peak = 0
    for batch in batches[:steps]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _sync()
        t1 = time.perf_counter()

        scores = model(batch["seq_ids"], lengths=batch["lengths"])
        _sync()
        t2 = time.perf_counter()

        matrix = scores.scores
        n = int(matrix.shape[0])
        total = None
        dp_time = 0.0
        for b in range(n):
            length = int(batch["lengths"][b])
            s = matrix[b, :length, :length]
            s = torch.where(torch.isfinite(s), s, torch.full_like(s, NEG_BIG))
            t3 = time.perf_counter()
            loss, _terms = combined_loss(
                s, batch["masks"][b], batch["gt_pairs"][b], weights,
                teacher_probs=batch["teacher_probs"][b],
                student_probs=torch.sigmoid(s),
                labels=torch.as_tensor(batch["labels"][b], dtype=torch.float64),
                return_terms=True,
            )
            _sync()
            dp_time += time.perf_counter() - t3
            total = loss if total is None else total + loss
        total = total / n
        _sync()
        t4 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        total.backward()
        _sync()
        t5 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        _sync()
        t6 = time.perf_counter()

        seg["to_device"].append(t1 - t0)
        seg["forward"].append(t2 - t1)
        seg["dp_inside"].append(dp_time)
        seg["backward"].append(t5 - t4)
        seg["step"].append(t6 - t0)
        peak = max(peak, torch.cuda.max_memory_allocated())

    out = {k: round(float(np.mean(v)), 4) for k, v in seg.items() if v}
    out["peak_mem_MiB"] = round(peak / 2 ** 20, 1)
    out["n_steps_timed"] = min(steps, len(batches))
    out["max_len_seen"] = max(int(b["lengths"].max()) for b in batches[:steps])
    return out


def mem_probe(device: str, lengths: List[int], batch_sizes: List[int]) -> List[Dict[str, object]]:
    """Peak memory + step time for flat vs cascade at long lengths and big batches."""
    encoder = build_encoder(size="35M")
    d = int(encoder.d_model)
    rows: List[Dict[str, object]] = []
    for kind in ("flat", "cascade"):
        for L in lengths:
            for B in batch_sizes:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    enc = build_encoder(size="35M").to(device)
                    head = (FlatDecisionHead(d_model=d) if kind == "flat"
                            else HierarchicalCascade(d_model=d, block_size=8, top_k=2)).to(device)
                    m = DecisionModel(enc, head).to(device)
                    m.train()
                    seq_ids = torch.randint(0, 4, (B, L), device=device)
                    t0 = time.perf_counter()
                    out = m(seq_ids, lengths=torch.full((B,), L, device=device))
                    loss = torch.where(torch.isfinite(out.scores), out.scores,
                                       torch.full_like(out.scores, NEG_BIG)).sum() * 1e-6
                    loss.backward()
                    _sync()
                    dt = time.perf_counter() - t0
                    rows.append({"head": kind, "L": L, "B": B, "ok": True,
                                 "peak_mem_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                                 "fwd_bwd_s": round(dt, 3)})
                    del m, head, enc, out, loss
                except torch.cuda.OutOfMemoryError:
                    rows.append({"head": kind, "L": L, "B": B, "ok": False,
                                 "peak_mem_MiB": None, "fwd_bwd_s": None})
                    torch.cuda.empty_cache()
    return rows


def cascade_objective_probe(device: str) -> Dict[str, object]:
    """Is the cascade's sparse score matrix usable as a Gibbs score matrix?

    The objective is ``log Z(x) - sum_{(i,j) in gt} s_ij`` where ``Z`` sums over
    *all* non-crossing structures.  If the cascade writes ``-inf`` (here replaced
    by NEG_BIG) outside the active blocks, ``Z`` is computed over a matrix whose
    off-support entries are enormously negative, so ``log Z`` no longer describes
    a distribution over the structures the model can actually emit.
    """
    L = 60
    seq_ids = torch.randint(0, 4, (1, L), device=device)
    enc = build_encoder(size="35M").to(device)
    cascade = HierarchicalCascade(d_model=int(enc.d_model), block_size=8, top_k=2).to(device)
    flat = FlatDecisionHead(d_model=int(enc.d_model)).to(device)
    with torch.no_grad():
        h = enc(seq_ids)
        c = cascade(h, seq_ids)
        f = flat(h, seq_ids)

    # a legal GT structure for this random sequence, via the flat head's own MAP
    from rnajepa.harness import nussinov_map, valid_pair_mask
    seq = "".join("ACGU"[int(i)] for i in seq_ids[0].tolist())
    mask = valid_pair_mask(seq)
    gt = nussinov_map(f.scores[0].detach().cpu().numpy(), mask)
    gt = [tuple(p) for p in gt]

    def nll_of(s: torch.Tensor) -> float:
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, NEG_BIG))
        return float(negative_log_likelihood(s, mask, gt))

    frac_illegal = float((~torch.isfinite(c.scores[0])).float().mean())
    return {
        "L": L,
        "n_gt_pairs": len(gt),
        "flat_nll": round(nll_of(f.scores[0]), 3),
        "cascade_nll": round(nll_of(c.scores[0]), 3),
        "cascade_frac_neg_inf": round(frac_illegal, 4),
        "cascade_flops": float(cascade.last_flops or 0.0),
        "flat_flops": float(flat.last_flops or 0.0),
        "verdict": ("cascade score matrix is NOT usable as a Gibbs score matrix: "
                    "the objective's log Z is taken over the full non-crossing space"
                    if nll_of(c.scores[0]) > 1e3 else
                    "cascade matrix numerically survivable in the objective"),
    }


def l0_recall_probe(examples: List[DecisionExample], device: str) -> Dict[str, object]:
    """Untrained L0 block-pair recall -- the cascade's unrecoverable false-negative gate."""
    enc = build_encoder(size="35M").to(device)
    cascade = HierarchicalCascade(d_model=int(enc.d_model), block_size=8, top_k=2).to(device)
    recs = []
    with torch.no_grad():
        for ex in examples[:40]:
            ids = ex.seq_ids[None, :].to(device)
            h = enc(ids)
            out = cascade(h, ids)
            recs.append(l0_helix_recall(out.block_active[0].cpu(), ex.gt_pairs, block_size=8))
    return {"n": len(recs), "mean_l0_helix_recall_untrained": round(float(np.mean(recs)), 4),
            "min": round(float(np.min(recs)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--skip-mem", action="store_true")
    args = ap.parse_args()

    device = args.device
    print(f"device={device} name={torch.cuda.get_device_name(0)} "
          f"free={torch.cuda.mem_get_info()[0] / 2**30:.1f}GiB", flush=True)

    examples = load_examples(DATA, TEACHER, args.n)
    print(f"loaded {len(examples)} examples, max_len={max(e.length for e in examples)}", flush=True)

    print("\n=== 1. flat-head step profile (real length mix) ===", flush=True)
    print(json.dumps(profile_flat(examples, device, args.steps), indent=1), flush=True)

    print("\n=== 2. cascade vs flat: objective compatibility ===", flush=True)
    print(json.dumps(cascade_objective_probe(device), indent=1), flush=True)

    print("\n=== 3. untrained L0 helix recall (P6 gate) ===", flush=True)
    print(json.dumps(l0_recall_probe(examples, device), indent=1), flush=True)

    if not args.skip_mem:
        print("\n=== 4. peak memory / time: flat vs cascade ===", flush=True)
        for row in mem_probe(device, lengths=[256, 498], batch_sizes=[2, 8, 16]):
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
