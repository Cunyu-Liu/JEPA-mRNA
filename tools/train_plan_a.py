"""Plan-A: gradual unfreezing of the RiNALMo-giga backbone under the CRF objective.

Control arm: rinalmo_r2d_b4_s0 (frozen giga embeddings + resnet2d head,
TS0 micro F1 0.6629 / bpRNA-new 0.5010). The single variable under study is
backbone adaptation; everything else must match the control arm's protocol.

v2 discipline audit (2026-09-27). The v1 draft deviated from the control in five
ways that would have confounded the backbone-adaptation axis; v2 corrects all:

1. four-term objective: lambda_nll = lambda_distill = lambda_rlcd = lambda_cal
   = 1.0 with the thermodynamic teacher (thermo:viennarna, 42 shards, strict
   sequence keying). v1 ran nll only: ObjectiveWeights() defaults
   lambda_distill = 0, and the smoke test "passing" with teacher_probs=None was
   exactly the evidence of that gap.
2. batching: length-bucketed batch_size 4 with the exact iter_batches RNG
   (np.random.default_rng((seed, epoch)) permuting bucket order), so both arms
   see the same batch sequence. v1 used random pairing at batch 2: that changes
   the BatchNorm statistics the resnet2d scorer sees and the loss averaging.
3. encoder frame: <cls> + seq + <eos>, attention_mask, hidden[:, 1:L+1] --
   identical to the frozen-embedding extraction protocol
   (tools/extract_rinalmo_embeddings.py), so the in-loop forward reproduces the
   features the control arm consumed. v1 used a bare frame with no cls/eos and
   no attention_mask.
4. optimizer: AdamW betas=(0.9, 0.98) with the 5-step warmup LambdaLR of
   run_training, applied per layer-wise group.
5. NEG_BIG = -1e4 (the frozen sentinel), labels = the symmetric
   pair_indicator matrix, gt_pairs sorted, mask from harness.valid_pair_mask.

Plan-A's own variables (the thing under study):
- the 650M backbone is in the training loop with top-down gradual unfreezing
  (2 blocks / 800 steps after a 1600-step head warmup), layer-wise LR
  (backbone 1e-5, head 1e-4);
- the backbone stays dropout-free throughout (all nn.Dropout p set to 0 while
  the module tree is in train() mode): the frozen control's features are
  deterministic, so dropout here would be a regulariser the control never saw.
  train() mode is required for HF gradient checkpointing to engage (the gate
  is `self.training`), which is what fits the run on a 3g.20gb MIG slice.

Parity check: until the first unfreeze (step 1600) this arm should track the
control arm's loss trajectory (same data, same batch order, same head family);
that is the built-in health signal for the whole protocol alignment.

Memory: gradient checkpointing over the 33 encoder blocks (use_reentrant=False)
+ bf16 autocast on the encoder only (the head stays fp32 like the control);
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set by the launcher.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/cunyuliu/rna-jepa")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX  # noqa: E402
from rnajepa.decision_head import (  # noqa: E402
    FlatDecisionHead, pair_mask_from_ids, turner_phys_scores)
from rnajepa.rlcd import ObjectiveWeights, combined_loss  # noqa: E402
from rnajepa.harness import valid_pair_mask  # noqa: E402
from rnajepa.distill import load_teacher_shard, pair_indicator  # noqa: E402
from rinalmo_preflight import VOCAB, load_encoder  # noqa: E402
from torch.utils.checkpoint import checkpoint as _grad_ckpt  # noqa: E402

NEG_BIG = -1.0e4

TEACHER_DIR = "/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr0"
CKPT_DIR = "/mnt/cunyuliu/rna-jepa/ckpts"


def load_records(path):
    """(seq, sorted pairs) per line -- the control arm's load_dataset_from_jsonl."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = str(r["seq"]).upper().replace("T", "U")
            pairs = [tuple(p) for p in r["pairs"]] if "pairs" in r else []
            records.append((seq, sorted(pairs)))
    return records


def load_teacher_map(directory):
    """seq -> (L, L) float64 soft labels, the TeacherLabelStore.from_dir protocol."""
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    mapping = {}
    for shard in manifest.get("shards", []):
        seqs, probs = load_teacher_shard(os.path.join(directory, str(shard["file"])))
        for s, p in zip(seqs, probs):
            mapping[str(s)] = np.asarray(p, dtype=np.float64)
    return mapping


class BatchStream:
    """Length-bucketed endless batch stream, the iter_batches RNG exactly.

    ``sorted by length -> contiguous buckets of batch_size -> bucket order
    permuted per epoch by np.random.default_rng((seed, epoch))``.  Reproduced
    one-for-one so the control arm and this arm see the same batches in the
    same order.
    """

    def __init__(self, records, batch_size, seed):
        lengths = [len(s) for s, _ in records]
        order = sorted(range(len(records)), key=lambda i: lengths[i])
        self.buckets = [order[i:i + batch_size]
                        for i in range(0, len(order), batch_size)]
        self.seed = seed
        self.state = {"epoch": 0, "queue": []}

    def _permute(self):
        rng = np.random.default_rng((self.seed, self.state["epoch"]))
        return [self.buckets[int(b)] for b in rng.permutation(len(self.buckets))]

    def next(self):
        if not self.state["queue"]:
            # first permutation uses epoch 0, exactly as _batch_stream's first
            # iter_batches(dataset, ..., epoch=0) call does
            self.state["queue"] = self._permute()
            self.state["epoch"] += 1
        return self.state["queue"].pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr0.jsonl")
    ap.add_argument("--teacher-dir", default=TEACHER_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-head-steps", type=int, default=1600)
    ap.add_argument("--unfreeze-every", type=int, default=800)
    ap.add_argument("--unfreeze-per-step", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--snapshot-every", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--zchunk", type=int, default=64, help="column width of the "
                    "checkpointed z construction (numerically exact; see docstring)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    records = load_records(args.data)
    print(f"[plan-a] {len(records)} sequences from {args.data}", flush=True)
    teacher_map = load_teacher_map(args.teacher_dir)
    print(f"[plan-a] teacher: {len(teacher_map)} soft labels from {args.teacher_dir}",
          flush=True)
    missing = [s for s, _ in records if s not in teacher_map]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} sequences have no teacher label "
                         f"(first: len {len(missing[0])}); refusing to run a "
                         "distillation term over a partial teacher.")
    weights = ObjectiveWeights(lambda_nll=1.0, lambda_distill=1.0,
                               lambda_rlcd=1.0, lambda_cal=1.0)

    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    print(f"[plan-a] backbone: {report['n_params']/1e6:.0f}M params, "
          f"d={report['d_model']}, layers={report['n_layers']}", flush=True)
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    encoder = encoder.to(device)
    head = head.to(device)

    # dropout-free train() mode: matches the control's deterministic (cached)
    # features while keeping HF gradient checkpointing engaged.
    encoder.train()
    n_dropout = 0
    for m in encoder.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n_dropout += 1
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[plan-a] encoder in train() mode with {n_dropout} dropout modules "
          "set to p=0 (deterministic forward, checkpointing active)", flush=True)

    for p in encoder.parameters():
        p.requires_grad_(False)
    blocks = getattr(encoder, "encoder", None)
    if blocks is None:
        blocks = getattr(encoder, "transformer", None)
    if blocks is not None and hasattr(blocks, "layer"):
        blocks = blocks.layer
    if blocks is None or not hasattr(blocks, "__getitem__"):
        raise RuntimeError(f"cannot locate transformer blocks on {type(encoder)}")
    n_blocks = len(blocks)
    print(f"[plan-a] {n_blocks} transformer blocks found", flush=True)

    head_params = list(head.parameters())
    enc_params = list(encoder.parameters())
    n_head = sum(p.numel() for p in head_params)
    print(f"[plan-a] head params: {n_head/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW([
        {"params": head_params, "lr": args.head_lr},
        {"params": enc_params, "lr": args.backbone_lr},
    ], weight_decay=0.0, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / max(1, args.warmup_steps)))

    stream = BatchStream(records, args.batch_size, args.seed)
    n_unfrozen = 0
    start_step = 0

    meta = {
        "arm": os.path.basename(args.out.rstrip("/")),
        "control_arm": "rinalmo_r2d_b4_s0",
        "control_results": {"ts0_micro_f1": 0.6629, "bprna_new_micro_f1": 0.5010},
        "backbone": "rinalmo-giga (pretrained safetensors, in-loop)",
        "head": "FlatDecisionHead scorer=resnet2d d_z=128 (head params "
                f"{n_head})",
        "objective": {"lambda_nll": 1.0, "lambda_distill": 1.0,
                      "lambda_rlcd": 1.0, "lambda_cal": 1.0,
                      "teacher": "thermo:viennarna (strict seq keying)",
                      "distill_kind": "kl", "rlcd_reward": "brier", "beta": 1.0,
                      "n_bins": 10, "tau": 0.1, "nll_normalization": "sum"},
        "protocol_parity": [
            "length-bucketed batch=4, iter_batches RNG (seed,epoch)",
            "encoder frame <cls>+seq+<eos> + attention_mask, hidden[1:L+1]",
            "encoder dropout p=0 (deterministic, matches cached features)",
            "AdamW betas=(0.9,0.98) wd=0, 5-step warmup LambdaLR",
            "NEG_BIG=-1e4, symmetric labels, sorted gt_pairs, valid_pair_mask",
        ],
        "schedule": {"steps": args.steps, "batch_size": args.batch_size,
                     "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
                     "warmup_head_steps": args.warmup_head_steps,
                     "unfreeze_every": args.unfreeze_every,
                     "unfreeze_per_step": args.unfreeze_per_step,
                     "n_blocks": n_blocks},
        "seed": args.seed, "backbone_report": report,
    }
    meta["status"] = "running"
    with open(os.path.join(args.out, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    resume_path = os.path.join(args.out, "resume.pt")
    if os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        head.load_state_dict(state["head"])
        encoder.load_state_dict(state["encoder"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        n_unfrozen = int(state["n_unfrozen"])
        start_step = int(state["step"])
        stream.state = state["stream_state"]
        print(f"[plan-a] resumed from step {start_step} "
              f"({n_unfrozen}/{n_blocks} blocks unfrozen)", flush=True)

    log_path = os.path.join(args.out, "train_log.jsonl")
    t0 = time.time()
    trainable = lambda: [p for p in head_params + enc_params if p.requires_grad]

    for step in range(start_step + 1, args.steps + 1):
        if step > args.warmup_head_steps and n_unfrozen < n_blocks:
            due = ((step - args.warmup_head_steps) // args.unfreeze_every) \
                * args.unfreeze_per_step
            due = min(due, n_blocks)
            while n_unfrozen < due:
                blk = blocks[n_blocks - 1 - n_unfrozen]
                for p in blk.parameters():
                    p.requires_grad_(True)
                n_unfrozen += 1
            if step % args.unfreeze_every == 0 or n_unfrozen == n_blocks:
                print(f"[plan-a] step {step}: unfroze to {n_unfrozen}/{n_blocks} "
                      "blocks (top-down)", flush=True)

        bucket = stream.next()
        seqs = [records[i][0] for i in bucket]
        gts = [records[i][1] for i in bucket]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        B = len(seqs)

        enc_ids = torch.zeros(B, max_len + 2, dtype=torch.long)
        attn = torch.zeros(B, max_len + 2, dtype=torch.long)
        head_ids = torch.zeros(B, max_len, dtype=torch.long)
        masks, labels_all = [], []
        for bi, s in enumerate(seqs):
            ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in s] \
                + [VOCAB["<eos>"]]
            enc_ids[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[bi, :len(ids)] = 1
            head_ids[bi, :len(s)] = torch.tensor(
                [BASE_TO_INDEX.get(c, UNK_INDEX) for c in s], dtype=torch.long)
            masks.append(valid_pair_mask(s))
            labels_all.append(pair_indicator(len(s), gts[bi]))

        enc_ids = enc_ids.to(device)
        attn = attn.to(device)
        head_ids = head_ids.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(input_ids=enc_ids, attention_mask=attn)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h[:, 1:max_len + 1, :].float()
        # quantise to the control arm's cached-feature precision (extraction wrote
        # float16; the control's head consumed fp32-cast fp16 values). Doing the
        # same here keeps the frozen-phase forward numerically aligned with the
        # control arm's inputs, and bounds activation memory a little too.
        h = h.to(torch.float16).to(torch.float32)
        # zero the pad region exactly as the control arm's _stack_embeddings does:
        # the resnet2d BatchNorm statistics span the full (B, C, L, L) tensor, so a
        # pad region holding live encoder states would shift them.
        for bi, L in enumerate(lengths):
            if L < max_len:
                h[bi, L:, :] = 0.0
        lengths_t = torch.tensor(lengths, device=device)

        # ---- memory-engineered head forward (numerically exact) ----------------
        # The control arm's head path needs ~35GB dynamic at B=4, L=500 (the z
        # construction retains a (B, L, L, 3840) tensor = 15.7GB, the type head
        # another ~10GB). This run must fit a 3g.20gb MIG slice, so:
        # (a) z is built column-chunked with each chunk checkpointed:
        #     PairRepresentation.cross is Linear + elementwise ops only, so
        #     torch.utils.checkpoint recomputes it exactly; torch.cat of the
        #     recomputed chunks is bit-identical to the whole-matrix result.
        # (b) the type head is skipped: its output never enters the loss, it
        #     receives no gradient, AdamW updates nothing (grad=None ->
        #     skipped), and it has no BatchNorm running stats. The control arm
        #     therefore trained it into its exact init state, which the
        #     evaluation of this arm can reproduce by re-initialising it.
        # (c) the resnet2d scorer is checkpointed too, with a BatchNorm-snapshot
        #     restore: recompute re-runs every BN forward, which would update
        #     the running stats twice per step and silently diverge from the
        #     control arm. The fix: after backward, restore the stats captured
        #     right after the *first* forward (exactly one update, as the control
        #     arm applied). The gamma/beta gradients are unaffected -- they come
        #     from the batch-statistics normalisation, which is identical in
        #     both passes.
        bns = [m for m in head.resnet2d.modules()
               if isinstance(m, torch.nn.BatchNorm2d)]

        def _snap():
            return [(bn.running_mean.clone(), bn.running_var.clone(),
                     bn.num_batches_tracked.clone()) for bn in bns]

        def _restore(snap):
            with torch.no_grad():
                for bn, (rm, rv, nbt) in zip(bns, snap):
                    bn.running_mean.copy_(rm)
                    bn.running_var.copy_(rv)
                    bn.num_batches_tracked.copy_(nbt)

        def _z_chunk(hi, hj):
            return head.pair_repr.cross(hi, hj)

        chunks = []
        zc = args.zchunk if args.zchunk > 0 else max_len
        for j0 in range(0, max_len, zc):
            chunks.append(_grad_ckpt(_z_chunk, h, h[:, j0:j0 + zc],
                                     use_reentrant=False))
        z = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
        s_raw = _grad_ckpt(head.resnet2d, z, use_reentrant=False)
        bn_after_fwd = _snap()  # stats after exactly one update
        s = s_raw
        if head.use_turner_prior:
            s = s + head.prior_weight.to(s.dtype) * turner_phys_scores(
                head_ids).to(device)
        s = head.calibration(s, lengths_t)
        mask_full = pair_mask_from_ids(head_ids, head.min_loop).to(device)
        s = torch.where(mask_full, s, torch.full_like(s, float("-inf")))
        matrix = s

        total = None
        terms_agg = {}
        for b in range(B):
            L = lengths[b]
            s = matrix[b, :L, :L]
            s = torch.where(torch.isfinite(s), s, torch.full_like(s, NEG_BIG))
            mask_b = torch.as_tensor(masks[b], device=s.device)
            loss_b, terms_b = combined_loss(
                s, mask_b, gts[b], weights,
                teacher_probs=torch.as_tensor(teacher_map[seqs[b]],
                                              dtype=torch.float64, device=s.device),
                student_probs=torch.sigmoid(s),
                labels=torch.as_tensor(labels_all[b], dtype=torch.float64,
                                       device=s.device),
                distill_kind="kl", reward="brier", beta=1.0,
                n_bins=10, tau=0.1, nll_normalization="sum",
                return_terms=True)
            total = loss_b if total is None else total + loss_b
            for k, v in terms_b.items():
                terms_agg[k] = terms_agg.get(k, 0.0) + float(v.detach())
        total = total / B
        for k in terms_agg:
            terms_agg[k] /= B

        opt.zero_grad(set_to_none=True)
        total.backward()
        # restore the one-update BN statistics (see (c) above): the backward's
        # recompute pass re-ran the BN forwards and applied a second update.
        _restore(bn_after_fwd)
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            trainable(), args.grad_clip))
        opt.step()
        sched.step()

        if step % 25 == 0 or step == 1:
            rec = {"step": step, "loss": round(float(total), 4),
                   "terms": {k: round(v, 4) for k, v in terms_agg.items()},
                   "unfrozen": n_unfrozen, "gnorm": round(gnorm, 2),
                   "lr_head": sched.get_last_lr()[0],
                   "lr_backbone": sched.get_last_lr()[-1],
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                   "sec": round(time.time() - t0, 1)}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[plan-a] step {step}/{args.steps} loss={float(total):.4f} "
                  f"terms={ {k: round(v, 3) for k, v in terms_agg.items()} } "
                  f"unfrozen={n_unfrozen}/{n_blocks} gnorm={rec['gnorm']} "
                  f"peak={rec['peak_gb']}GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"step": step, "head": head.state_dict(),
                        "encoder": encoder.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "n_unfrozen": n_unfrozen,
                        "stream_state": stream.state}, resume_path)
        if step % args.snapshot_every == 0 or step == args.steps:
            torch.save({"step": step,
                        "model": {"encoder": encoder.state_dict(),
                                  "head": head.state_dict()},
                        "meta": meta},
                       os.path.join(CKPT_DIR,
                                    f"{os.path.basename(args.out.rstrip('/'))}_step{step}.pt"))

    meta["status"] = "completed"
    meta["steps_completed"] = int(args.steps)
    with open(os.path.join(args.out, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[plan-a] DONE", flush=True)


if __name__ == "__main__":
    main()
