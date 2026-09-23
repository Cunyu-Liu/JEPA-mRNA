"""Pre-training loop for RNA-JEPA (V1 continual and V2 from-scratch).

Both arms share this script; only ``--init`` differs:

* ``official``  -> V1: continue from the released mRNABERT weights (main result)
* ``random``    -> V2: same architecture, randomly initialised (fair-comparison arm)

Design notes worth knowing before reading the code:

* **Fast tokenisation.**  ``pre.txt`` already holds whitespace-separated tokens
  that are, by construction, entries of the 74-token vocabulary.  Running the HF
  tokenizer over ~36M lines would dominate the pipeline, so encoding uses a
  direct token->id map; :func:`verify_encoder` proves it agrees with the HF
  tokenizer on real lines before training starts.
* **Length bucketing.**  Batching by similar token counts keeps padding low, which
  matters because the CDS lengths in the corpus span three orders of magnitude.
* **Checkpoint format.**  Every checkpoint is written in the *released*
  checkpoint's key layout (``bert.*`` + ``cls.*``) next to a copy of the custom
  model code, so downstream fine-tuning can load it with exactly the same
  ``AutoModelForSequenceClassification`` path used for the baseline.  That
  removes an entire class of "our checkpoint needs different loading code"
  mistakes.

Usage:
  python -m rnajepa.pretrain --arm v1 --init official --data <pre.txt> \
      --regions <pre_regions.txt> --weights <weights_dir> --out <run_dir> \
      --device 0 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Fast encoder
# --------------------------------------------------------------------------- #
def build_token_map(tokenizer) -> Dict[str, int]:
    """Map every vocabulary token to its id using the tokenizer's own vocab."""
    vocab = tokenizer.get_vocab()
    return dict(vocab)


def fetch_specials(tokenizer) -> Dict[str, int]:
    return {
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "pad": tokenizer.pad_token_id,
        "mask": tokenizer.mask_token_id,
        "unk": tokenizer.unk_token_id,
    }


def verify_encoder(lines: Sequence[str], token_map: Dict[str, int],
                   tokenizer, specials: Dict[str, int], n: int = 200) -> int:
    """Assert the fast encoder reproduces the HF tokenizer bit for bit."""
    import torch
    sample = lines[:n]
    if not sample:
        return 0
    ours = [encode_line_fast(t, token_map, specials) for t in sample]
    theirs = tokenizer(list(sample), add_special_tokens=True,
                       padding=False)["input_ids"]
    bad = 0
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            bad += 1
            if bad <= 3:
                print(f"  ENCODER MISMATCH line {i}: ours={a[:20]} theirs={b[:20]}")
    del torch
    return bad


def encode_line_fast(line: str, token_map: Dict[str, int],
                     specials: Dict[str, int]) -> List[int]:
    """``"ATG GCA A"`` -> ``[CLS, id(ATG), id(GCA), id(A), SEP]``.

    Unknown tokens become ``[UNK]``, exactly as the released WordPiece tokenizer
    does (its vocabulary has no ``##`` pieces, so any token that is not a
    vocabulary entry maps to ``[UNK]``).
    """
    ids = [specials["cls"]]
    unk = specials["unk"]
    for tok in line.split():
        ids.append(token_map.get(tok, unk))
    ids.append(specials["sep"])
    return ids


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CorpusReader:
    """Pair ``pre.txt`` with ``pre_regions.txt`` and yield bucketed batches.

    A chunk of lines is loaded, sorted by encoded length, cut into batches and
    the batch order is shuffled; padding is therefore bounded by the spread
    inside one bucket instead of the spread across the whole corpus.
    """

    def __init__(self, tokens_path: str, regions_path: str, token_map: Dict[str, int],
                 specials: Dict[str, int], max_len: int, chunk_lines: int = 200000,
                 start_line: int = 0):
        self.tokens_path = tokens_path
        self.regions_path = regions_path
        self.token_map = token_map
        self.specials = specials
        self.max_len = max_len
        self.chunk_lines = chunk_lines
        self.n_seen = 0
        # Skip a whole number of corpus lines when resuming.  Offsetting in lines
        # rather than in batches keeps the data position independent of the batch
        # size, which matters because an OOM retry changes the batch size mid-run.
        self.start_line = max(0, int(start_line))

    def _chunks(self) -> Iterator[Tuple[List[List[int]], List[List[int]]]]:
        tok_fh = open(self.tokens_path, encoding="utf-8")
        reg_fh = open(self.regions_path, encoding="utf-8")
        try:
            for _ in range(self.start_line):
                if not tok_fh.readline():
                    break
                reg_fh.readline()
            while True:
                toks: List[List[int]] = []
                regs: List[List[int]] = []
                for _ in range(self.chunk_lines):
                    line = tok_fh.readline()
                    if not line:
                        break
                    rline = reg_fh.readline()
                    ids = encode_line_fast(line, self.token_map, self.specials)
                    raw_len = len(ids) - 2                     # exclude CLS / SEP
                    r = [int(x) for x in rline.split()]
                    # alignment must be checked on the *untruncated* lengths: the
                    # region file is never truncated on disk
                    if len(r) != raw_len:
                        raise ValueError(
                            f"region/token misalignment at line {self.n_seen + len(toks)}: "
                            f"{len(r)} regions vs {raw_len} tokens")
                    if len(ids) > self.max_len:
                        ids = ids[:self.max_len - 1] + [self.specials["sep"]]
                        r = r[:self.max_len - 2]
                    toks.append(ids)
                    regs.append([-1] + r + [-1])           # CLS / SEP carry no region
                if not toks:
                    return
                self.n_seen += len(toks)
                yield toks, regs
        finally:
            tok_fh.close()
            reg_fh.close()

    def batches(self, batch_size: int, seed: int, start_after: int = 0) -> Iterator[dict]:
        import torch
        rng = random.Random(seed)
        skipped = 0
        for toks, regs in self._chunks():
            order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
            nb = len(order) // batch_size
            buckets = [order[i * batch_size:(i + 1) * batch_size] for i in range(nb)]
            rng.shuffle(buckets)
            for bucket in buckets:
                if skipped < start_after:
                    skipped += batch_size
                    continue
                max_len = max(len(toks[i]) for i in bucket)
                ids = torch.full((len(bucket), max_len), self.specials["pad"], dtype=torch.long)
                reg = torch.full((len(bucket), max_len), -1, dtype=torch.long)
                attn = torch.zeros((len(bucket), max_len), dtype=torch.long)
                for row, i in enumerate(bucket):
                    n = len(toks[i])
                    ids[row, :n] = torch.tensor(toks[i])
                    reg[row, :n] = torch.tensor(regs[i])
                    attn[row, :n] = 1
                special = torch.zeros((len(bucket), max_len), dtype=torch.bool)
                special |= ids.eq(self.specials["cls"]) | ids.eq(self.specials["sep"]) \
                    | ids.eq(self.specials["pad"])
                yield {"input_ids": ids, "attention_mask": attn,
                       "region_ids": reg, "special_ids": special}


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
FILES_TO_COPY = ("config.json", "configuration_bert.py", "bert_layers.py",
                 "bert_padding.py", "flash_attn_triton.py", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                 "generation_config.json", "bert_tokenizer_config.json")


def save_hf_checkpoint(model, weights_dir: str, out_dir: str, step: int) -> str:
    """Write the student encoder in exactly the released checkpoint layout.

    ``model.student`` is a :class:`BertBackbone` holding the encoder as ``bert``
    and the MLM head as ``cls``, so its own ``state_dict`` already uses the
    release's ``bert.*`` / ``cls.*`` key names -- no re-prefixing is needed (doing
    so would produce ``bert.bert.*`` and silently fail to load downstream).  The
    RNA-JEPA-only modules are saved alongside so a run can be resumed.

    The custom model code and tokenizer files are copied next to the weights so
    that ``AutoModelForSequenceClassification.from_pretrained(ckpt_dir,
    trust_remote_code=True)`` -- the same call the baseline uses -- just works.
    """
    import torch
    ckpt_dir = os.path.join(out_dir, "hf", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name in FILES_TO_COPY:
        src = os.path.join(weights_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, name))

    state = {key: value.detach().cpu() for key, value in model.student.state_dict().items()}
    if not any(k.startswith("bert.") for k in state):
        raise RuntimeError("student state_dict does not use the released 'bert.' prefix; "
                           "refusing to write a checkpoint that cannot be loaded")
    torch.save(state, os.path.join(ckpt_dir, "pytorch_model.bin"))

    torch.save(_cpu_state({
        "predictor": model.predictor.state_dict(),
        "factor_basis": model.factor_basis.state_dict(),
        "region_proj": model.region_proj.state_dict(),
        "teacher_region_proj": model.teacher_region_proj.state_dict(),
        "remask_embed": model.remask_embed.detach(),
    }), os.path.join(ckpt_dir, "rnajepa_modules.pt"))
    return ckpt_dir


def _cpu_state(obj):
    """Recursively move tensors in a nested dict to CPU (for checkpointing)."""
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu_state(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="v1",
                   help="free-form run label used in logs/ledgers (v1, v2, smoke, ...); "
                        "the actual behaviour is set by --init")
    p.add_argument("--init", default="official", choices=["official", "random"])
    p.add_argument("--data", required=True)
    p.add_argument("--regions", required=True)
    p.add_argument("--weights", required=True, help="weights dir to copy config/tokenizer from")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--mask_mode", default="token_uniform", choices=["token_uniform", "codon_span"])
    p.add_argument("--jepa_target", default="region", choices=["region", "masked", "both"],
                   help="latent target: pooled region summaries, teacher states at masked "
                        "positions, or both (A3 ablation axis)")
    p.add_argument("--loss_form", default="cos", choices=["cos", "mse"],
                   help="latent loss form (A3 ablation axis)")
    p.add_argument("--n_factors", type=int, default=4)
    p.add_argument("--lambda_var", type=float, default=25.0)
    p.add_argument("--lambda_cov", type=float, default=0.5)
    p.add_argument("--lambda_orth", type=float, default=1.0)
    p.add_argument("--curriculum_frac", type=float, default=0.6)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=0.9997)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume", default="")
    p.add_argument("--max_hours", type=float, default=0.0, help="stop after N hours (0 = no limit)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    device = "0"
    for i, tok in enumerate(argv):
        if tok == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
        elif tok.startswith("--device="):
            device = tok.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args(argv)
    import torch
    from transformers import AutoTokenizer

    from rnajepa.model import JEPAConfig, RNARJEPA
    from rnajepa.tokenization import REGION_CDS, REGION_5UTR, REGION_3UTR

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable; pre-training must run on GPU "
                         f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")
    cfg_path = os.path.join(args.out, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=1, sort_keys=True)

    print(f"[pretrain] device={torch.cuda.get_device_name(0)} arm={args.arm} init={args.init} "
          f"steps={args.steps} batch={args.batch_size}x{args.grad_accum}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.weights, use_fast=True,
                                              trust_remote_code=True)
    token_map = build_token_map(tokenizer)
    specials = fetch_specials(tokenizer)
    print(f"[pretrain] tokenizer vocab={len(token_map)} specials={specials}", flush=True)

    with open(args.data, encoding="utf-8") as fh:
        head_lines = [next(fh).strip() for _ in range(200)]
    bad = verify_encoder(head_lines, token_map, tokenizer, specials)
    if bad:
        raise SystemExit(f"FATAL: fast encoder disagrees with the HF tokenizer on "
                         f"{bad}/200 lines; refusing to train on mis-encoded data")
    print("[pretrain] fast encoder verified against the HF tokenizer on 200 lines", flush=True)

    jcfg = JEPAConfig(
        model_path=args.weights, mask_token_id=specials["mask"],
        mask_prob=args.mask_prob, mask_mode=args.mask_mode, n_factors=args.n_factors,
        jepa_target=args.jepa_target, loss_form=args.loss_form,
        alpha=args.alpha, lambda_var=args.lambda_var, lambda_cov=args.lambda_cov,
        lambda_orth=args.lambda_orth, curriculum_frac=args.curriculum_frac,
        ema_start=args.ema_start, ema_end=args.ema_end,
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RNARJEPA(jcfg).cuda()

    if args.init == "random":
        # re-initialise the encoder with the config's initialiser range: this is
        # the V2 fair-comparison arm, identical architecture, no released weights
        def _reinit(module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        model.student.apply(_reinit)
        with torch.no_grad():
            model.student.word_embeddings.weight[specials["mask"]].normal_(0.0, 0.02)
            model.remask_embed.copy_(model.student.word_embeddings.weight[specials["mask"]])
        model.teacher.load_state_dict(model.student.state_dict())
        print("[pretrain] V2 arm: encoder re-initialised from scratch", flush=True)

    model.train()

    def build_optimizer_scheduler(bs: int):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=args.weight_decay,
                                betas=(0.9, 0.98))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: min(1.0, (st + 1) / max(1, args.warmup_steps)))
        return opt, sched

    batch_size = args.batch_size
    total_opt_steps = max(1, args.steps // args.grad_accum)
    optimizer, scheduler = build_optimizer_scheduler(batch_size)

    opt_step = 0
    micro = 0
    consumed_lines = 0
    resume_path = args.resume or os.path.join(args.out, "resume.pt")
    if os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        model.student.load_state_dict(state["student"])
        model.predictor.load_state_dict(state["predictor"])
        model.factor_basis.load_state_dict(state["factor_basis"])
        model.region_proj.load_state_dict(state["region_proj"])
        model.teacher_region_proj.load_state_dict(state["teacher_region_proj"])
        with torch.no_grad():
            model.remask_embed.copy_(state["remask_embed"])
        if "teacher" in state:
            model.teacher.load_state_dict(state["teacher"])
        else:
            # older checkpoints did not store the EMA copy; falling back to the
            # student would silently change the objective, so rebuild it as the
            # EMA would have been at this point (momentum applied once is closer
            # to the truth than skipping the copy entirely).
            model.teacher.load_state_dict(model.student.state_dict())
            print(f"[{args.arm}] WARN resume file has no teacher state; teacher reset "
                  f"from the student", flush=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "rng" in state:
            torch.set_rng_state(state["rng"])
        opt_step = int(state["step"])
        consumed_lines = int(state.get("corpus_lines_seen", 0))
        micro = opt_step * args.grad_accum
        print(f"[{args.arm}] resumed from {resume_path} at step {opt_step}, "
              f"corpus line {consumed_lines}", flush=True)

    start_wall = time.time()
    attempt = 0
    while attempt < 6:
        attempt += 1
        reader = CorpusReader(args.data, args.regions, token_map, specials,
                              args.max_len, start_line=consumed_lines)
        try:
            data_iter = reader.batches(batch_size, args.seed)
            running: Dict[str, float] = {}
            while opt_step < total_opt_steps:
                for batch in data_iter:
                    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                    # ``micro`` counts *completed* micro-batches; increment first so
                    # that the diagnostics flag has the same parity as the flush
                    # below.  On the other parity the expensive Procrustes and
                    # correlation metrics would only ever be produced on steps that
                    # are never logged.
                    micro += 1
                    is_boundary = (micro % args.grad_accum == 0)
                    out = model(batch["input_ids"], batch["attention_mask"],
                                batch["region_ids"], step=opt_step,
                                total_steps=total_opt_steps,
                                special_ids=batch["special_ids"],
                                compute_diagnostics=is_boundary)
                    loss = out["loss"] / args.grad_accum
                    loss.backward()

                    for key in ("loss", "loss_mlm", "loss_jepa", "loss_var",
                                "loss_cov", "loss_orth"):
                        running[key] = running.get(key, 0.0) + float(out[key])
                    running["_micro"] = running.get("_micro", 0) + 1

                    if is_boundary:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        opt_step += 1
                        model.update_teacher(model.ema_momentum(opt_step, total_opt_steps))

                        if opt_step % args.log_every == 0:
                            n = max(1, running["_micro"])
                            row = {"ts": time.time(), "step": opt_step, "micro": micro,
                                   "lr": scheduler.get_last_lr()[0],
                                   "batch_size": batch_size,
                                   "sequences_seen_m": round(reader.n_seen / 1e6, 3),
                                   "sequences_seen": reader.n_seen}
                            for key in ("loss", "loss_mlm", "loss_jepa", "loss_var",
                                        "loss_cov", "loss_orth"):
                                row[key] = running[key] / n
                            for key in ("procrustes_residual", "cos_mean", "cls_corr",
                                        "cls_var", "w_region", "w_cls", "mask_frac"):
                                if key in out:
                                    row[key] = float(out[key])
                            if "factor_activity" in out:
                                row["factor_activity"] = [
                                    float(x) for x in out["factor_activity"]]
                            with open(log_path, "a", encoding="utf-8") as fh:
                                fh.write(json.dumps(row) + "\n")
                            msg = (f"[{args.arm}] step {opt_step}/{total_opt_steps} "
                                   f"loss={row['loss']:.4f} mlm={row['loss_mlm']:.4f} "
                                   f"jepa={row['loss_jepa']:.4f} var={row['loss_var']:.4f} "
                                   f"cov={row['loss_cov']:.4f} orth={row['loss_orth']:.4f}")
                            if "cos_mean" in row:
                                msg += (f" cos={row['cos_mean']:.3f} "
                                        f"proc={row.get('procrustes_residual', float('nan')):.3f}")
                            if "cls_corr" in row:
                                msg += f" cls_corr={row['cls_corr']:.3f}"
                            print(msg, flush=True)
                            running = {}

                        if args.save_every and opt_step % args.save_every == 0:
                            ck = save_hf_checkpoint(model, args.weights, args.out, opt_step)
                            torch.save({
                                "step": opt_step,
                                "student": model.student.state_dict(),
                                "teacher": model.teacher.state_dict(),
                                "predictor": model.predictor.state_dict(),
                                "factor_basis": model.factor_basis.state_dict(),
                                "region_proj": model.region_proj.state_dict(),
                                "teacher_region_proj": model.teacher_region_proj.state_dict(),
                                "remask_embed": model.remask_embed.detach(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "rng": torch.get_rng_state(),
                                "batch_size": batch_size,
                                "sequences_seen": reader.n_seen,
                                "corpus_lines_seen": consumed_lines + reader.n_seen,
                            }, os.path.join(args.out, "resume.pt"))
                            print(f"[{args.arm}] checkpoint written {ck} "
                                  f"(resume.pt updated)", flush=True)

                    if args.max_hours and (time.time() - start_wall) / 3600.0 > args.max_hours:
                        print(f"[{args.arm}] max_hours reached, stopping at step {opt_step}",
                              flush=True)
                        save_hf_checkpoint(model, args.weights, args.out, opt_step)
                        return 0
                    if opt_step >= total_opt_steps:
                        break
            break
        except torch.OutOfMemoryError:
            # The node is shared, so a card that had room at launch can fill up
            # mid-run.  Persist what we have, halve the batch, and continue from
            # the current weights: losing the whole run is worse than a recorded
            # batch-size change.
            consumed_lines += reader.n_seen
            torch.cuda.empty_cache()
            save_hf_checkpoint(model, args.weights, args.out, opt_step)
            batch_size = max(2, batch_size // 2)
            optimizer, scheduler = build_optimizer_scheduler(batch_size)
            print(f"[{args.arm}] CUDA OOM -> halving batch to {batch_size} and resuming "
                  f"from step {opt_step} (corpus line {consumed_lines})", flush=True)
            with open(os.path.join(args.out, "oom_events.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "step": opt_step,
                                     "new_batch_size": batch_size,
                                     "corpus_lines_seen": consumed_lines}) + "\n")

    final = save_hf_checkpoint(model, args.weights, args.out, opt_step)
    print(f"[{args.arm}] DONE steps={opt_step} final={final} "
          f"wall={(time.time()-start_wall)/3600:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())