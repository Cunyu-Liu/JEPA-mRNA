"""Unified fine-tuning runner for the mRNABERT / RNA-JEPA head-to-head study.

Protocol fidelity
-----------------
This mirrors the official ``regression.py`` / ``classification.py`` closely
enough that the *only* intended difference from the baseline is the pre-trained
encoder:

* ``AutoModelForSequenceClassification`` with ``trust_remote_code=True``, so the
  released custom head (CLS -> ``BertPooler`` -> dropout -> linear) is used
  verbatim;
* ``num_labels=1`` for regression (MSE loss) and ``num_labels=n_train_classes``
  for classification (cross-entropy);
* the same ``SupervisedDataset``/collator semantics: CSV with a ``sequence``
  header, whitespace-separated tokens, right padding to the longest sequence in
  the batch, ``attention_mask = input_ids != pad_token_id``;
* ``load_best_model_at_end`` with dev-based selection, test evaluated once.

Reliability additions that do not change the science:

* **CUDA is mandatory.**  The device is selected before ``torch`` is imported
  and both the process and the model are asserted to be on GPU.  A silent CPU
  fallback would make every number meaningless, so it is a hard error.
* metrics come from :mod:`rnajepa.metrics`, which pins the positive-class-F1
  convention and always emits both F1 variants plus MCC;
* per-run artefacts: ``result.json``, ``predictions.tsv``, ``history.json``,
  ``run_meta.json`` (token-length stats, epochs actually run, best epoch,
  wall time, peak GPU memory).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rnajepa.metrics import classification_metrics, regression_metrics


# --------------------------------------------------------------------------- #
# Device selection has to happen before `import torch`.
# --------------------------------------------------------------------------- #
def _preselect_device(argv: Sequence[str]) -> str:
    device = "0"
    for i, tok in enumerate(argv):
        if tok == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
        elif tok.startswith("--device="):
            device = tok.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # keep tokenizers from spawning threads we do not need
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return str(device)


def _require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: CUDA is not available. This project forbids CPU training and "
            "CPU verification: a CPU run would silently produce meaningless numbers. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return torch


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_csv(path: str) -> List[tuple]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = [h.strip().lower() for h in rows[0]]
    if header[:2] != ["sequence", "label"]:
        raise ValueError(f"{path}: expected header 'sequence,label', got {rows[0][:3]}")
    out = []
    for row in rows[1:]:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue
        out.append((row[0], row[1]))
    return out


@dataclass
class TokenizedSplit:
    input_ids: "object"
    attention_mask: "object"
    labels: List[float]
    raw_sequences: List[str]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def tokenize_split(path: str, tokenizer, max_len: int, is_regression: bool) -> TokenizedSplit:
    """Tokenise one CSV split exactly the way the official dataset class does."""
    import torch

    rows = read_csv(path)
    texts = [r[0] for r in rows]
    labels = [float(r[1]) if is_regression else int(float(r[1])) for r in rows]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    )
    lengths = enc["attention_mask"].sum(dim=1)
    print(f"    {os.path.basename(path)}: n={len(texts)} "
          f"tokens mean={lengths.float().mean():.1f} max={int(lengths.max())} "
          f"truncated={(lengths == max_len).sum().item()}", flush=True)
    return TokenizedSplit(enc["input_ids"], enc["attention_mask"], labels, texts)


@dataclass
class Collator:
    """Right-pad a batch to its longest sequence, exactly like the official code.

    ``labels`` must be a tensor: the HF Trainer feeds collator output straight
    into ``prediction_step``, and a plain Python list would be treated as a
    non-tensor prediction.  Regression uses float labels (MSELoss); the
    classification head needs long labels (CrossEntropyLoss).
    """
    pad_token_id: int
    is_regression: bool

    def __call__(self, instances):
        import torch
        input_ids = [i["input_ids"] for i in instances]
        labels = [i["labels"] for i in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.tensor(
            labels, dtype=torch.float if self.is_regression else torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }


def make_compute_metrics(is_regression: bool, multiclass: bool):
    def compute(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        import numpy as np
        labels = np.asarray(labels)
        if is_regression:
            preds = np.asarray(logits).squeeze()
            return regression_metrics(labels, preds)
        preds = np.argmax(logits, axis=-1)
        if multiclass:
            return classification_metrics(labels, preds)
        # binary: probability of the positive class for ROC-AUC
        if logits.shape[-1] == 2:
            probs = _softmax(logits)[:, 1]
        else:
            probs = np.asarray(logits).squeeze()
        return classification_metrics(labels, preds, probs)
    return compute


def _softmax(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Fine-tune an mRNA encoder on one task split.")
    p.add_argument("--task", required=True)
    p.add_argument("--task_kind", required=True,
                   choices=["regression", "binary_classification", "multiclass_classification"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="freeze the pre-trained encoder and train only the pooler + "
                        "classifier head (linear-probe control; separates 'features are "
                        "uninformative' from 'fine-tuning destabilises them')")
    p.add_argument("--encoder_lr_scale", type=float, default=1.0,
                   help="multiply the encoder learning rate by this factor while the "
                        "randomly initialised pooler/classifier keep --lr. 1.0 reproduces "
                        "the official single-LR protocol; <1 is the discriminative-LR "
                        "remedy for tasks where full fine-tuning destroys the encoder.")
    p.add_argument("--run_name", default=None)
    args = p.parse_args(argv)
    args.run_name = args.run_name or f"{args.task}_s{args.seed}"
    return args


def _first_array(obj):
    """Return the first ndarray inside ``obj``, unwrapping HF's prediction tuples.

    ``Trainer.prediction_step`` builds its logits as *every* value of the model
    output dict except the ignored keys.  The released custom model returns a
    ``SequenceClassifierOutput`` whose ``hidden_states``/``attentions`` are
    ``None``, and its config does not list ``keys_to_ignore_at_inference``, so
    those ``None``s travel alongside the real logits and break ``np.asarray``.
    We both declare the ignore keys on the config (below) and unwrap defensively
    here, so a future config change cannot silently corrupt the metrics.
    """
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if item is not None:
                return _first_array(item)
        raise ValueError("prediction container holds no array")
    return np.asarray(obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    device = _preselect_device(argv)
    torch = _require_gpu()

    args = parse_args(argv)
    is_regression = args.task_kind == "regression"
    multiclass = args.task_kind == "multiclass_classification"

    import numpy as np
    import transformers
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from transformers.models.bert.configuration_bert import BertConfig

    transformers.logging.set_verbosity_warning()
    transformers.logging.disable_progress_bar()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[{args.task}] device=cuda:{device} seed={args.seed} "
          f"model={args.model_path}", flush=True)
    print(f"[{args.task}] gpu={torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.max_len, padding_side="right",
        use_fast=True, trust_remote_code=True,
    )

    train = tokenize_split(os.path.join(args.data_dir, "train.csv"), tokenizer,
                           args.max_len, is_regression)
    dev = tokenize_split(os.path.join(args.data_dir, "dev.csv"), tokenizer,
                         args.max_len, is_regression)
    test = tokenize_split(os.path.join(args.data_dir, "test.csv"), tokenizer,
                          args.max_len, is_regression)

    if is_regression:
        num_labels = 1
        problem_type = "regression"
    else:
        classes = sorted({int(l) for l in train.labels})
        num_labels = len(classes)
        if num_labels < 2:
            raise SystemExit(f"FATAL: train split of {args.task} has {num_labels} class(es)")
        problem_type = None  # let BertForSequenceClassification infer

    config = BertConfig.from_pretrained(args.model_path, num_labels=num_labels)
    if problem_type:
        config.problem_type = problem_type
    # The released config does not declare which output fields are not
    # predictions, so Trainer would treat hidden_states/attentions (both None
    # here) as extra logits.  Declaring them mirrors what stock HF configs do.
    config.keys_to_ignore_at_inference = ["past_key_values", "hidden_states", "attentions"]
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_path, config=config, trust_remote_code=True)

    # ``from_pretrained`` materialises on CPU; move it explicitly and *then*
    # assert, so a silent CPU fallback cannot slip through.
    model = model.cuda()
    if not next(model.parameters()).is_cuda:
        raise SystemExit("FATAL: model parameters are not on CUDA after moving. "
                         "Refusing to continue (silent CPU fallback).")

    if args.grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:  # pragma: no cover - depends on the custom model code
            print(f"[{args.task}] WARN gradient checkpointing requested but unsupported",
                  flush=True)

    if args.freeze_encoder:
        # freeze everything the released checkpoint provides; only the randomly
        # initialised pooler + classifier learn
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("bert.encoder") or name.startswith("bert.embeddings"):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{args.task}] freeze_encoder: {frozen/1e6:.1f}M frozen, "
              f"{trainable/1e6:.2f}M trainable (pooler + classifier)", flush=True)

    metric_for_best = "spearman" if is_regression else "accuracy"
    # transformers renamed ``evaluation_strategy`` -> ``eval_strategy`` in 4.41;
    # mRNABERT pins 4.32.0, so pick whichever the installed version accepts.
    import inspect
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in _ta_params else "eval_strategy"
    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "hf_ckpt"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        **{eval_key: "epoch"},
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        run_name=args.run_name,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        # Progress bars would rewrite the same line thousands of times in the
        # log, which hides real messages and makes stall detection unreliable.
        disable_tqdm=not sys.stdout.isatty(),
    )

    optimizers = None
    if args.encoder_lr_scale != 1.0:
        # Discriminative learning rates: the pre-trained encoder moves gently while the
        # randomly initialised pooler/classifier learn at full speed.  Built explicitly
        # because HF Trainer cannot express per-group multipliers.
        from transformers.optimization import get_cosine_schedule_with_warmup
        enc, head = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert.encoder") or name.startswith("bert.embeddings"):
                enc.append(param)
            else:
                head.append(param)
        groups = [
            {"params": enc, "lr": args.lr * args.encoder_lr_scale,
             "weight_decay": args.weight_decay, "name": "encoder"},
            {"params": head, "lr": args.lr,
             "weight_decay": args.weight_decay, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
        steps_per_epoch = max(1, math.ceil(len(train) / args.batch_size / args.grad_accum))
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
            num_training_steps=max(total_steps, args.warmup_steps + 1))
        optimizers = (optimizer, scheduler)
        print(f"[{args.task}] discriminative LR: encoder={args.lr * args.encoder_lr_scale:.2e} "
              f"({len(enc)} tensors) head={args.lr:.2e} ({len(head)} tensors), "
              f"{total_steps} steps", flush=True)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=Collator(pad_token_id=tokenizer.pad_token_id,
                               is_regression=is_regression),
        compute_metrics=make_compute_metrics(is_regression, multiclass),
        tokenizer=tokenizer,
        optimizers=optimizers,
    )

    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()

    # one-shot test evaluation on the dev-selected checkpoint
    test_out = trainer.predict(test)
    logits = _first_array(test_out.predictions)
    labels = np.asarray(test_out.label_ids)

    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    if is_regression:
        preds = logits[:, 0] if logits.shape[1] == 1 else np.asarray(logits).squeeze()
        probs = None
        final = regression_metrics(labels, preds)
    else:
        argmax = np.argmax(logits, axis=-1)
        preds = argmax
        if not multiclass and logits.shape[-1] == 2:
            probs = _softmax(logits)[:, 1]
        else:
            probs = None
        final = classification_metrics(labels, argmax, probs)

    final["task"] = args.task
    final["task_kind"] = args.task_kind
    final["seed"] = args.seed

    dev_best = None
    for entry in reversed(trainer.state.log_history):
        if f"eval_{metric_for_best}" in entry:
            dev_best = entry[f"eval_{metric_for_best}"]
            break

    pred_path = os.path.join(args.out_dir, "predictions.tsv")
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["index", "label", "pred", "prob"])
        for i in range(len(labels)):
            w.writerow([i, labels[i], preds[i], "" if probs is None else probs[i]])

    with open(os.path.join(args.out_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=1, allow_nan=False, sort_keys=True)
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=1, default=str)

    meta = {
        "task": args.task,
        "task_kind": args.task_kind,
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "epochs_ran": int(math.ceil(trainer.state.epoch or 0)),
        "best_dev_metric": metric_for_best,
        "best_dev_value": dev_best,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "test_examples": len(test),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "wall_time_s": time.time() - t_start,
        "peak_gpu_mem_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, allow_nan=False, sort_keys=True)

    if args.save_model:
        model.save_pretrained(os.path.join(args.out_dir, "best"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, "best"))
    trainer.save_state()

    primary = "spearman" if is_regression else "accuracy"
    print(f"[{args.task}] DONE test_{primary}={final[primary]:.4f} "
          f"wall={meta['wall_time_s']:.1f}s peak_gpu={meta['peak_gpu_mem_bytes']/2**30:.2f}GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())