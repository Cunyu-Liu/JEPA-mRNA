"""Generate fine-tuning queue specs for a (model, task, seed) matrix.

Writes one JSON spec per cell into ``queue/pending/``; ``scripts/monitor.py`` then
dispatches them onto whichever GPUs have spare memory, one per free card, every 10
minutes.  Nothing here starts a job directly -- the queue is the single place that
decides what runs, so the auto-dispatcher and manual submission cannot fight.

Idempotent: a cell whose ``result.json`` already exists is skipped, so re-running after
adding seeds only enqueues the new work.  Spec filenames carry an explicit order prefix
so higher-value tasks (the paper's Fig.3 heart) are dispatched first.

Usage:
  python scripts/queue_matrix.py --model_label mrnabert_official \
      --model_path /mnt/cunyuliu/rna-jepa/weights/mRNABERT_attn_fallback \
      --seeds 42 --tasks cds_mrfp,rbp_0 --priority 1

  python scripts/queue_matrix.py --model_label v1_cont \
      --model_path /mnt/cunyuliu/rna-jepa/runs/v1_cont/hf/step_60000 \
      --seeds 42,43,44,45,46 --all-tasks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("FATAL: pyyaml required (pip install pyyaml)", file=sys.stderr)
    raise SystemExit(2)

ROOT = os.environ.get("RNAJEPA_ROOT", "/home/cunyuliu/rna-jepa")
ART = os.environ.get("RNAJEPA_ART", "/mnt/cunyuliu/rna-jepa")
DATA = os.path.join(ART, "data", "downstream_extracted")
QUEUE = os.path.join(ART, "queue", "pending")
EVAL_OUT = os.path.join(ART, "eval_out")
LEDGER = os.path.join(ART, "ledger.jsonl")
TASKS_YAML = os.path.join(ROOT, "configs", "tasks.yaml")

# Dispatch order: the main-result tasks first.  1 = paper heart (Fig.3), 2 = family
# coverage, 3 = extra depth.
PRIORITY = {
    "cds_mrfp": 1, "rbp_0": 1, "te_human": 1, "full_in_cell_half_life": 1,
    "utr5_U1": 2, "cds_stability": 2, "m6a_HEK293T_fold0": 2, "protein_solubility": 2,
    "cds_fungal": 2, "cds_cov": 3, "cds_ecoli": 3, "rbp_1": 3,
    "cds_riboswitch_1": 3,
}


def out_dir_for(model_label: str, task: str, seed: int) -> str:
    return os.path.join(EVAL_OUT, model_label, f"{task}_s{seed}")


def sweep_out_dir_for(model_label: str, task: str, seed: int, lr: float) -> str:
    return os.path.join(EVAL_OUT, f"{model_label}_lrsweep", f"{task}_lr{lr:g}_s{seed}")


def already_done(model_label: str, task: str, seed: int) -> bool:
    return os.path.isfile(os.path.join(out_dir_for(model_label, task, seed), "result.json"))


def running_or_queued(name: str) -> bool:
    """A cell is in flight if its spec is still pending or its run was dispatched."""
    candidates = [
        os.path.join(QUEUE, f"{name}.json"),
        os.path.join(ART, "queue", "running", f"{name}.jsonl"),
        os.path.join(ART, "queue", "done", f"{name}.json"),
    ]
    return any(os.path.isfile(c) for c in candidates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_label", required=True)
    ap.add_argument("--model_path", required=True,
                    help="weights dir (or a pre-training run's hf/step_N directory)")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds")
    ap.add_argument("--tasks", default="", help="comma-separated task keys")
    ap.add_argument("--all-tasks", action="store_true")
    ap.add_argument("--priority", type=int, default=0,
                    help="extra order prefix; lower runs first")
    ap.add_argument("--min_free", type=int, default=0,
                    help="override the per-task free-memory requirement")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--lrs", default="",
                    help="comma-separated learning rates. When given, one cell is enqueued "
                         "per (task, lr) under a separate <label>_lrsweep output root, for "
                         "calibrating the protocol on the reference model only. The chosen "
                         "value is then applied to every model, so the comparison stays fair.")
    ap.add_argument("--epochs_override", type=int, default=0,
                    help="shorter schedule for sweeps (the sweep only needs the dev curve)")
    ap.add_argument("--force", action="store_true",
                    help="re-enqueue cells that already have result.json. Used to repair "
                         "cells whose run_meta.json shows a protocol deviation (the OOM "
                         "retry can halve the batch) -- the new run overwrites the same "
                         "output directory, so there stays exactly one artefact per cell.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--python", default=os.environ.get(
        "RNAJEPA_PYTHON", "/home/cunyuliu/miniconda3/envs/lucaone/bin/python"),
        help="interpreter for the queued jobs; embedded in the spec because the "
             "dispatcher does not inherit this shell")
    args = ap.parse_args()

    with open(TASKS_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    tasks = cfg["tasks"]

    if args.all_tasks:
        selected = list(tasks)
    elif args.tasks:
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        print("FATAL: give --tasks or --all-tasks", file=sys.stderr)
        return 2
    unknown = [t for t in selected if t not in tasks]
    if unknown:
        print(f"FATAL: unknown task key(s): {unknown}", file=sys.stderr)
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(QUEUE, exist_ok=True)

    # Refuse to enqueue against a checkpoint that does not exist yet.  Without this a
    # matrix can be queued for a pre-training arm that is still running, the
    # dispatcher will happily launch every cell, and they will all fail on the missing
    # model path -- burning GPU slots and filling the ledger with noise.
    if not os.path.isdir(args.model_path):
        print(f"FATAL: model path does not exist yet: {args.model_path}")
        print("       queue this matrix once the run has written its checkpoint")
        return 3
    has_weights = os.path.isfile(os.path.join(args.model_path, "pytorch_model.bin"))
    if not has_weights:
        print(f"WARNING: {args.model_path} has no pytorch_model.bin; the loader will fall "
              f"back to config+tokenizer only and the run would be meaningless")

    # sort by priority then task name so the queue order is deterministic
    selected.sort(key=lambda t: (PRIORITY.get(t, 9), t))

    written, skipped_done, skipped_inflight, missing_data = 0, 0, 0, 0
    for task in selected:
        spec = tasks[task]
        src = os.path.join(DATA, spec["source"])
        if not os.path.isfile(os.path.join(src, "train.csv")):
            print(f"  SKIP {task}: no data at {src}")
            missing_data += 1
            continue
        lrs = [float(x) for x in args.lrs.split(",") if x.strip()] or [float(spec["lr"])]
        for seed in seeds:
          for lr in lrs:
            is_sweep = bool(args.lrs)
            name = (f"{args.model_label}_{task}_lr{lr:g}_s{seed}" if is_sweep
                    else f"{args.model_label}_{task}_s{seed}")
            if args.force:
                pass
            elif is_sweep:
                if os.path.isfile(os.path.join(sweep_out_dir_for(args.model_label, task, seed, lr),
                                               "result.json")):
                    skipped_done += 1
                    continue
            elif already_done(args.model_label, task, seed):
                skipped_done += 1
                continue
            if running_or_queued(name) and not args.dry_run:
                skipped_inflight += 1
                continue
            out_dir = (sweep_out_dir_for(args.model_label, task, seed, lr) if is_sweep
                       else out_dir_for(args.model_label, task, seed))
            epochs = args.epochs_override or spec["epochs"]
            # min_free follows the task's memory appetite. The 20 GB MIG slices are a
            # third of this node's idle capacity, so the threshold is set to admit them
            # where the task can actually fit rather than demanding a whole card.
            if args.min_free:
                min_free = args.min_free
            elif spec["max_len"] > 1024:
                min_free = 18000
            else:
                min_free = 10000
            cmd = (
                f"cd {ROOT} && RNAJEPA_PYTHON={args.python} "
                f"bash scripts/submit_finetune.sh"
                f" --task {task} --task_kind {spec['kind']}"
                f" --model_path {args.model_path}"
                f" --data_dir {src}"
                f" --out_dir {out_dir}"
                f" --seed {seed} --max_len {spec['max_len']} --batch {spec['batch']}"
                f" --epochs {epochs} --lr {lr}"
                f" --min_free {min_free}"
                + (" --fp16" if args.fp16 else "")
                + (" --save_model" if args.save_model else "")
                + f" --tag {args.model_label}"
            )
            order = args.priority * 100 + PRIORITY.get(task, 9)
            fname = f"{order:03d}_{name}.json"
            payload = {"cmd": cmd, "min_free_mib": min_free, "tag": args.model_label,
                       "task": task, "seed": seed, "model_label": args.model_label,
                       "timeout_s": 86400}
            if args.dry_run:
                print(f"[dry] {fname}\n      {cmd}")
            else:
                with open(os.path.join(QUEUE, fname), "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=1)
                written += 1

    print(f"enqueued {written} cell(s) into {QUEUE}")
    print(f"  skipped: {skipped_done} already have result.json, "
          f"{skipped_inflight} already pending/running, {missing_data} task(s) without data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())