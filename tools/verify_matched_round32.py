#!/usr/bin/env python3
"""独立复核 matched_30k 已完成臂：micro、逐序列 macro/INF、来源与哈希三通道。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

W = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
ARMS = [f"{family}_30k_s{seed}" for family in ("chanrg_clean", "mixed") for seed in range(3)]
SPLITS = {"chanrg_test": 10597, "chanrg_gena": 5337, "chanrg_genc": 3936, "chanrg_genf": 1734}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_result(arm: str, split: str, expected_n: int, meta: dict) -> dict:
    path = W / "eval_decision" / arm / split / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device"] == "cuda", (path, data["device"])
    assert data["n_sequences"] == expected_n == len(data["per_sequence"]), path
    micro = data["pair_level"]["micro"]
    tp, fp, fn = int(micro["tp"]), int(micro["fp"]), int(micro["fn"])
    micro_recomputed = 2 * tp / (2 * tp + fp + fn)
    assert np.isclose(micro_recomputed, micro["f1"], atol=1e-10, rtol=0), path

    per_sequence = data["per_sequence"]
    f1_values, inf_values = [], []
    for index, row in enumerate(per_sequence):
        assert row["n_pred_pairs"] == len(row["pred_pairs"]), (path, index)
        p, r = float(row["precision"]), float(row["recall"])
        recomputed = 2 * p * r / (p + r) if p + r else 0.0
        assert np.isclose(recomputed, row["f1"], atol=1e-8, rtol=0), (path, index)
        assert bool(row["is_legal"]), (path, index)
        f1_values.append(float(row["f1"]))
        inf_values.append(float(row["inf"]))
    macro_f1 = float(np.mean(f1_values))
    macro_inf = float(np.mean(inf_values))
    macro = data["pair_level"]["macro"]
    assert np.isclose(macro_f1, macro["f1"], atol=1e-10, rtol=0), path
    assert np.isclose(macro_inf, macro["inf"], atol=1e-10, rtol=0), path
    assert data["legality"]["illegal_structure_rate"] == 0, path
    assert data["legality"]["hairpin_violation_rate"] == 0, path

    checkpoint = Path(data["checkpoint"])
    assert checkpoint.is_file(), checkpoint
    assert checkpoint.name == "best.pt", checkpoint
    best_reference = meta.get("validation", {}).get("best_checkpoint", "")
    if best_reference:
        assert os.path.realpath(best_reference) == os.path.realpath(checkpoint), (best_reference, checkpoint)
    corpus = Path(data["data"])
    assert corpus.is_file(), corpus
    with corpus.open(encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == expected_n, corpus
    return {
        "split": split,
        "n_sequences": expected_n,
        "micro_f1": float(micro["f1"]),
        "macro_f1": macro_f1,
        "macro_inf": macro_inf,
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "tp_fp_fn": [tp, fp, fn],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_result": str(path),
        "source_result_sha256": sha256(path),
        "source_data": str(corpus),
        "source_data_sha256": sha256(corpus),
        "checks": ["cuda_and_coverage", "micro_from_counts", "per_sequence_f1_macro_inf",
                   "legality", "best_checkpoint", "data_line_count_and_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", type=Path, default=W / "audits" / "round32_matched_verification.json")
    args = parser.parse_args()
    completed, pending, results = [], [], {}
    for arm in ARMS:
        meta_path = W / "runs" / arm / "run_meta.json"
        paths = [W / "eval_decision" / arm / split / "result.json" for split in SPLITS]
        if not meta_path.is_file() or not all(path.is_file() for path in paths):
            pending.append(arm)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["environment"]["resolved_device"].startswith("cuda"), arm
        assert meta["converged"] is True and meta["stop_reason"] == "early_stopping", arm
        assert meta["best_step"] is not None, arm
        arm_results = {}
        for split, expected_n in SPLITS.items():
            arm_results[split] = verify_result(arm, split, expected_n, meta)
        results[arm] = {
            "run_meta": str(meta_path),
            "run_meta_sha256": sha256(meta_path),
            "steps_completed": meta["steps_completed"],
            "best_step": meta["best_step"],
            "best_validation_loss": meta["best_validation_loss"],
            "results": arm_results,
            "ood_mean": float(np.mean([arm_results[split]["micro_f1"] for split in
                                       ("chanrg_gena", "chanrg_genc", "chanrg_genf")])),
        }
        completed.append(arm)
    if args.require_complete:
        assert not pending, f"incomplete arms: {pending}"
        assert len(completed) == 6
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "require_complete": args.require_complete,
        "completed_arms": completed,
        "pending_arms": pending,
        "n_verified_cells": sum(len(arm["results"]) for arm in results.values()),
        "arms": results,
        "verification_channels": [
            "aggregate micro from TP/FP/FN",
            "per-sequence F1, macro-F1 and macro-INF independent recomputation",
            "best checkpoint, corpus line counts and SHA256 provenance",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and checked["n_verified_cells"] == payload["n_verified_cells"]
    temporary.replace(args.out)
    print(f"MATCHED_VERIFICATION_PASS cells={payload['n_verified_cells']} completed={completed} pending={pending}")
    for arm, record in results.items():
        values = [record["results"][split]["micro_f1"] for split in SPLITS]
        print(arm, "Test/GenA/GenC/GenF", "/".join(f"{value:.6f}" for value in values),
              f"OOD_mean={record['ood_mean']:.6f}")
    print(f"REPORT {args.out} sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
