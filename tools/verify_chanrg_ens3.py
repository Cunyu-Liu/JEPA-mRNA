#!/usr/bin/env python3
"""Independently verify CHANRG single-seed and ensemble result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SPLITS = ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")
ARMS = {
    "s0": "chanrg_arm_gt",
    "s1": "chanrg_arm_gt_s1",
    "s2": "chanrg_arm_gt_s2",
    "ens2": "chanrg_ens_gt",
    "ens3": "chanrg_ens3_gt",
    "board_r2d": None,
    "board_xens2": None,
}
CKPTS = {
    "s0": "chanrg_r2d_s0_gt_step20000.pt",
    "s1": "chanrg_r2d_s1_gt_step20000.pt",
    "s2": "chanrg_r2d_s2_gt_step20000.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def verify_result(path: Path, *, legacy: bool = False) -> tuple[dict, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    source_rows = data["per_sequence"]
    device = data.get("device", "not_recorded" if legacy else None)
    if not legacy and device != "cuda":
        errors.append(f"device={device!r}, expected 'cuda'")
    if data["n_sequences"] != len(source_rows):
        errors.append(f"n_sequences={data['n_sequences']} but per_sequence={len(source_rows)}")

    tp_total = fp_total = fn_total = 0
    rows = []
    for index, row in enumerate(source_rows):
        n_pred = row.get("n_pred_pairs", row.get("n_pred"))
        n_gt = row.get("n_gt_pairs", row.get("n_gt"))
        if "recall" in row:
            tp = int(round(row["recall"] * n_gt))
        else:
            tp = int(round(row["f1"] * (n_pred + n_gt) / 2))
        fp = n_pred - tp
        fn = n_gt - tp
        if min(tp, fp, fn) < 0:
            errors.append(f"negative pair count at row {index}")
        tp_total += tp
        fp_total += fp
        fn_total += fn
        # Both evaluators define an exactly correct empty/empty structure as F1=1.
        expected_f1 = 1.0 if n_pred + n_gt == 0 else 2 * tp / (n_pred + n_gt)
        if not close(row["f1"], expected_f1):
            errors.append(f"per-sequence F1 mismatch at row {index}")
        if row.get("is_legal") is False:
            errors.append(f"illegal structure at row {index}")
        rows.append({"f1": row["f1"], "inf": row.get("inf"),
                     "length": row.get("length"), "n_gt_pairs": n_gt})

    micro = data["pair_level"]["micro"]
    for key, actual in (("tp", tp_total), ("fp", fp_total), ("fn", fn_total)):
        if micro[key] != actual:
            errors.append(f"micro {key}: stored={micro[key]} recomputed={actual}")
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    for key, actual in (("precision", precision), ("recall", recall), ("f1", micro_f1)):
        if not close(micro[key], actual):
            errors.append(f"micro {key}: stored={micro[key]} recomputed={actual}")

    macro_f1 = statistics.fmean(row["f1"] for row in rows)
    macro_inf = (statistics.fmean(row["inf"] for row in rows)
                 if rows and all(row["inf"] is not None for row in rows) else None)
    macro = data["pair_level"]["macro"]
    if not close(macro["f1"], macro_f1):
        errors.append(f"macro F1: stored={macro['f1']} recomputed={macro_f1}")
    if macro_inf is not None and not close(macro["inf"], macro_inf):
        errors.append(f"macro INF: stored={macro['inf']} recomputed={macro_inf}")
    if "n_gt_pairs_total" in data and data["n_gt_pairs_total"] != sum(
            row["n_gt_pairs"] for row in rows):
        errors.append("n_gt_pairs_total mismatch")
    legality = data.get("legality", {})
    illegal_rate = legality.get("illegal_structure_rate")
    hairpin_rate = legality.get("hairpin_violation_rate")
    if illegal_rate not in (None, 0) or hairpin_rate not in (None, 0):
        errors.append("non-zero illegality or hairpin violation rate")

    metrics = {
        "path": str(path),
        "tag": data["tag"],
        "device": device,
        "legacy_schema": legacy,
        "n_sequences": len(rows),
        "n_skipped": data.get("n_skipped"),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "macro_inf": macro_inf,
        "precision": precision,
        "recall": recall,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "illegal_structure_rate": illegal_rate,
        "hairpin_violation_rate": hairpin_rate,
    }
    return {"metrics": metrics, "rows": rows}, errors


def compare(reference: dict, candidate: dict) -> dict:
    ref_rows = reference["rows"]
    cand_rows = candidate["rows"]
    if len(ref_rows) != len(cand_rows):
        raise ValueError("cannot compare different sequence counts")
    for index, (left, right) in enumerate(zip(ref_rows, cand_rows)):
        same_length = (left["length"] is None or right["length"] is None
                       or left["length"] == right["length"])
        if not same_length or left["n_gt_pairs"] != right["n_gt_pairs"]:
            raise ValueError(f"sequence alignment mismatch at row {index}")
    deltas = [right["f1"] - left["f1"] for left, right in zip(ref_rows, cand_rows)]
    return {
        "delta_micro_f1": candidate["metrics"]["micro_f1"] - reference["metrics"]["micro_f1"],
        "delta_macro_f1": candidate["metrics"]["macro_f1"] - reference["metrics"]["macro_f1"],
        "mean_per_sequence_delta_f1": statistics.fmean(deltas),
        "median_per_sequence_delta_f1": statistics.median(deltas),
        "n_win": sum(value > 0 for value in deltas),
        "n_tie": sum(value == 0 for value in deltas),
        "n_loss": sum(value < 0 for value in deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path,
                        default=Path("/root/rna_jepa_data/eval_decision"))
    parser.add_argument("--s2-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/"
                                     "rna_ssp_20261024/eval_decision/chanrg_arm_gt_s2"))
    parser.add_argument("--ckpt-root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/"
                                     "rna_ssp_20261024/weights"))
    parser.add_argument("--require-s2", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    required_arms = (["s0", "s1", "ens2", "ens3", "board_r2d", "board_xens2"]
                     + (["s2"] if args.require_s2 else []))
    results = {}
    errors = []
    for arm, directory in ARMS.items():
        results[arm] = {}
        for split in SPLITS:
            if arm == "s2":
                path = args.s2_root / split / "result.json"
            elif arm.startswith("board_"):
                path = args.eval_root / split / arm.removeprefix("board_") / "result.json"
            else:
                path = args.eval_root / directory / split / "result.json"
            if not path.is_file():
                message = f"missing {arm}/{split}: {path}"
                if arm in required_arms:
                    errors.append(message)
                else:
                    print(f"SKIP {message}")
                continue
            result, result_errors = verify_result(path, legacy=arm.startswith("board_"))
            results[arm][split] = result
            errors.extend(f"{arm}/{split}: {message}" for message in result_errors)

    comparisons = {}
    for split in SPLITS:
        if "ens2" in results and split in results["ens2"] and split in results["ens3"]:
            comparisons[f"ens3_minus_ens2/{split}"] = compare(
                results["ens2"][split], results["ens3"][split])
        for arm in ("s0", "s1", "s2", "board_r2d", "board_xens2"):
            if split in results.get(arm, {}) and split in results["ens3"]:
                comparisons[f"ens3_minus_{arm}/{split}"] = compare(
                    results[arm][split], results["ens3"][split])

    checkpoints = {}
    for arm, value in CKPTS.items():
        path = args.ckpt_root / value
        if path.is_file():
            checkpoints[arm] = {"path": str(path), "sha256": sha256(path)}
        else:
            errors.append(f"missing checkpoint for {arm}: {path}")

    report = {
        "schema_version": 1,
        "required_arms": required_arms,
        "splits": list(SPLITS),
        "results": {arm: {split: value["metrics"] for split, value in by_split.items()}
                    for arm, by_split in results.items()},
        "comparisons": comparisons,
        "checkpoint_sha256": checkpoints,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    for arm in ARMS:
        for split in SPLITS:
            if split in results.get(arm, {}):
                metric = results[arm][split]["metrics"]
                print(f"{arm:5s} {split:12s} n={metric['n_sequences']:5d} "
                      f"micro={metric['micro_f1']:.9f} macro={metric['macro_f1']:.9f}")
    for name, comparison in comparisons.items():
        if name.startswith(("ens3_minus_ens2", "ens3_minus_board")):
            print(f"{name}: d_micro={comparison['delta_micro_f1']:+.9f} "
                  f"d_macro={comparison['delta_macro_f1']:+.9f}")
    print(f"VERIFICATION {report['status']}: {args.out}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
