#!/usr/bin/env python3
"""从原始 result.json 构建五类消融×八 split 矩阵，缺一格即失败。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

W = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
OLD = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_shutdown_20261025/results/eval_decision_ref")
NEW = W / "eval_decision" / "main_ablations"
SPLITS = [
    ("bprna_ts0", 1288), ("bprna_new", 5388), ("ref_pdb_ts1", 63),
    ("ref_pdb_ts2", 39), ("ref_pdb_ts3", 19), ("ref_pdb_ts_hard", 28),
    ("archiveii_embok_clean", 2544), ("testsetb", 428),
]
ROW_NAMES = ["DP off (greedy)", "Turner prior = 0", "2D scorer -> pair MLP",
             "No distillation", "No RLCD"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_for(row: str, split: str) -> tuple[Path, str | None]:
    if row == "control":
        return NEW / "dp_r2d" / split / "result.json", "exact"
    if row == "control_B2":
        return NEW / "control_b2" / split / "result.json", None
    if row == "DP off (greedy)":
        return NEW / "dp_r2d" / split / "result.json", "greedy"
    if row == "Turner prior = 0":
        if split in {"archiveii_embok_clean", "testsetb"}:
            return NEW / "turnerzero" / split / "result.json", None
        return OLD / f"turnerzero_r2dtr1c_s0_step20000_{split}" / "result.json", None
    if row == "2D scorer -> pair MLP":
        if split in {"archiveii_embok_clean", "testsetb"}:
            return NEW / "fftr1c" / split / "result.json", None
        return OLD / f"ow_rinalmo_fftr1c_b4_s0_step20000_{split}" / "result.json", None
    if row == "No distillation":
        if split in {"archiveii_embok_clean", "testsetb"}:
            return NEW / "nodistill" / split / "result.json", None
        return OLD / f"ow_rinalmo_r2dtr1c_nodistill_b4_s0_step20000_{split}" / "result.json", None
    if row == "No RLCD":
        return NEW / "norlcd_b2" / split / "result.json", None
    raise KeyError(row)


def extract(path: Path, decoder: str | None, expected_n: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("device") != "cuda":
        raise ValueError(f"non-CUDA result: {path}: {data.get('device')}")
    if data.get("n_sequences") != expected_n:
        raise ValueError(f"coverage mismatch: {path}: {data.get('n_sequences')} != {expected_n}")
    if decoder:
        metric = data["decoders"][decoder]
        per_seq = [row["decoders"][decoder]["f1"] for row in data["per_sequence"]]
    else:
        metric = data["pair_level"]["micro"]
        per_seq = [row["f1"] for row in data["per_sequence"]]
    tp, fp, fn = int(metric["tp"]), int(metric["fp"]), int(metric["fn"])
    recomputed = 2 * tp / (2 * tp + fp + fn)
    if not np.isclose(recomputed, metric["f1"], atol=1e-10, rtol=0):
        raise ValueError(f"micro mismatch: {path}: {recomputed} != {metric['f1']}")
    if len(per_seq) != expected_n:
        raise ValueError(f"per-sequence coverage mismatch: {path}")
    macro = float(np.mean(per_seq))
    if "macro_f1" in metric and not np.isclose(macro, metric["macro_f1"], atol=1e-10, rtol=0):
        raise ValueError(f"macro mismatch: {path}")
    if not decoder:
        macro_saved = data["pair_level"].get("macro", {}).get("f1")
        if macro_saved is not None and not np.isclose(macro, macro_saved, atol=1e-10, rtol=0):
            raise ValueError(f"saved macro mismatch: {path}: {macro} != {macro_saved}")
        legality = data.get("legality", {})
        if legality.get("illegal_structure_rate", 0) != 0 or legality.get("hairpin_violation_rate", 0) != 0:
            raise ValueError(f"legality failure: {path}")
    return {
        "micro_f1": float(metric["f1"]), "macro_f1": macro,
        "precision": float(metric["precision"]), "recall": float(metric["recall"]),
        "tp": tp, "fp": fp, "fn": fn, "n_sequences": expected_n,
        "source_result": str(path), "source_result_sha256": sha256(path),
        "checkpoint": data.get("checkpoint", ""),
        "checkpoint_sha256": data.get("checkpoint_sha256", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=W / "tables")
    args = parser.parse_args()
    rows = ["control", "control_B2", *ROW_NAMES]
    matrix = {row: {} for row in rows}
    for row in rows:
        for split, expected_n in SPLITS:
            path, decoder = source_for(row, split)
            matrix[row][split] = extract(path, decoder, expected_n)

    # DP 的 exact 分支同时是控制组，必须与旧控制板逐 split 一致（跨硬件容许 0.0002）。
    for split, expected_n in SPLITS:
        old_control_path = OLD / f"ow_rinalmo_r2dtr1c_b4_s0_step20000_{split}" / "result.json"
        old_control = extract(old_control_path, None, expected_n)
        delta = abs(old_control["micro_f1"] - matrix["control"][split]["micro_f1"])
        if delta > 0.0002:
            raise ValueError(f"DP exact/control mismatch on {split}: {delta}")

    payload = {
        "schema_version": 1,
        "coverage": "40/40 ablation cells + 8/8 B4 control + 8/8 B2 control",
        "splits": [{"name": split, "n_sequences": n} for split, n in SPLITS],
        "rows": matrix,
        "delta_from_control": {
            row: {split: matrix[row][split]["micro_f1"] - matrix["control_B2" if row == "No RLCD" else "control"][split]["micro_f1"]
                  for split, _ in SPLITS}
            for row in ROW_NAMES
        },
        "notes": [
            "DP row uses greedy decoding; exact DP from the same score matrices is the control.",
            "Turner, 2D-scorer and distillation rows combine the original six splits with newly measured ArchiveII-clean and TestSetB.",
            "No-RLCD uses a paired B=2 control and B=2 no-RLCD early-stopping protocol after B=4 OOM; only this row is compared with control_B2.",
            "Every cell passed CUDA, sample coverage, micro TP/FP/FN and per-sequence macro checks.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "main_ablation_5x8.json"
    csv_path = args.out_dir / "main_ablation_5x8.csv"
    md_path = args.out_dir / "main_ablation_5x8.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", *rows])
        for split, _ in SPLITS:
            writer.writerow([split] + [f"{matrix[row][split]['micro_f1']:.8f}" for row in rows])
    lines = [
        "# Five ablations × eight main-benchmark splits",
        "",
        "All values are strict pair-level micro F1. Every cell passed independent micro/macro and CUDA/coverage checks.",
        "",
        "| split | " + " | ".join(rows) + " |",
        "|---|" + "---:|" * len(rows),
    ]
    for split, _ in SPLITS:
        lines.append("| " + split + " | " + " | ".join(f"{matrix[row][split]['micro_f1']:.4f}" for row in rows) + " |")
    lines += ["", "## ΔF1 versus matched control", "", "| ablation | " + " | ".join(split for split, _ in SPLITS) + " |",
              "|---|" + "---:|" * len(SPLITS)]
    for row in ROW_NAMES:
        lines.append("| " + row + " | " + " | ".join(f"{payload['delta_from_control'][row][split]:+.4f}" for split, _ in SPLITS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checked = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(checked["rows"]) == 7
    assert all(len(values) == 8 for values in checked["rows"].values())
    print(f"MAIN_ABLATION_MATRIX_PASS rows=5 splits=8 controls_B4_B2=16 out={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
