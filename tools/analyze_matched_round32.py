#!/usr/bin/env python3
"""T-A127：matched_30k 六臂终态统计与预注册门判定。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
SPLITS = {"chanrg_test": 10597, "chanrg_gena": 5337, "chanrg_genc": 3936, "chanrg_genf": 1734}
OOD = ("chanrg_gena", "chanrg_genc", "chanrg_genf")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def result(arm: str, split: str) -> dict:
    path = W / "eval_decision" / arm / split / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device"] == "cuda" and data["n_sequences"] == len(data["per_sequence"])
    return data


def micro(data: dict) -> float:
    return float(data["pair_level"]["micro"]["f1"])


def zero_shot() -> dict:
    registry = json.loads(Path("/home/rna_jev/tables/result_registry.json").read_text(encoding="utf-8"))
    out = {}
    for row in registry["rows"]:
        if row["model"] == "zero_shot_xens2_tr1c":
            out[row["split"]] = float(row["metrics"]["micro_f1"])
    assert set(out) == set(SPLITS), out
    return out


def paired(seed: int, split: str) -> dict:
    clean = result(f"chanrg_clean_30k_s{seed}", split)
    mixed = result(f"mixed_30k_s{seed}", split)
    assert clean["n_sequences"] == mixed["n_sequences"] == SPLITS[split]
    def rows(data):
        answer = {}
        for index, row in enumerate(data["per_sequence"]):
            key = row.get("name") or str(index)
            assert key not in answer
            answer[key] = float(row["f1"])
        return answer
    a, b = rows(clean), rows(mixed)
    assert set(a) == set(b)
    delta = np.array([b[key] - a[key] for key in a])
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        statistic, p_value = 0.0, 1.0
    else:
        test = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "seed": seed,
        "clean_micro": micro(clean), "mixed_micro": micro(mixed),
        "delta_micro": micro(mixed) - micro(clean),
        "delta_macro": float(delta.mean()), "delta_median": float(np.median(delta)),
        "n_win_mixed": int((delta > 0).sum()), "n_tie": int((delta == 0).sum()), "n_loss_mixed": int((delta < 0).sum()),
        "wilcoxon_statistic": statistic, "wilcoxon_p_value": p_value,
    }


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        out[index] = min(1.0, running)
    return out.tolist()


def main() -> int:
    zero = zero_shot()
    arms = {}
    for family in ("chanrg_clean", "mixed"):
        for seed in range(3):
            arm = f"{family}_30k_s{seed}"
            meta = json.loads((W / "runs" / arm / "run_meta.json").read_text(encoding="utf-8"))
            assert meta["converged"] is True and meta["stop_reason"] == "early_stopping"
            values = {split: micro(result(arm, split)) for split in SPLITS}
            arms[arm] = {
                "best_step": meta["best_step"], "steps_completed": meta["steps_completed"],
                "metrics": values, "ood_mean": float(np.mean([values[split] for split in OOD])),
                "four_split_mean": float(np.mean(list(values.values()))),
            }
    pairs = {split: [paired(seed, split) for seed in range(3)] for split in SPLITS}
    for split, rows in pairs.items():
        adjusted = holm([row["wilcoxon_p_value"] for row in rows])
        for row, p in zip(rows, adjusted):
            row["holm_p_value_within_split"] = p
    zero_ood = float(np.mean([zero[split] for split in OOD]))
    summary = {}
    for arm, record in arms.items():
        family = arm.rsplit("_s", 1)[0]
        summary.setdefault(family, {})[arm] = record
    family_stats = {}
    for family, records in summary.items():
        family_stats[family] = {
            "micro_mean": {split: float(np.mean([record["metrics"][split] for record in records.values()])) for split in SPLITS},
            "micro_sample_std": {split: float(np.std([record["metrics"][split] for record in records.values()], ddof=1)) for split in SPLITS},
            "ood_mean_of_seeds": float(np.mean([record["ood_mean"] for record in records.values()])),
            "ood_sample_std_of_seed_means": float(np.std([record["ood_mean"] for record in records.values()], ddof=1)),
        }
    mixed_mean = family_stats["mixed_30k"]["micro_mean"]
    verdict = {
        "genf_mix_hypothesis_pass": bool(mixed_mean["chanrg_genf"] >= 0.3112),
        "genf_sota_pass": bool(mixed_mean["chanrg_genf"] >= 0.4035),
        "ood_mean_sota_pass": bool(family_stats["mixed_30k"]["ood_mean_of_seeds"] >= zero_ood),
        "ofoldna_ood_mean_reference": 0.4410,
        "ofoldna_ood_mean_pass": bool(family_stats["mixed_30k"]["ood_mean_of_seeds"] >= 0.4410),
        "zero_shot_ood_mean": zero_ood,
        "note": "Gates are evaluated on three-seed means; paired per-sequence Wilcoxon-Holm remains mandatory for interpretation.",
    }
    payload = {
        "schema_version": 1, "status": "PASS", "n_arms": 6, "n_cells": 24,
        "arms": arms, "family_statistics": family_stats,
        "paired_clean_vs_mixed": pairs, "zero_shot_xens2": zero,
        "verdicts": verdict,
        "statistical_note": "Two-sided per-sequence Wilcoxon signed-rank for each seed, Holm within each split; no family-cluster independence claim.",
    }
    out_json = W / "tables" / "matched_30k_statistics.json"
    out_md = W / "tables" / "matched_30k_statistics.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# matched_30k three-seed statistics", "", "| family | Test | GenA | GenC | GenF | OOD mean |", "|---|---:|---:|---:|---:|---:|"]
    for family, record in family_stats.items():
        lines.append("| " + family + " | " + " | ".join(f"{record['micro_mean'][split]:.4f}±{record['micro_sample_std'][split]:.4f}" for split in SPLITS) + f" | {record['ood_mean_of_seeds']:.4f} |")
    lines += ["", "## clean vs mixed paired tests", "", "| split | seed | Δmicro | Δmacro | Wilcoxon p | Holm within split |", "|---|---:|---:|---:|---:|---:|"]
    for split, rows in pairs.items():
        for row in rows:
            lines.append(f"| {split} | {row['seed']} | {row['delta_micro']:+.6f} | {row['delta_macro']:+.6f} | {row['wilcoxon_p_value']:.3g} | {row['holm_p_value_within_split']:.3g} |")
    lines += ["", "## Preregistered verdicts", "", json.dumps(verdict, ensure_ascii=False, indent=2)]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checked = json.loads(out_json.read_text(encoding="utf-8"))
    assert checked["n_arms"] == 6 and checked["n_cells"] == 24 and len(checked["paired_clean_vs_mixed"]) == 4
    print(f"MATCHED_STATS_PASS out={out_json} sha256={sha256(out_json)}")
    print(json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
