#!/usr/bin/env python3
"""Independent VL0 grid-selection and four-identity metric recomputation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea  # noqa: E402
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask  # noqa: E402


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def clean_pairs(raw_pairs):
    pairs = sorted({tuple(sorted((int(p[0]), int(p[1])))) for p in raw_pairs})
    occupied = set()
    for i, j in pairs:
        if i < 0 or i == j or i in occupied or j in occupied:
            raise ValueError(f"invalid or conflicting pair {(i, j)}")
        occupied.update((i, j))
    for index, (i, j) in enumerate(pairs):
        for k, l in pairs[index + 1:]:
            if k >= j:
                break
            if i < k < j < l:
                raise ValueError(f"crossing pairs {(i, j)} and {(k, l)}")
    return pairs


def stems(raw_pairs):
    pair_set = set(clean_pairs(raw_pairs))
    output = []
    for i, j in sorted(pair_set):
        if (i - 1, j + 1) in pair_set:
            continue
        stem = []
        a, b = i, j
        while (a, b) in pair_set:
            stem.append((a, b))
            a, b = a + 1, b - 1
        output.append(tuple(stem))
    if sum(len(stem) for stem in output) != len(pair_set):
        raise AssertionError("independent stem partition failed")
    return output


def counts(predicted, truth):
    predicted_set, truth_set = set(predicted), set(truth)
    return {"tp": len(predicted_set & truth_set),
            "fp": len(predicted_set - truth_set),
            "fn": len(truth_set - predicted_set)}


def close(a, b, tolerance=1e-10):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decode_dir = args.stage / "decode_vl0"
    summary = json.loads((decode_dir / "vl0_selected_summary.json").read_text())
    frozen = json.loads((decode_dir / "frozen_decode_config.json").read_text())
    config = frozen["config"]
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise ValueError(f"{name}: {detail}")

    expected_identity = {
        "C0": ("train_on", "plana_on", "MAP"),
        "C1": ("train_on", "plana_off", "MEA"),
        "C2": ("train_off", "plana_off", "MEA"),
        "C3": ("train_off", "plana_on", "MEA"),
    }
    for arm, (trajectory, score, decoder) in expected_identity.items():
        check(f"identity_{arm}", config[arm]["score_variant"] == f"{trajectory}/{score}" and
              config[arm]["decoder"] == decoder)
    check("no_external_selection", not summary["old_external_panels_used"] and
          not frozen["old_external_panels_may_select_parameters"])
    check("development_dataset", summary["development_dataset"] == "bprna_vl0")

    with (decode_dir / "vl0_grid_per_sequence.csv").open() as handle:
        grid_rows = list(csv.DictReader(handle))
    with (decode_dir / "vl0_grid_summary.csv").open() as handle:
        stored_grid = list(csv.DictReader(handle))
    stored_grid_map = {(row["score_variant"], row["decoder"], float(row["temperature"]),
                        float(row["gamma"]) if row["gamma"] else None): row for row in stored_grid}
    grouped = defaultdict(list)
    for row in grid_rows:
        if row["decoder"] == "MEA":
            grouped[(row["score_variant"], float(row["temperature"]), float(row["gamma"]))].append(row)
    variant_map = {"C1_off": "C1", "C2_off": "C2", "C3_on": "C3"}
    for variant, arm in variant_map.items():
        candidates = []
        for (group_variant, temperature, gamma), rows in grouped.items():
            if group_variant != variant:
                continue
            check(f"grid_size_{variant}_{temperature}_{gamma}", len(rows) == summary["n_sequences"], len(rows))
            micro = [sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")]
            per_sequence = [prf(int(row["tp"]), int(row["fp"]), int(row["fn"])) for row in rows]
            macro_f1 = float(np.mean([row["f1"] for row in per_sequence]))
            macro_precision = float(np.mean([row["precision"] for row in per_sequence]))
            macro_recall = float(np.mean([row["recall"] for row in per_sequence]))
            micro_metrics = prf(*micro)
            stored = stored_grid_map[(variant, "MEA", temperature, gamma)]
            for metric, value in (("precision", micro_metrics["precision"]),
                                  ("recall", micro_metrics["recall"]), ("f1", micro_metrics["f1"])):
                check(f"grid_micro_{metric}_{variant}_{temperature}_{gamma}",
                      close(value, stored[f"micro_{metric}"], 1e-12))
            for metric, value in (("precision", macro_precision), ("recall", macro_recall), ("f1", macro_f1)):
                check(f"grid_macro_{metric}_{variant}_{temperature}_{gamma}",
                      close(value, stored[f"macro_{metric}"], 1e-12))
            candidates.append((micro_metrics["f1"], macro_f1, micro_metrics["precision"], temperature, gamma))
        check(f"grid_candidate_count_{variant}", len(candidates) == 20, len(candidates))
        selected = min(candidates, key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))
        check(f"selected_temperature_{arm}", close(config[arm]["temperature"], selected[3], 0))
        check(f"selected_gamma_{arm}", close(config[arm]["gamma"], selected[4], 0))
        check(f"selected_micro_f1_{arm}", close(config[arm]["vl0_micro_f1"], selected[0], 1e-12))
        check(f"selected_macro_f1_{arm}", close(config[arm]["vl0_macro_f1"], selected[1], 1e-12))

    selected_rows = [json.loads(line) for line in (decode_dir / "vl0_selected_per_sequence.jsonl").read_text().splitlines() if line.strip()]
    selected_map = {(row["arm"], row["sequence_sha256"]): row for row in selected_rows}
    on_files = {path.stem: path for path in (args.stage / "export_train_on/bprna_vl0").glob("*.npz")}
    off_files = {path.stem: path for path in (args.stage / "export_train_off/bprna_vl0").glob("*.npz")}
    check("same_file_identities", set(on_files) == set(off_files))
    check("complete_sequence_set", len(on_files) == summary["n_sequences"], len(on_files))
    check("selected_row_count", len(selected_rows) == 4 * len(on_files), len(selected_rows))
    recomputed = {arm: [] for arm in expected_identity}
    for index, sequence_hash in enumerate(sorted(on_files), 1):
        with np.load(on_files[sequence_hash], allow_pickle=False) as on, np.load(off_files[sequence_hash], allow_pickle=False) as off:
            check(f"sequence_match_{sequence_hash}", str(on["seq"]) == str(off["seq"]) and
                  np.array_equal(on["gt_pairs"], off["gt_pairs"]))
            seq = str(on["seq"])
            truth = clean_pairs(on["gt_pairs"])
            mask = valid_pair_mask(seq)
            for arm, (trajectory, score, decoder) in expected_identity.items():
                source = on if trajectory == "train_on" else off
                scores = np.asarray(source[score], dtype=np.float64)
                if decoder == "MAP":
                    predicted = nussinov_map(scores, mask)
                else:
                    _, posterior = inside_outside(scores / config[arm]["temperature"], mask)
                    check(f"finite_posterior_{arm}_{sequence_hash}", bool(np.isfinite(posterior).all()))
                    predicted = decode_mea(posterior, mask, config[arm]["gamma"])
                predicted = clean_pairs(predicted)
                pair_counts = counts(predicted, truth)
                stem_counts = counts(stems(predicted), stems(truth))
                row = selected_map[(arm, sequence_hash)]
                check(f"pair_counts_{arm}_{sequence_hash}", row["pair"] == pair_counts,
                      f"{row['pair']} != {pair_counts}")
                check(f"stem_counts_{arm}_{sequence_hash}", row["stem"] == stem_counts,
                      f"{row['stem']} != {stem_counts}")
                check(f"row_identity_{arm}_{sequence_hash}", row["name"] == str(on["name"]) and
                      row["n_truth_pairs"] == len(truth) and row["n_predicted_pairs"] == len(predicted))
                recomputed[arm].append({"pair": pair_counts, "stem": stem_counts})
        if index % 20 == 0:
            print(f"[verify-prior-science] independently decoded {index}/{len(on_files)}", flush=True)

    for arm, rows in recomputed.items():
        for level in ("pair", "stem"):
            micro = [sum(row[level][key] for row in rows) for key in ("tp", "fp", "fn")]
            micro_metrics = prf(*micro)
            per_sequence = [prf(row[level]["tp"], row[level]["fp"], row[level]["fn"]) for row in rows]
            for metric in ("precision", "recall", "f1"):
                check(f"summary_micro_{level}_{metric}_{arm}",
                      close(micro_metrics[metric], summary["arms"][arm][f"micro_{level}_{metric}"]))
                macro = float(np.mean([row[metric] for row in per_sequence]))
                check(f"summary_macro_{level}_{metric}_{arm}",
                      close(macro, summary["arms"][arm][f"macro_{level}_{metric}"]))
    report = {
        "status": "PASS",
        "scope": "INDEPENDENT_VL0_GRID_AND_SELECTED_DECODE_RECOMPUTATION",
        "mode": summary["mode"], "seed": summary["seed"], "n_sequences": len(on_files),
        "n_checks": len(checks), "checks": checks,
        "shared_init_sha256": summary["shared_init_sha256"],
        "train_on_checkpoint_sha256": summary["train_on_checkpoint_sha256"],
        "train_off_checkpoint_sha256": summary["train_off_checkpoint_sha256"],
        "decode_config": config, "arms": summary["arms"],
        "old_external_panels_used": False, "route_winner_selected": False,
        "interpretation_limit": "development-only verification; no production repair or SOTA claim",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n_checks": len(checks),
                      "n_sequences": len(on_files), "mode": summary["mode"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
