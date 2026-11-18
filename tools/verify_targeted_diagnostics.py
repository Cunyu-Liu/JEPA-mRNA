#!/usr/bin/env python3
"""Independent consistency checks for the two targeted data diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask

EXPECTED_CEILINGS = {"chanrg_test": 0.8045510353421697, "chanrg_gena": 0.8529983421940205,
                     "chanrg_genc": 0.8167761413243034, "chanrg_genf": 0.7580659457542989}


def _close(a, b, tol=1e-10):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chanrg-dir", type=Path, required=True)
    parser.add_argument("--fn-dir", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise ValueError(f"{name}: {detail}")

    sequence = pd.read_parquet(args.chanrg_dir / "chanrg_per_sequence.parquet")
    family = pd.read_csv(args.chanrg_dir / "chanrg_per_family.csv")
    chanrg = json.loads((args.chanrg_dir / "chanrg_truncation_summary.json").read_text())
    numeric = ["n_raw_gt_pairs", "n_task_legal_pairs", "n_task_removed_pairs",
               "n_noncanonical_pairs", "n_canonical_nested_one_to_one_pairs"]
    for column in numeric:
        sequence[column] = pd.to_numeric(sequence[column], errors="coerce").fillna(0).astype(int)
        family[column] = pd.to_numeric(family[column], errors="coerce").fillna(0).astype(int)
    check("chanrg_sequence_count", len(sequence) == chanrg["n_sequence_rows"] == 21604, len(sequence))
    check("chanrg_family_count", len(family) == chanrg["n_family_rows"] == 933, len(family))
    for row in family.itertuples():
        selected = sequence[(sequence.dataset_id == row.dataset_id) &
                            (sequence.family == row.family) & (sequence.clan == row.clan)]
        for column in numeric:
            check(f"family_join_{row.dataset_id}_{row.family}_{column}",
                  int(selected[column].sum()) == int(getattr(row, column)),
                  f"{selected[column].sum()} != {getattr(row, column)}")
        check(f"family_removed_{row.dataset_id}_{row.family}",
              int(row.n_task_removed_pairs) == int(row.n_raw_gt_pairs - row.n_task_legal_pairs))
    for dataset_id, expected in EXPECTED_CEILINGS.items():
        summary = chanrg["datasets"][dataset_id]
        selected = sequence[sequence.dataset_id == dataset_id]
        raw, legal = int(selected.n_raw_gt_pairs.sum()), int(selected.n_task_legal_pairs.sum())
        check(f"summary_raw_{dataset_id}", raw == summary["n_raw_gt_pairs"])
        check(f"summary_legal_{dataset_id}", legal == summary["n_task_legal_pairs"])
        check(f"ceiling_{dataset_id}", _close(legal / raw, expected) and
              _close(summary["task_native_theoretical_max_recall_micro"], expected), legal / raw)
        check(f"scope_separation_{dataset_id}",
              summary["unified_canonical_nested_theoretical_max_recall"] == 1.0 and
              summary["n_task_removed_pairs"] == raw - legal)

    pairs = pd.read_parquet(args.fn_dir / "high_frequency_fn_per_pair.parquet")
    fn = json.loads((args.fn_dir / "high_frequency_fn_summary.json").read_text())
    buckets = pd.read_csv(args.fn_dir / "high_frequency_fn_buckets.csv")
    sources = json.loads((args.fn_dir / "tr1c_local_positive_source_counts.json").read_text())
    check("fn_row_count", len(pairs) == fn["n_pairs"])
    check("fn_frequency_gate", bool((pairs.train_count_local >= 100).all()))
    check("fn_status_domain", set(pairs.candidate_status) <= {"TP", "FN"} and
          set(pairs.baseline_map_status) <= {"TP", "FN"})
    expected_mechanism = np.where(pairs.candidate_status == "TP", "NOT_FN",
                           np.where(pairs.posterior >= 0.5, "HIGH_MARGINAL_DECODER_REMOVED",
                                    "LOW_MARGINAL_MODEL_SCORE"))
    check("fn_mechanism_partition", bool((pairs.fn_mechanism.to_numpy() == expected_mechanism).all()))
    for key, group in pairs.groupby("local_key"):
        expected = int(group.train_count_local.iloc[0])
        observed = int(sum(sources[key].values()))
        check(f"tr1c_source_count_{key}", observed == expected, f"{observed} != {expected}")
    for dataset_id in ["ALL", *sorted(pairs.dataset_id.unique())]:
        selected = pairs if dataset_id == "ALL" else pairs[pairs.dataset_id == dataset_id]
        summary = fn["summaries"][dataset_id]
        false_negative = selected[selected.candidate_status == "FN"]
        check(f"fn_total_{dataset_id}", len(selected) == summary["n_high_frequency_pairs"])
        check(f"fn_count_{dataset_id}", len(false_negative) == summary["n_candidate_fn"])
        check(f"fn_high_{dataset_id}", int((false_negative.posterior >= 0.5).sum()) ==
              summary["n_high_marginal_decoder_removed"])
        check(f"fn_low_{dataset_id}", int((false_negative.posterior < 0.5).sum()) ==
              summary["n_low_marginal_model_score"])
        persistent = (selected.candidate_status == "FN") & (selected.baseline_map_status == "FN")
        recovered = (selected.candidate_status == "TP") & (selected.baseline_map_status == "FN")
        lost = (selected.candidate_status == "FN") & (selected.baseline_map_status == "TP")
        shared_tp = (selected.candidate_status == "TP") & (selected.baseline_map_status == "TP")
        check(f"fn_persistent_{dataset_id}", int(persistent.sum()) == summary["n_persistent_fn_vs_turner_on_map"])
        check(f"fn_recovered_{dataset_id}", int(recovered.sum()) == summary["n_recovered_from_turner_on_map"])
        check(f"fn_lost_{dataset_id}", int(lost.sum()) == summary["n_lost_from_turner_on_map"])
        check(f"fn_shared_tp_{dataset_id}", int(shared_tp.sum()) == summary["n_shared_tp_vs_turner_on_map"])
        check(f"fn_state_partition_{dataset_id}", int(persistent.sum() + recovered.sum() + lost.sum() + shared_tp.sum()) == len(selected))
        check(f"fn_net_change_{dataset_id}", int(lost.sum() - recovered.sum()) == summary["candidate_fn_change_vs_turner_on_map"])
        check(f"fn_baseline_total_{dataset_id}", int((persistent | recovered).sum()) == summary["n_turner_on_map_fn"])
        check(f"fn_exact_coverage_{dataset_id}", _close(summary["exact_sequence_evaluable_fraction"],
              summary["n_exact_sequence_evaluable_pairs"] / len(selected)) if len(selected) else True)
        check(f"fn_conflict_{dataset_id}", int((selected.exact_sequence_label_status == "CONFLICT").sum()) ==
              summary["n_exact_sequence_label_conflicts"])
    for dimension in ("length_bin", "loop_bin", "helix_bin", "register_bin", "source", "label_source"):
        selected = buckets[buckets.dimension == dimension]
        check(f"bucket_total_{dimension}", int(selected.n_high_frequency_pairs.sum()) == len(pairs))
        check(f"bucket_fn_{dimension}", int(selected.n_candidate_fn.sum()) ==
              int((pairs.candidate_status == "FN").sum()))
    route = json.loads(args.route_summary.read_text())["rows"]
    route = {(r["dataset_id"], r["score_variant"]): r for r in route}
    decoded_sequences = 0
    pair_mismatches = []

    def f1_value(tp, fp, fn_count):
        return 2 * tp / (2 * tp + fp + fn_count) if 2 * tp + fp + fn_count else 1.0

    for dataset_id, expected_counts in fn["full_panel_candidate_pair_counts"].items():
        files = sorted((args.raw_root / dataset_id).glob("*.npz"))
        check(f"raw_file_count_{dataset_id}", len(files) == 64, len(files))
        candidate_counts, baseline_counts = Counter(), Counter()
        candidate_macro, baseline_macro = [], []
        dataset_pairs = pairs[pairs.dataset_id == dataset_id]
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                seq, name = str(z["seq"]), str(z["name"])
                seq_hash = str(z["sequence_sha256"])
                truth = {tuple(map(int, p)) for p in z["gt_pairs"]}
                mask = valid_pair_mask(seq)
                on = np.asarray(z["xens2_on"], dtype=np.float64)
                off = np.asarray(z["xens2_off"], dtype=np.float64)
                _, posterior = inside_outside(off / 2.0, mask)
                candidate = set(map(tuple, decode_mea(posterior, mask, 4.0)))
                baseline = set(map(tuple, nussinov_map(on, mask)))
                for label, prediction, counts, macro in (("candidate", candidate, candidate_counts, candidate_macro),
                                                         ("baseline", baseline, baseline_counts, baseline_macro)):
                    tp, fp, fn_count = len(prediction & truth), len(prediction - truth), len(truth - prediction)
                    counts.update({"tp": tp, "fp": fp, "fn": fn_count, "n_truth": len(truth)})
                    macro.append(f1_value(tp, fp, fn_count))
                selected_rows = dataset_pairs[(dataset_pairs.name == name) &
                                              (dataset_pairs.sequence_sha256 == seq_hash)]
                for row in selected_rows.itertuples():
                    pair = (int(row.pair_i), int(row.pair_j))
                    mismatch = ((pair in candidate) != (row.candidate_status == "TP") or
                                (pair in baseline) != (row.baseline_map_status == "TP") or
                                not _close(posterior[pair], row.posterior, 1e-9) or
                                not _close(off[pair], row.raw_score_off, 1e-9))
                    if mismatch and len(pair_mismatches) < 20:
                        pair_mismatches.append({"dataset_id": dataset_id, "name": name, "pair": pair})
                decoded_sequences += 1
        expected_baseline = fn["full_panel_turner_on_map_pair_counts"][dataset_id]
        check(f"independent_candidate_counts_{dataset_id}", dict(candidate_counts) == expected_counts,
              f"{dict(candidate_counts)} != {expected_counts}")
        check(f"independent_baseline_counts_{dataset_id}", dict(baseline_counts) == expected_baseline,
              f"{dict(baseline_counts)} != {expected_baseline}")
        for label, counts, macro, variant, decoder in (
                ("candidate", candidate_counts, candidate_macro, "xens2_off", "mea"),
                ("baseline", baseline_counts, baseline_macro, "xens2_on", "map")):
            tp, fp, fn_count = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn_count) if tp + fn_count else 1.0
            f1 = f1_value(tp, fp, fn_count)
            expected = route[(dataset_id, variant)]
            check(f"route_{label}_precision_{dataset_id}", _close(precision, expected[f"{decoder}_micro_precision"]))
            check(f"route_{label}_recall_{dataset_id}", _close(recall, expected[f"{decoder}_micro_recall"]))
            check(f"route_{label}_f1_{dataset_id}", _close(f1, expected[f"{decoder}_micro_f1"]))
            check(f"route_{label}_macro_f1_{dataset_id}", _close(float(np.mean(macro)), expected[f"{decoder}_macro_f1"]))
        expected = route[(dataset_id, "xens2_off")]
        check(f"route_config_{dataset_id}", expected["mea_temperature"] == 2.0 and expected["mea_gamma"] == 4.0)
        print(f"[verify-targeted] independently decoded {dataset_id}: 64", flush=True)
    check("independent_pair_values", not pair_mismatches, pair_mismatches)
    check("independent_sequence_count", decoded_sequences == 512, decoded_sequences)
    check("no_external_retuning", fn["old_external_panels_used_for_route_or_parameter_selection"] is False)
    report = {"status": "PASS", "n_checks": len(checks), "checks": checks,
              "chanrg_rows": len(sequence), "chanrg_family_rows": len(family),
              "high_frequency_pair_rows": len(pairs),
              "independently_decoded_sequences": decoded_sequences,
              "interpretation_limit": "diagnostic consistency only; not route selection, production repair, or SOTA evidence"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
