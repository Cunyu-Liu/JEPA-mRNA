#!/usr/bin/env python3
"""Descriptive VL0-only unfreezing trajectory; never changes the terminal matrix verdict."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask
from rnajepa.prior_alignment import sha256_file
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf

ARMS = ("C0", "C1", "C2", "C3")
STEPS = (3200, 9600, 16800, 20000)
LEVELS = ("pair", "stem")
METRICS = tuple(f"{scope}_{level}_{metric}" for level in LEVELS for scope in ("micro", "macro")
                for metric in ("precision", "recall", "f1"))


def length_bucket(length):
    if length <= 64:
        return "001-064"
    if length <= 128:
        return "065-128"
    if length <= 256:
        return "129-256"
    if length <= 512:
        return "257-512"
    return "513-1024"


def summarize_rows(rows):
    output = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        row = {"arm": arm, "n_sequences": len(selected)}
        for level in LEVELS:
            micro = prf(sum(x[level]["tp"] for x in selected),
                        sum(x[level]["fp"] for x in selected),
                        sum(x[level]["fn"] for x in selected))
            macro = [prf(x[level]["tp"], x[level]["fp"], x[level]["fn"]) for x in selected]
            for metric in ("precision", "recall", "f1"):
                row[f"micro_{level}_{metric}"] = micro[metric]
                row[f"macro_{level}_{metric}"] = float(np.mean([x[metric] for x in macro]))
        output[arm] = row
    return output


def _mean(values):
    return float(statistics.fmean(values))


def _deltas(means, left="C2", right="C1"):
    return {metric: means[left][metric] - means[right][metric] for metric in METRICS}


def aggregate_trajectory(records):
    """Aggregate verified records with {seed, step, summary, rows} keys."""
    by_step = {}
    for step in STEPS:
        selected = [record for record in records if record["step"] == step]
        if len(selected) != 3 or {record["seed"] for record in selected} != {0, 1, 2}:
            raise ValueError(f"step {step} does not have exactly seeds 0/1/2")
        means = {arm: {metric: _mean([record["summary"]["arms"][arm][metric] for record in selected])
                       for metric in METRICS} for arm in ARMS}
        rows = [row for record in selected for row in record["rows"]]
        per_seed = {str(record["seed"]): {
            "C2_minus_C1": _deltas(record["summary"]["arms"]),
            "C2_minus_C0": _deltas(record["summary"]["arms"], "C2", "C0"),
            "C2_minus_C3": _deltas(record["summary"]["arms"], "C2", "C3"),
        } for record in selected}
        stem_deltas = []
        per_sequence = {}
        for record in selected:
            arms = defaultdict(dict)
            for row in record["rows"]:
                arms[row["sequence_sha256"]][row["arm"]] = row
            for sequence_hash, variants in arms.items():
                c1, c2 = variants["C1"], variants["C2"]
                c1_f1 = prf(c1["stem"]["tp"], c1["stem"]["fp"], c1["stem"]["fn"])["f1"]
                c2_f1 = prf(c2["stem"]["tp"], c2["stem"]["fp"], c2["stem"]["fn"])["f1"]
                delta = c2_f1 - c1_f1
                stem_deltas.append(delta)
                per_sequence[f"{record['seed']}:{sequence_hash}"] = {
                    "seed": record["seed"], "sequence_sha256": sequence_hash,
                    "name": c2["name"], "length": c2["length"], "length_bucket": c2["length_bucket"],
                    "C2_minus_C1_stem_f1": delta,
                }
        ordered = sorted(per_sequence.items(), key=lambda item: abs(item[1]["C2_minus_C1_stem_f1"]), reverse=True)
        top5 = ordered[:5]
        retained = [value["C2_minus_C1_stem_f1"] for key, value in ordered[5:]]
        total_abs = sum(abs(value["C2_minus_C1_stem_f1"]) for key, value in ordered)
        bucket_rows = defaultdict(list)
        for row in rows:
            bucket_rows[row["length_bucket"]].append(row)
        buckets = {}
        for bucket, selected_rows in sorted(bucket_rows.items()):
            summary = summarize_rows(selected_rows)
            buckets[bucket] = {
                "n_rows": len(selected_rows), "arms": summary,
                "C2_minus_C1": _deltas(summary),
            }
        step_means = {"arms": means, "C2_minus_C1": _deltas(means),
                      "C2_minus_C0": _deltas(means, "C2", "C0"),
                      "C2_minus_C3": _deltas(means, "C2", "C3")}
        by_step[str(step)] = {
            **step_means,
            "per_seed": per_seed,
            "all_seed_macro_pair_f1_C2_above_C1": all(
                value["C2_minus_C1"]["macro_pair_f1"] > 0 for value in per_seed.values()),
            "stem_micro_macro_sign_disagreement": (
                (step_means["C2_minus_C1"]["micro_stem_f1"] > 0) !=
                (step_means["C2_minus_C1"]["macro_stem_f1"] > 0)),
            "stem_length_buckets": buckets,
            "stem_few_sequence_check": {
                "top5_by_absolute_per_sequence_delta": [value for key, value in top5],
                "top5_absolute_delta_fraction": (sum(abs(value["C2_minus_C1_stem_f1"]) for key, value in top5) / total_abs
                                                   if total_abs else 0.0),
                "macro_stem_f1_C2_minus_C1_without_top5": _mean(retained) if retained else None,
            },
        }
    early = by_step["3200"]["C2_minus_C1"]["macro_pair_f1"]
    terminal = by_step["20000"]["C2_minus_C1"]["macro_pair_f1"]
    if terminal <= 0:
        trend = "DISAPPEARED_OR_REVERSED"
    elif terminal > early + 0.005:
        trend = "EXPANDED"
    elif terminal < early - 0.005:
        trend = "WEAKENED"
    else:
        trend = "STABLE"
    return {
        "status": "PASS", "scope": "DESCRIPTIVE_VL0_UNFREEZING_TRAJECTORY_ONLY",
        "steps": by_step,
        "C2_macro_pair_f1_advantage_trend": trend,
        "C2_advantage_persists_through_full_unfreeze": all(
            by_step[str(step)]["C2_minus_C1"]["macro_pair_f1"] > 0 and
            by_step[str(step)]["all_seed_macro_pair_f1_C2_above_C1"]
            for step in (16800, 20000)),
        "stem_micro_macro_disagreement_steps": [step for step in STEPS if
            by_step[str(step)]["stem_micro_macro_sign_disagreement"]],
        "terminal_matrix_verdict_modified": False,
        "external_evaluation_allowed": False,
        "production_repair": False, "sota_claim": False,
        "old_external_panels_used": False,
    }


def analyze_step(artifact_root, seed, step):
    if step == 20000:
        raise ValueError("terminal step is already authoritatively decoded; use --aggregate")
    if step not in STEPS:
        raise ValueError(f"unsupported milestone {step}")
    stage = artifact_root / f"full_seed{seed}"
    root = stage / "trajectory" / f"step{step}"
    on_root, off_root = root / "raw_train_on", root / "raw_train_off"
    on_meta = json.loads((on_root / "export_metadata.json").read_text())
    off_meta = json.loads((off_root / "export_metadata.json").read_text())
    terminal_summary = json.loads((stage / "decode_vl0/vl0_selected_summary.json").read_text())
    frozen = json.loads((stage / "decode_vl0/frozen_decode_config.json").read_text())
    contract_path = REPO / "configs/prior_alignment_2x2_v1.json"
    contract_sha = sha256_file(contract_path)
    assert on_meta["checkpoint_step"] == off_meta["checkpoint_step"] == step
    assert on_meta["n_selected"] == off_meta["n_selected"] == 196
    assert on_meta["seed"] == off_meta["seed"] == seed
    assert on_meta["shared_init_sha256"] == off_meta["shared_init_sha256"]
    assert on_meta["contract_sha256"] == off_meta["contract_sha256"] == contract_sha
    assert terminal_summary["terminal_step"] == 20000 and terminal_summary["seed"] == seed
    assert terminal_summary["contract_sha256"] == frozen["contract_sha256"] == contract_sha
    assert not on_meta["old_external_panels_used"] and not off_meta["old_external_panels_used"]
    checkpoint_receipts = {}
    for arm, meta in (("train_on", on_meta), ("train_off", off_meta)):
        checkpoint = stage / "snapshots" / f"{arm}_step{step}.pt"
        receipt = json.loads(Path(str(checkpoint) + ".json").read_text())
        assert receipt["checkpoint_sha256"] == meta["checkpoint"]["sha256"] == sha256_file(checkpoint)
        checkpoint_receipts[arm] = receipt
    on_files = {path.stem: path for path in (on_root / "bprna_vl0").glob("*.npz")}
    off_files = {path.stem: path for path in (off_root / "bprna_vl0").glob("*.npz")}
    if set(on_files) != set(off_files) or len(on_files) != 196:
        raise ValueError("milestone VL0 sequence identities are incomplete or mismatched")
    config = frozen["config"]
    rows = []
    for sequence_hash in sorted(on_files):
        with np.load(on_files[sequence_hash], allow_pickle=False) as on, np.load(off_files[sequence_hash], allow_pickle=False) as off:
            assert str(on["seq"]) == str(off["seq"]) and np.array_equal(on["gt_pairs"], off["gt_pairs"])
            seq = str(on["seq"])
            truth = {tuple(map(int, pair)) for pair in on["gt_pairs"]}
            mask = valid_pair_mask(seq)
            sources = {"C0": np.asarray(on["plana_on"], dtype=np.float64),
                       "C1": np.asarray(on["plana_off"], dtype=np.float64),
                       "C2": np.asarray(off["plana_off"], dtype=np.float64),
                       "C3": np.asarray(off["plana_on"], dtype=np.float64)}
            for arm, scores in sources.items():
                selected = config[arm]
                if arm == "C0":
                    predicted = nussinov_map(scores, mask)
                else:
                    _, posterior = inside_outside(scores / selected["temperature"], mask)
                    if not np.isfinite(posterior).all():
                        raise ValueError(f"non-finite posterior for {arm} {sequence_hash}")
                    predicted = decode_mea(posterior, mask, selected["gamma"])
                row = {"dataset_id": "bprna_vl0", "sequence_sha256": sequence_hash,
                       "name": str(on["name"]), "arm": arm, "length": len(seq),
                       "length_bucket": length_bucket(len(seq)),
                       "train_trajectory": "train_on" if arm in ("C0", "C1") else "train_off",
                       "decoder": selected["decoder"], "temperature": selected["temperature"],
                       "gamma": selected["gamma"], "n_truth_pairs": len(truth),
                       "n_predicted_pairs": len(predicted)}
                row.update(pair_and_stem_counts(predicted, truth))
                rows.append(row)
    summary = {"schema_version": 1, "mode": "full", "seed": seed, "checkpoint_step": step,
               "development_dataset": "bprna_vl0", "n_sequences": 196,
               "contract_sha256": contract_sha, "shared_init_sha256": on_meta["shared_init_sha256"],
               "checkpoint_receipts": checkpoint_receipts, "decode_config_source": "terminal_20000_VL0_frozen",
               "decode_config": config, "arms": summarize_rows(rows),
               "main_matrix_endpoint": False, "old_external_panels_used": False,
               "production_repair": False, "sota_claim": False}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "vl0_selected_per_sequence.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (root / "vl0_selected_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_records(artifact_root):
    records = []
    for seed in (0, 1, 2):
        stage = artifact_root / f"full_seed{seed}"
        lengths = {}
        for path in (stage / "raw_train_on/bprna_vl0").glob("*.npz"):
            with np.load(path, allow_pickle=False) as raw:
                lengths[path.stem] = len(str(raw["seq"]))
        for step in STEPS:
            root = stage / "decode_vl0" if step == 20000 else stage / "trajectory" / f"step{step}"
            summary = json.loads((root / "vl0_selected_summary.json").read_text())
            rows = _read_rows(root / "vl0_selected_per_sequence.jsonl")
            for row in rows:
                row["length"] = lengths[row["sequence_sha256"]]
                row["length_bucket"] = length_bucket(row["length"])
            assert summary["seed"] == seed and not summary["old_external_panels_used"]
            records.append({"seed": seed, "step": step, "summary": summary, "rows": rows})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate_trajectory(collect_records(args.artifact_root))
        out = args.out or args.artifact_root / "unfreeze_trajectory_analysis.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in (
            "status", "C2_macro_pair_f1_advantage_trend", "C2_advantage_persists_through_full_unfreeze",
            "stem_micro_macro_disagreement_steps", "terminal_matrix_verdict_modified")}, indent=2))
    else:
        if args.seed is None or args.step is None:
            parser.error("--seed and --step are required unless --aggregate is used")
        result = analyze_step(args.artifact_root, args.seed, args.step)
        print(json.dumps({"status": "PASS", "seed": args.seed, "step": args.step,
                          "main_matrix_endpoint": False, "arms": result["arms"]}, indent=2))


if __name__ == "__main__":
    main()
