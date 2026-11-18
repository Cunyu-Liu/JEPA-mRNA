#!/usr/bin/env python3
"""Dissect high-frequency TR1c-environment FN without changing any model or decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_coverage_audit import (MAIN_SPLITS, dataset_rows, helix_bin, keys_for, loop_bin,
                                 normalize_pairs, pair_geometry, register_bin, seq_of, sha256)
from pair_decode_common import decode_mea
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask

HIGH_FREQUENCY_MIN = 100
HIGH_MARGINAL_THRESHOLD = 0.5


def _ratio(n, d):
    return float(n / d) if d else None


def _source(row, fallback):
    return str(row.get("source") or row.get("label_source") or fallback or "UNKNOWN")


def _summarize(rows):
    n = len(rows)
    fn = [r for r in rows if r["candidate_status"] == "FN"]
    high = [r for r in fn if r["posterior"] >= HIGH_MARGINAL_THRESHOLD]
    exact = [r for r in rows if r["exact_sequence_label_status"] != "NO_EXACT_TR1C_SEQUENCE"]
    conflict = [r for r in rows if r["exact_sequence_label_status"] == "CONFLICT"]
    return {
        "n_high_frequency_pairs": n, "n_candidate_fn": len(fn),
        "candidate_fn_rate": _ratio(len(fn), n),
        "n_high_marginal_decoder_removed": len(high),
        "n_low_marginal_model_score": len(fn) - len(high),
        "high_marginal_fraction_among_fn": _ratio(len(high), len(fn)),
        "median_fn_posterior": float(np.median([r["posterior"] for r in fn])) if fn else None,
        "n_persistent_fn_vs_turner_on_map": sum(r["candidate_status"] == r["baseline_map_status"] == "FN" for r in rows),
        "n_recovered_from_turner_on_map": sum(r["candidate_status"] == "TP" and r["baseline_map_status"] == "FN" for r in rows),
        "n_lost_from_turner_on_map": sum(r["candidate_status"] == "FN" and r["baseline_map_status"] == "TP" for r in rows),
        "n_shared_tp_vs_turner_on_map": sum(r["candidate_status"] == r["baseline_map_status"] == "TP" for r in rows),
        "n_turner_on_map_fn": sum(r["baseline_map_status"] == "FN" for r in rows),
        "candidate_fn_change_vs_turner_on_map": (sum(r["candidate_status"] == "FN" and r["baseline_map_status"] == "TP" for r in rows) -
                                                   sum(r["candidate_status"] == "TP" and r["baseline_map_status"] == "FN" for r in rows)),
        "n_exact_sequence_evaluable_pairs": len(exact),
        "exact_sequence_evaluable_fraction": _ratio(len(exact), n),
        "n_exact_sequence_label_conflicts": len(conflict),
        "exact_sequence_conflict_fraction": _ratio(len(conflict), len(exact)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
    parser.add_argument("--registry", type=Path, default=REPO / "manifests/foldnar_v1/data_registry.json")
    parser.add_argument("--coverage", type=Path, default=root / "pair_environment_coverage.parquet")
    parser.add_argument("--raw-root", type=Path, default=root / "raw_scores_chunk64/xens2")
    parser.add_argument("--frozen-config", type=Path, default=root / "decode_xens2_vl0/frozen_decode_config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = {row["dataset_id"]: row for row in json.loads(args.registry.read_text())["rows"]}
    frozen = json.loads(args.frozen_config.read_text())
    mea = frozen["config"]["xens2_off"]["mea"]
    temperature, gamma = float(mea["temperature"]), float(mea["gamma"])
    if (temperature, gamma) != (2.0, 4.0) or frozen.get("development_dataset") != "bprna_vl0":
        raise ValueError("this diagnostic is bound to the frozen xens2_off MEA(T=2,gamma=4) candidate")
    coverage = pd.read_parquet(args.coverage)
    coverage = coverage[coverage.dataset_id.isin(MAIN_SPLITS) & coverage.legal_mask &
                        (coverage.train_count_local >= HIGH_FREQUENCY_MIN)]
    pair_rows = []
    target_local_keys = set()
    target_sequences = defaultdict(list)
    full_counts = {}
    baseline_full_counts = {}
    for dataset_id in MAIN_SPLITS:
        files = sorted((args.raw_root / dataset_id).glob("*.npz"))
        if len(files) != 64:
            raise ValueError(f"expected 64 frozen raw files for {dataset_id}, found {len(files)}")
        aggregate = Counter()
        baseline_aggregate = Counter()
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                seq, name = str(z["seq"]), str(z["name"])
                seq_hash = str(z["sequence_sha256"])
                truth = {tuple(map(int, p)) for p in z["gt_pairs"]}
                mask = valid_pair_mask(seq)
                on = np.asarray(z["xens2_on"], dtype=np.float64)
                off = np.asarray(z["xens2_off"], dtype=np.float64)
                _, posterior = inside_outside(off / temperature, mask)
                if not np.isfinite(posterior).all():
                    raise ValueError(f"non-finite posterior: {dataset_id}/{name}")
                candidate = set(map(tuple, decode_mea(posterior, mask, gamma)))
                baseline = set(map(tuple, nussinov_map(on, mask)))
                tp, fp, fn = len(candidate & truth), len(candidate - truth), len(truth - candidate)
                aggregate.update({"tp": tp, "fp": fp, "fn": fn, "n_truth": len(truth)})
                baseline_aggregate.update({"tp": len(baseline & truth), "fp": len(baseline - truth),
                                           "fn": len(truth - baseline), "n_truth": len(truth)})
                selected = coverage[(coverage.dataset_id == dataset_id) &
                                    (coverage.sequence_sha256 == seq_hash) &
                                    (coverage.name == name)]
                posterior_legal = posterior[mask]
                for row in selected.to_dict("records"):
                    pair = (int(row["pair_i"]), int(row["pair_j"]))
                    if pair not in truth:
                        raise ValueError(f"coverage/raw GT mismatch: {dataset_id}/{name}/{pair}")
                    local_key = keys_for(row)[2]
                    target_local_keys.add(local_key)
                    record = {
                        "dataset_id": dataset_id, "name": name, "sequence_sha256": seq_hash,
                        "pair_i": pair[0], "pair_j": pair[1], "pair_type": row["pair_type"],
                        "span": row["span"], "length": row["length"], "length_bin": row["length_bin"],
                        "helix_length": row["helix_length"], "helix_bin": helix_bin(row["helix_length"]),
                        "register": row["register"], "register_bin": register_bin(row["register"]),
                        "loop_bin": loop_bin(row), "source": row.get("source") or "UNKNOWN",
                        "label_source": row.get("label_source") or "UNKNOWN",
                        "family": row.get("family") or "UNKNOWN", "clan": row.get("clan") or "UNKNOWN",
                        "train_count_local": int(row["train_count_local"]),
                        "train_count_local_exact": int(row["train_count_local_exact"]),
                        "local_key": local_key,
                        "candidate_status": "TP" if pair in candidate else "FN",
                        "baseline_map_status": "TP" if pair in baseline else "FN",
                        "posterior": float(posterior[pair]),
                        "raw_score_off": float(off[pair]),
                        "posterior_legal_percentile": float(np.mean(posterior_legal <= posterior[pair])),
                    }
                    pair_rows.append(record)
                    target_sequences[seq_hash].append((pair, local_key))
        full_counts[dataset_id] = dict(aggregate)
        baseline_full_counts[dataset_id] = dict(baseline_aggregate)
    if len({r["sequence_sha256"] for r in pair_rows}) == 0:
        raise ValueError("no high-frequency pairs selected")
    # Positive occurrence counts by source are not treated as negative labels.
    local_source_counts = defaultdict(Counter)
    exact_sequence_labels = defaultdict(list)
    train_spec = registry["bprna_tr1c"]
    for line_number, row in dataset_rows(registry, "bprna_tr1c"):
        seq = seq_of(row)
        seq_hash = hashlib.sha256(seq.encode()).hexdigest()
        pairs, _, _ = normalize_pairs(row.get("pairs"), len(seq))
        source = _source(row, train_spec.get("label_source"))
        if seq_hash in target_sequences:
            exact_sequence_labels[seq_hash].append({"source": source, "pairs": pairs,
                                                    "name": str(row.get("name", row.get("id", line_number)))})
        for feature in pair_geometry(seq, pairs).values():
            if not feature["legal_mask"]:
                continue
            local_key = keys_for(feature)[2]
            if local_key in target_local_keys:
                local_source_counts[local_key][source] += 1
    for record in pair_rows:
        seq_hash, pair = record["sequence_sha256"], (record["pair_i"], record["pair_j"])
        distribution = dict(local_source_counts[record["local_key"]])
        record["tr1c_local_positive_sources_json"] = json.dumps(distribution, sort_keys=True)
        record["tr1c_local_source_count"] = len(distribution)
        labels = exact_sequence_labels.get(seq_hash, [])
        if not labels:
            status, positive, negative = "NO_EXACT_TR1C_SEQUENCE", [], []
        else:
            positive = sorted({x["source"] for x in labels if pair in x["pairs"]})
            negative = sorted({x["source"] for x in labels if pair not in x["pairs"]})
            if positive and negative:
                status = "CONFLICT"
            elif positive:
                status = "POSITIVE_ONLY"
            else:
                status = "NEGATIVE_ONLY"
        record["exact_sequence_label_status"] = status
        record["exact_sequence_positive_sources"] = "|".join(positive)
        record["exact_sequence_negative_sources"] = "|".join(negative)
        record["fn_mechanism"] = ("NOT_FN" if record["candidate_status"] == "TP" else
                                  "HIGH_MARGINAL_DECODER_REMOVED" if record["posterior"] >= HIGH_MARGINAL_THRESHOLD
                                  else "LOW_MARGINAL_MODEL_SCORE")
    args.out.mkdir(parents=True, exist_ok=True)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_parquet(args.out / "high_frequency_fn_per_pair.parquet", index=False)
    summaries = {"ALL": _summarize(pair_rows)}
    for dataset_id in MAIN_SPLITS:
        summaries[dataset_id] = _summarize([r for r in pair_rows if r["dataset_id"] == dataset_id])
    bucket_rows = []
    for dimension in ("length_bin", "loop_bin", "helix_bin", "register_bin", "source", "label_source"):
        for bucket in sorted(set(pair_df[dimension].astype(str))):
            selected = [r for r in pair_rows if str(r[dimension]) == bucket]
            bucket_rows.append({"dimension": dimension, "bucket": bucket, **_summarize(selected)})
    pd.DataFrame(bucket_rows).to_csv(args.out / "high_frequency_fn_buckets.csv", index=False)
    source_json = {key: dict(value) for key, value in sorted(local_source_counts.items())}
    (args.out / "tr1c_local_positive_source_counts.json").write_text(json.dumps(source_json, indent=2) + "\n")
    report = {
        "schema_version": 1, "scope": "FROZEN_CANDIDATE_NON_INDEPENDENT_DIAGNOSTIC_ONLY",
        "candidate": "xens2_turneroff_mea_v1", "temperature": temperature, "gamma": gamma,
        "high_frequency_definition": f"legal pair and TR1c local environment count >= {HIGH_FREQUENCY_MIN}",
        "high_marginal_definition": f"inside-outside posterior >= {HIGH_MARGINAL_THRESHOLD}",
        "label_conflict_definition": "same sequence and coordinates are positive in one TR1c source and absent in another; local positive counts alone cannot establish a conflict",
        "old_external_panels_used_for_route_or_parameter_selection": False,
        "n_sequences": len({(r["dataset_id"], r["name"]) for r in pair_rows}), "n_pairs": len(pair_rows),
        "summaries": summaries, "full_panel_candidate_pair_counts": full_counts,
        "full_panel_turner_on_map_pair_counts": baseline_full_counts,
        "input_sha256": {"registry": sha256(args.registry), "coverage": sha256(args.coverage),
                         "frozen_config": sha256(args.frozen_config)},
    }
    (args.out / "high_frequency_fn_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_pairs": len(pair_rows), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
