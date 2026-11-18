#!/usr/bin/env python3
"""Recompute every external raw/DP row from NPZ and independently verify ranking metrics."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask
from pair_decode_common import decode_centroid, decode_mea


def prf(pred, truth):
    pred = {tuple(map(int, p)) for p in pred}
    truth = {tuple(map(int, p)) for p in truth}
    tp, fp = len(pred & truth), len(pred - truth)
    fn = len(truth - pred)
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": tp + fp,
            "precision": p, "recall": r, "f1": f}


def independent_ap(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="mergesort")
    y, s = labels[order], scores[order]
    distinct = np.where(np.diff(s))[0]
    indices = np.r_[distinct, len(s) - 1]
    tp = np.cumsum(y)[indices].astype(float)
    fp = (indices + 1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ranking_stats(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y, s = labels[order], scores[order]
    distinct = np.where(np.diff(s))[0]
    indices = np.r_[distinct, len(s) - 1]
    tp = np.cumsum(y)[indices].astype(float)
    fp = (indices + 1) - tp
    p = tp / np.maximum(tp + fp, 1)
    r = tp / tp[-1] if tp[-1] else np.zeros_like(tp)
    f = 2 * p * r / np.maximum(p + r, 1e-15)
    thresholds = s[indices]
    best = max(range(len(indices)), key=lambda i: (f[i], r[i], p[i], -thresholds[i]))
    out = {"best_threshold_f1": float(f[best])}
    for target in (.5, .7):
        candidates = [i for i in range(len(indices)) if p[i] >= target]
        if candidates:
            chosen = max(candidates, key=lambda i: (r[i], p[i], -thresholds[i]))
            out[f"recall_at_precision_{target}"] = float(r[chosen])
        else:
            out[f"recall_at_precision_{target}"] = 0.0
    return out


def key(row, columns):
    return tuple(None if pd.isna(row.get(c)) else row.get(c) for c in columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-dir", type=Path, required=True)
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    args = ap.parse_args()
    manifest = json.loads((args.decode_dir / "decode_run_manifest.json").read_text())
    raw_root = Path(manifest["raw_root"])
    frozen = json.loads((args.decode_dir / "frozen_decode_config.json").read_text())
    config = frozen["config"]
    reported = pd.read_parquet(args.decode_dir / "decode_per_sequence.parquet")
    reported_rank = pd.read_parquet(args.decode_dir / "ranking_per_sequence.parquet")
    reported_rank_summary = pd.read_parquet(args.decode_dir / "ranking_summary.parquet")
    actual, actual_rank, pooled = [], [], {}
    datasets = sorted(reported.dataset_id.unique())
    for dataset in datasets:
        files = sorted((raw_root / dataset).glob("*.npz"))
        if args.limit_per_dataset:
            files = files[:args.limit_per_dataset]
        for index, path in enumerate(files, 1):
            with np.load(path, allow_pickle=False) as z:
                name, seq = str(z["name"]), str(z["seq"])
                truth = {tuple(map(int, p)) for p in z["gt_pairs"]}
                mask = valid_pair_mask(seq)
                gt_mask = np.zeros((len(seq), len(seq)), dtype=bool)
                for i, j in truth:
                    if 0 <= i < len(seq) and 0 <= j < len(seq):
                        gt_mask[i, j] = True
                labels = gt_mask[mask]
                for variant in sorted(k for k in z.files if k.endswith("_on") or k.endswith("_off")):
                    cfg = config[variant]
                    scores = np.asarray(z[variant], dtype=np.float64)
                    raw = scores[mask]
                    base = {"dataset_id": dataset, "name": name, "score_variant": variant,
                            "n_gt": len(truth), "n_legal_gt": int(labels.sum())}
                    def add(decoder, temperature, gamma, pred):
                        row = dict(base, decoder=decoder, temperature=float(temperature),
                                   gamma=float(gamma) if gamma is not None else None)
                        row.update(prf(pred, truth))
                        actual.append(row)
                    add("MAP", 1.0, None, nussinov_map(scores, mask))
                    for label in ("max_f1", "P50", "P70"):
                        threshold = cfg["raw_independent"][label]["threshold"]
                        selected = np.asarray(raw) >= threshold
                        pred = [(int(i), int(j)) for (i, j), keep in zip(np.argwhere(mask), selected) if keep]
                        add(f"INDEPENDENT_RAW_{label.upper().replace('_', '')}", 1.0, None, pred)
                    rank_base = dict(base, score_kind="RAW", temperature=1.0,
                                     average_precision=independent_ap(labels, raw), n_legal_candidates=len(raw))
                    actual_rank.append(rank_base)
                    pool_key = (dataset, variant, "RAW", 1.0)
                    pooled.setdefault(pool_key, ([], []))
                    pooled[pool_key][0].append(labels); pooled[pool_key][1].append(raw)
                    needed = {cfg["marginal_independent"]["temperature"], cfg["mea"]["temperature"], cfg["centroid"]["temperature"]}
                    for temperature in sorted(needed):
                        _, posterior = inside_outside(scores / temperature, mask)
                        post = posterior[mask]
                        actual_rank.append(dict(base, score_kind="POSTERIOR", temperature=float(temperature),
                                                average_precision=independent_ap(labels, post), n_legal_candidates=len(post)))
                        if temperature == cfg["marginal_independent"]["temperature"]:
                            for label in ("max_f1", "P50", "P70"):
                                threshold = cfg["marginal_independent"][label]["threshold"]
                                selected = post >= threshold
                                pred = [(int(i), int(j)) for (i, j), keep in zip(np.argwhere(mask), selected) if keep]
                                add(f"INDEPENDENT_POST_{label.upper().replace('_', '')}", temperature, None, pred)
                            pool_key = (dataset, variant, "POSTERIOR", float(temperature))
                            pooled.setdefault(pool_key, ([], []))
                            pooled[pool_key][0].append(labels); pooled[pool_key][1].append(post)
                        if temperature == cfg["mea"]["temperature"]:
                            add("MEA", temperature, cfg["mea"]["gamma"], decode_mea(posterior, mask, cfg["mea"]["gamma"]))
                        if temperature == cfg["centroid"]["temperature"]:
                            add("CENTROID", temperature, None, decode_centroid(posterior, mask))
            if index % 20 == 0:
                print(f"[verify-external] {dataset} {index}/{len(files)}", flush=True)
    failures = []
    columns = ["dataset_id", "name", "score_variant", "decoder", "temperature", "gamma"]
    rep = {key(r, columns): r for r in reported.to_dict("records")}
    act = {key(r, columns): r for r in actual}
    if set(rep) != set(act):
        failures.append({"check": "decode_key_sets", "reported_only": len(set(rep)-set(act)), "actual_only": len(set(act)-set(rep))})
    for k in set(rep) & set(act):
        for field in ("tp", "fp", "fn", "n_gt", "n_legal_gt"):
            if int(rep[k][field]) != int(act[k][field]):
                failures.append({"check": "decode_counts", "key": k, "field": field,
                                 "reported": rep[k][field], "actual": act[k][field]})
    rank_columns = ["dataset_id", "name", "score_variant", "score_kind", "temperature"]
    rep_rank = {key(r, rank_columns): r for r in reported_rank.to_dict("records")}
    act_rank = {key(r, rank_columns): r for r in actual_rank}
    if set(rep_rank) != set(act_rank):
        failures.append({"check": "ranking_key_sets", "reported_only": len(set(rep_rank)-set(act_rank)), "actual_only": len(set(act_rank)-set(rep_rank))})
    for k in set(rep_rank) & set(act_rank):
        a, b = rep_rank[k]["average_precision"], act_rank[k]["average_precision"]
        if (pd.isna(a) != pd.isna(b)) or (not pd.isna(a) and not np.isclose(a, b, atol=1e-12, rtol=1e-10)):
            failures.append({"check": "per_sequence_ap", "key": k, "reported": a, "actual": b})
    summary_actual = {}
    for k, (label_parts, score_parts) in pooled.items():
        labels, scores = np.concatenate(label_parts), np.concatenate(score_parts)
        aps = [r["average_precision"] for r in actual_rank
               if (r["dataset_id"], r["score_variant"], r["score_kind"], float(r["temperature"])) == k
               and not pd.isna(r["average_precision"])]
        summary_actual[k] = dict(micro_average_precision=independent_ap(labels, scores),
                                 macro_average_precision=float(np.mean(aps)) if aps else None,
                                 **ranking_stats(labels, scores))
    for row in reported_rank_summary.to_dict("records"):
        k = (row["dataset_id"], row["score_variant"], row["score_kind"], float(row["temperature"]))
        expected = summary_actual.get(k)
        if expected is None:
            failures.append({"check": "ranking_summary_missing_actual", "key": k}); continue
        for field, value in expected.items():
            reported_value = row[field]
            if (pd.isna(reported_value) != (value is None)) or (value is not None and not np.isclose(reported_value, value, atol=1e-12, rtol=1e-10)):
                failures.append({"check": "ranking_summary_metric", "key": k, "field": field,
                                 "reported": reported_value, "actual": value})
    result = {"status": "PASS" if not failures else "FAIL", "n_decode_rows_recomputed": len(actual),
              "n_ranking_rows_recomputed": len(actual_rank), "n_ranking_summary_recomputed": len(summary_actual),
              "n_failed": len(failures), "failures": failures[:100]}
    (args.decode_dir / "independent_external_raw_dp_verification.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
