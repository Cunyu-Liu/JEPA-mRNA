#!/usr/bin/env python3
"""Tune pair decoders on VL0 only, then apply them once to frozen external panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask  # noqa: E402
from pair_decode_common import decode_centroid, decode_mea  # noqa: E402

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0)
TARGETS = (("P50", 0.5), ("P70", 0.7))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prf_counts(tp, fp, n_gt):
    fn = int(n_gt - tp)
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "n_pred": int(tp + fp),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def base_row(dataset, name, variant, decoder, T, gamma, n_gt, n_legal):
    return {"dataset_id": dataset, "name": name, "score_variant": variant, "decoder": decoder,
            "temperature": float(T), "gamma": float(gamma) if gamma is not None else None,
            "n_gt": int(n_gt), "n_legal_gt": int(n_legal)}


def pred_row(dataset, name, variant, decoder, T, gamma, pred, truth, n_legal):
    row = base_row(dataset, name, variant, decoder, T, gamma, len(truth), n_legal)
    pred = set(map(tuple, pred))
    row.update(prf_counts(len(pred & truth), len(pred - truth), len(truth)))
    return row


def selection_row(entry, variant, decoder, T, scores, threshold):
    selected = np.asarray(scores) >= threshold
    labels = entry["labels"]
    tp = int(np.count_nonzero(selected & labels))
    fp = int(np.count_nonzero(selected & ~labels))
    row = base_row(entry["dataset"], entry["name"], variant, decoder, T, None, entry["n_gt"], entry["n_legal"])
    row.update(prf_counts(tp, fp, entry["n_gt"]))
    row["threshold"] = float(threshold)
    return row


def ap_score(labels, scores):
    return float(average_precision_score(labels, scores)) if np.any(labels) else None


def rank_row(entry, variant, kind, T, scores):
    return {"dataset_id": entry["dataset"], "name": entry["name"], "score_variant": variant,
            "score_kind": kind, "temperature": float(T), "average_precision": ap_score(entry["labels"], scores),
            "n_legal_candidates": len(scores), "n_legal_gt": entry["n_legal"], "n_gt": entry["n_gt"]}


def pr_thresholds(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = np.arange(len(thresholds))
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-15)
    best = max(valid, key=lambda i: (f1[i], r[i], p[i], -float(thresholds[i])))
    result = {"max_f1": {"threshold": float(thresholds[best]), "precision": float(p[best]),
                         "recall": float(r[best]), "f1": float(f1[best])}}
    for label, target in TARGETS:
        candidates = [i for i in valid if p[i] >= target]
        if candidates:
            chosen = max(candidates, key=lambda i: (r[i], p[i], -float(thresholds[i])))
            result[label] = {"threshold": float(thresholds[chosen]), "precision": float(p[chosen]),
                             "recall": float(r[chosen]), "f1": float(f1[chosen]), "target_precision": target}
        else:
            result[label] = {"threshold": float(np.max(scores) + 1e-12), "precision": None,
                             "recall": 0.0, "f1": None, "target_precision": target}
    return result


def score_keys(z):
    return sorted(k for k in z.files if (k.endswith("_on") or k.endswith("_off")) and z[k].ndim == 2)


def entries_for_file(path, dataset, temperatures, full_grid):
    dp_rows, rank_rows, entries = [], [], []
    with np.load(path, allow_pickle=False) as z:
        name, seq = str(z["name"]), str(z["seq"])
        truth = {tuple(map(int, p)) for p in z["gt_pairs"]}
        mask = valid_pair_mask(seq)
        gt_mask = np.zeros((len(seq), len(seq)), dtype=bool)
        for i, j in truth:
            if 0 <= i < len(seq) and 0 <= j < len(seq):
                gt_mask[i, j] = True
        labels = gt_mask[mask]
        n_legal = int(labels.sum())
        for variant in score_keys(z):
            scores = np.asarray(z[variant], dtype=np.float64)
            raw = scores[mask]
            base = {"dataset": dataset, "name": name, "raw": raw, "post": raw,
                    "labels": labels, "n_gt": len(truth), "n_legal": n_legal,
                    "variant": variant, "temperature": 1.0, "kind": "RAW_BASE"}
            entries.append(base)
            rank_rows.append(rank_row(base, variant, "RAW", 1.0, raw))
            dp_rows.append(pred_row(dataset, name, variant, "MAP", 1.0, None,
                                    nussinov_map(scores, mask), truth, n_legal))
            for T in temperatures:
                _, posterior = inside_outside(scores / T, mask)
                entry = {"dataset": dataset, "name": name, "raw": raw, "post": posterior[mask],
                         "labels": labels, "n_gt": len(truth), "n_legal": n_legal,
                         "variant": variant, "temperature": float(T), "kind": "POSTERIOR"}
                entries.append(entry)
                rank_rows.append(rank_row(entry, variant, "POSTERIOR", T, posterior[mask]))
                if full_grid:
                    for gamma in GAMMAS:
                        dp_rows.append(pred_row(dataset, name, variant, "MEA", T, gamma,
                                                decode_mea(posterior, mask, gamma), truth, n_legal))
                    dp_rows.append(pred_row(dataset, name, variant, "CENTROID", T, None,
                                            decode_centroid(posterior, mask), truth, n_legal))
    return dp_rows, rank_rows, entries


def aggregate(rows, group_columns):
    if not rows:
        return []
    df = pd.DataFrame(rows)
    output = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tp, fp, fn = (int(group[k].sum()) for k in ("tp", "fp", "fn"))
        micro = prf_counts(tp, fp, tp + fn)
        row = dict(zip(group_columns, keys))
        row.update({f"micro_{k}": micro[k] for k in ("tp", "fp", "fn", "n_pred", "precision", "recall", "f1")})
        row.update({f"macro_{k}": float(group[k].mean()) for k in ("precision", "recall", "f1")})
        row["n_sequences"] = len(group)
        row["n_gt"] = int(group["n_gt"].sum())
        row["n_legal_gt"] = int(group["n_legal_gt"].sum())
        output.append(row)
    return output


def choose_summary(rows, decoder):
    candidates = [r for r in rows if r["decoder"] == decoder]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["micro_f1"], r["macro_f1"], r["micro_precision"],
                                          -r["temperature"], -(r["gamma"] or 0)))


def make_ranking_summary(entries, rank_rows, dataset, external=False, config=None):
    df = pd.DataFrame([r for r in rank_rows if r["dataset_id"] == dataset])
    output = []
    variants = sorted({e["variant"] for e in entries if e["dataset"] == dataset})
    for variant in variants:
        for kind in ("RAW", "POSTERIOR"):
            entry_kind = "RAW_BASE" if kind == "RAW" else "POSTERIOR"
            temperatures = sorted({e["temperature"] for e in entries if e["dataset"] == dataset and e["variant"] == variant and e["kind"] == entry_kind})
            for T in temperatures:
                if kind == "RAW" and T != 1.0:
                    continue
                if external and kind == "POSTERIOR" and T != config[variant]["marginal_independent"]["temperature"]:
                    continue
                selected = [e for e in entries if e["dataset"] == dataset and e["variant"] == variant and e["temperature"] == T and e["kind"] == entry_kind]
                scores = np.concatenate([e["raw" if kind == "RAW" else "post"] for e in selected])
                labels = np.concatenate([e["labels"] for e in selected])
                pr = pr_thresholds(labels, scores)
                aps = df[(df.score_variant == variant) & (df.score_kind == kind) & (df.temperature == T)]["average_precision"].dropna()
                output.append({"dataset_id": dataset, "score_variant": variant, "score_kind": kind, "temperature": T,
                               "micro_average_precision": ap_score(labels, scores), "macro_average_precision": float(aps.mean()) if len(aps) else None,
                               "n_ap_sequences": len(aps), "n_legal_candidates": len(labels), "n_legal_gt": int(labels.sum()),
                               "recall_at_precision_0.5": pr["P50"]["recall"], "recall_at_precision_0.7": pr["P70"]["recall"],
                               "best_threshold_f1": pr["max_f1"]["f1"]})
    return output


def tune_vl0(raw_root, limit=0):
    dataset = "bprna_vl0"
    dp_rows, rank_rows, entries = [], [], []
    files = sorted((raw_root / dataset).glob("*.npz"))
    if limit:
        files = files[:limit]
    for index, path in enumerate(files, 1):
        d, r, e = entries_for_file(path, dataset, TEMPERATURES, True)
        dp_rows += d
        rank_rows += r
        entries += e
        if index % 20 == 0:
            print(f"[decode-tune] {index}/{len(files)}", flush=True)
    config = {}
    variants = sorted({e["variant"] for e in entries})
    for variant in variants:
        current = [e for e in entries if e["variant"] == variant]
        raw_pool = [e for e in current if e["kind"] == "RAW_BASE"]
        raw_labels = np.concatenate([e["labels"] for e in raw_pool])
        raw_scores = np.concatenate([e["raw"] for e in raw_pool])
        raw_cfg = pr_thresholds(raw_labels, raw_scores)
        for label in ("max_f1", "P50", "P70"):
            for e in raw_pool:
                dp_rows.append(selection_row(e, variant, f"INDEPENDENT_RAW_{label.upper().replace('_', '')}", 1.0, e["raw"], raw_cfg[label]["threshold"]))
        dp_summary = aggregate([r for r in dp_rows if r["score_variant"] == variant],
                               ["dataset_id", "score_variant", "decoder", "temperature", "gamma"])
        mea = choose_summary(dp_summary, "MEA")
        centroid = choose_summary(dp_summary, "CENTROID")
        marginal_candidates = []
        marginal_cfg_by_t = {}
        for T in TEMPERATURES:
            selected = [e for e in current if e["temperature"] == T and e["kind"] == "POSTERIOR"]
            labels = np.concatenate([e["labels"] for e in selected])
            scores = np.concatenate([e["post"] for e in selected])
            cfg = pr_thresholds(labels, scores)
            marginal_cfg_by_t[T] = cfg
            new_rows = [selection_row(e, variant, "INDEPENDENT_POST_MAXF1", T, e["post"], cfg["max_f1"]["threshold"]) for e in selected]
            dp_rows += new_rows
            summary = aggregate(new_rows, ["dataset_id", "score_variant", "decoder", "temperature", "gamma"])[0]
            marginal_candidates.append(summary)
        best_post = choose_summary(marginal_candidates, "INDEPENDENT_POST_MAXF1")
        best_t = best_post["temperature"]
        best_post_cfg = marginal_cfg_by_t[best_t]
        for label in ("P50", "P70"):
            for e in [e for e in current if e["temperature"] == best_t and e["kind"] == "POSTERIOR"]:
                dp_rows.append(selection_row(e, variant, f"INDEPENDENT_POST_{label}", best_t, e["post"], best_post_cfg[label]["threshold"]))
        config[variant] = {
            "raw_independent": raw_cfg,
            "marginal_independent": {"temperature": best_t, **best_post_cfg},
            "mea": {"temperature": mea["temperature"], "gamma": mea["gamma"], "vl0_micro_f1": mea["micro_f1"], "vl0_macro_f1": mea["macro_f1"]},
            "centroid": {"temperature": centroid["temperature"], "vl0_micro_f1": centroid["micro_f1"], "vl0_macro_f1": centroid["macro_f1"]},
            "map": {"temperature": 1.0},
        }
    return dp_rows, rank_rows, entries, config


def apply_external(raw_root, dataset, config, limit=0):
    dp_rows, rank_rows, entries = [], [], []
    files = sorted((raw_root / dataset).glob("*.npz"))
    if limit:
        files = files[:limit]
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
            n_legal = int(labels.sum())
            for variant in score_keys(z):
                cfg = config[variant]
                scores = np.asarray(z[variant], dtype=np.float64)
                raw = scores[mask]
                dp_rows.append(pred_row(dataset, name, variant, "MAP", 1.0, None,
                                        nussinov_map(scores, mask), truth, n_legal))
                base = {"dataset": dataset, "name": name, "raw": raw, "post": raw,
                        "labels": labels, "n_gt": len(truth), "n_legal": n_legal,
                        "variant": variant, "temperature": 1.0, "kind": "RAW_BASE"}
                entries.append(base)
                rank_rows.append(rank_row(base, variant, "RAW", 1.0, raw))
                for label in ("max_f1", "P50", "P70"):
                    dp_rows.append(selection_row(base, variant, f"INDEPENDENT_RAW_{label.upper().replace('_', '')}", 1.0,
                                                 raw, cfg["raw_independent"][label]["threshold"]))
                needed = {cfg["marginal_independent"]["temperature"], cfg["mea"]["temperature"], cfg["centroid"]["temperature"]}
                for T in sorted(needed):
                    _, posterior = inside_outside(scores / T, mask)
                    entry = {"dataset": dataset, "name": name, "raw": raw, "post": posterior[mask],
                             "labels": labels, "n_gt": len(truth), "n_legal": n_legal,
                             "variant": variant, "temperature": float(T), "kind": "POSTERIOR"}
                    entries.append(entry)
                    rank_rows.append(rank_row(entry, variant, "POSTERIOR", T, posterior[mask]))
                    if T == cfg["marginal_independent"]["temperature"]:
                        for label in ("max_f1", "P50", "P70"):
                            dp_rows.append(selection_row(entry, variant, f"INDEPENDENT_POST_{label.upper().replace('_', '')}", T,
                                                         posterior[mask], cfg["marginal_independent"][label]["threshold"]))
                    if T == cfg["mea"]["temperature"]:
                        dp_rows.append(pred_row(dataset, name, variant, "MEA", T, cfg["mea"]["gamma"],
                                                decode_mea(posterior, mask, cfg["mea"]["gamma"]), truth, n_legal))
                    if T == cfg["centroid"]["temperature"]:
                        dp_rows.append(pred_row(dataset, name, variant, "CENTROID", T, None,
                                                decode_centroid(posterior, mask), truth, n_legal))
        if index % 20 == 0:
            print(f"[decode-external] {dataset} {index}/{len(files)}", flush=True)
    return dp_rows, rank_rows, entries


def route_rows(decode_summary, ranking_summary):
    summary = pd.DataFrame(decode_summary)
    ranking = pd.DataFrame(ranking_summary)
    output = []
    for (dataset, variant), group in summary.groupby(["dataset_id", "score_variant"]):
        map_row = group[group.decoder == "MAP"].iloc[0]
        raw_rank = ranking[(ranking.dataset_id == dataset) & (ranking.score_variant == variant) & (ranking.score_kind == "RAW")].iloc[0]
        row = {"dataset_id": dataset, "score_variant": variant,
               "raw_micro_average_precision": raw_rank.micro_average_precision,
               "raw_macro_average_precision": raw_rank.macro_average_precision,
               "raw_recall_at_precision_0.5": raw_rank["recall_at_precision_0.5"],
               "raw_recall_at_precision_0.7": raw_rank["recall_at_precision_0.7"],
               "map_micro_f1": map_row.micro_f1, "map_macro_f1": map_row.macro_f1,
               "map_micro_precision": map_row.micro_precision, "map_micro_recall": map_row.micro_recall}
        for decoder, prefix in (("MEA", "mea"), ("CENTROID", "centroid"),
                                ("INDEPENDENT_RAW_MAXF1", "raw_independent"),
                                ("INDEPENDENT_POST_MAXF1", "posterior_independent")):
            selected = group[group.decoder == decoder]
            if len(selected):
                best = max(selected.to_dict("records"), key=lambda r: (r["micro_f1"], r["macro_f1"]))
                row[f"{prefix}_micro_f1"] = best["micro_f1"]
                row[f"{prefix}_macro_f1"] = best["macro_f1"]
                row[f"{prefix}_micro_precision"] = best["micro_precision"]
                row[f"{prefix}_micro_recall"] = best["micro_recall"]
                row[f"{prefix}_delta_micro_f1_vs_map"] = best["micro_f1"] - map_row.micro_f1
                row[f"{prefix}_temperature"] = best["temperature"]
                row[f"{prefix}_gamma"] = best["gamma"]
        output.append(row)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=ART / "raw_scores_chunk64")
    ap.add_argument("--out", type=Path, default=ART / "decode_r2d")
    ap.add_argument("--datasets", default="archiveii_embok_clean,bprna_new,bprna_ts0,chanrg_test,chanrg_gena,chanrg_genc,chanrg_genf,rfam_fam")
    ap.add_argument("--development-only", action="store_true")
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--frozen-config", type=Path, default=None,
                    help="reuse an accepted VL0 frozen_decode_config.json and evaluate external panels only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = [] if args.development_only else args.datasets.split(",")
    if args.frozen_config:
        frozen_source = json.loads(args.frozen_config.read_text())
        if frozen_source.get("development_dataset") != "bprna_vl0" or frozen_source.get("smoke_limit_per_dataset"):
            raise ValueError("external reuse requires the complete accepted VL0 frozen configuration")
        config = frozen_source["config"]
        dp_rows, rank_rows, entries, ranking_summary = [], [], [], []
    else:
        dp_rows, rank_rows, entries, config = tune_vl0(args.raw_root, args.limit_per_dataset)
        ranking_summary = make_ranking_summary(entries, rank_rows, "bprna_vl0")
    for dataset in datasets:
        if not (args.raw_root / dataset).exists():
            continue
        d, r, e = apply_external(args.raw_root, dataset, config, args.limit_per_dataset)
        dp_rows += d
        rank_rows += r
        entries += e
        ranking_summary += make_ranking_summary(entries, rank_rows, dataset, external=True, config=config)
    decode_summary = aggregate(dp_rows, ["dataset_id", "score_variant", "decoder", "temperature", "gamma"])
    routes = route_rows(decode_summary, ranking_summary)
    pd.DataFrame(dp_rows).to_parquet(args.out / "decode_per_sequence.parquet", index=False)
    pd.DataFrame(decode_summary).to_parquet(args.out / "decode_summary.parquet", index=False)
    pd.DataFrame(rank_rows).to_parquet(args.out / "ranking_per_sequence.parquet", index=False)
    pd.DataFrame(ranking_summary).to_parquet(args.out / "ranking_summary.parquet", index=False)
    frozen = {"schema_version": 1, "development_dataset": "bprna_vl0",
              "selection_policy": "micro F1, then macro F1, then micro precision; no external selection",
              "temperature_grid": TEMPERATURES, "mea_gamma_grid": GAMMAS,
              "external_datasets": datasets, "config": config,
              "reused_frozen_config": str(args.frozen_config) if args.frozen_config else None,
              "smoke_limit_per_dataset": args.limit_per_dataset or None}
    (args.out / "frozen_decode_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.out / "route_summary.json").write_text(json.dumps({"schema_version": 1, "rows": routes}, indent=2) + "\n")
    manifest = {"schema_version": 1, "official": not args.limit_per_dataset,
                "raw_root": str(args.raw_root), "out": str(args.out),
                "datasets": datasets if args.frozen_config else ["bprna_vl0", *datasets],
                "reused_frozen_config": str(args.frozen_config) if args.frozen_config else None,
                "n_decode_rows": len(dp_rows), "n_ranking_rows": len(rank_rows),
                "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__).resolve()),
                "decode_common_sha256": sha256(REPO / "tools/pair_decode_common.py"),
                "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "decode_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_decode_rows": len(dp_rows), "n_ranking_rows": len(rank_rows),
                      "routes": routes, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
