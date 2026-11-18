#!/usr/bin/env python3
"""Independent E0 verification using a separate tree builder and zss."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from zss import Node, simple_distance

from foldnar_e0 import load_predictions, load_truth, summarize_rows, verified_positional_predictions

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
INDEPENDENT_PAIR_SPAN_BINS = (("0_3", 0, 4), ("4_9", 4, 10), ("10_24", 10, 25), ("25_49", 25, 50),
                              ("50_99", 50, 100), ("100_199", 100, 200), ("200_plus", 200, None))


def independent_pair_span_confusion(pred, truth):
    output = {}
    for label, lower, upper in INDEPENDENT_PAIR_SPAN_BINS:
        selected = lambda pairs: sum(1 for i, j in pairs if j - i >= lower and (upper is None or j - i < upper))
        output[label] = {"tp": selected(pred & truth), "fp": selected(pred - truth), "fn": selected(truth - pred)}
    return output


def _valid_pairs(raw_pairs, length=None):
    pairs = set()
    occupied = set()
    for raw in raw_pairs:
        if len(raw) != 2:
            raise ValueError("malformed pair")
        a, b = sorted((int(raw[0]), int(raw[1])))
        if a == b or a < 0 or (length is not None and b >= length):
            raise ValueError("invalid coordinates")
        if (a, b) in pairs or a in occupied or b in occupied:
            raise ValueError("duplicate or multiply paired nucleotide")
        pairs.add((a, b))
        occupied.update((a, b))
    ordered = sorted(pairs)
    for index, (a, b) in enumerate(ordered):
        for c, d in ordered[index + 1:]:
            if c >= b:
                break
            if a < c < b < d:
                raise ValueError("crossing pairs")
    return pairs


def _prf_sets(pred, truth):
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    return _prf_counts(tp, fp, fn)


def _prf_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _prf_counters(pred, truth):
    return _prf_counts(sum((pred & truth).values()), sum((pred - truth).values()), sum((truth - pred).values()))


def independent_metrics(raw_pred, raw_truth, length=None):
    """Build maximal-stem trees without importing rnajepa.fold_structure."""
    pred, truth = _valid_pairs(raw_pred, length), _valid_pairs(raw_truth, length)

    def components(pset):
        # Maximal consecutive nested runs, represented outer pair first.
        stems = []
        for start in sorted(pset):
            if (start[0] - 1, start[1] + 1) in pset:
                continue
            run = []
            cursor = start
            while cursor in pset:
                run.append(cursor)
                cursor = (cursor[0] + 1, cursor[1] - 1)
            stems.append(tuple(run))
        if sum(len(s) for s in stems) != len(pset):
            raise AssertionError("independent stem partition mismatch")

        # Immediate pair parent from a nesting sweep over active intervals.
        parent_pair = {}
        active = []
        for pair in sorted(pset, key=lambda x: (x[0], -x[1])):
            while active and not (active[-1][0] < pair[0] and pair[1] < active[-1][1]):
                active.pop()
            parent_pair[pair] = active[-1] if active else None
            active.append(pair)
        pair_to_stem = {pair: idx for idx, stem in enumerate(stems) for pair in stem}
        children = {idx: [] for idx in range(len(stems))}
        roots = []
        for idx, stem in enumerate(stems):
            enclosing_pair = parent_pair[stem[0]]
            if enclosing_pair is None:
                roots.append(idx)
            else:
                children[pair_to_stem[enclosing_pair]].append(idx)
        for value in children.values():
            value.sort(key=lambda idx: stems[idx][0])
        roots.sort(key=lambda idx: stems[idx][0])

        loops = []
        signatures = {}

        def visit(idx):
            stem = stems[idx]
            child_ids = children[idx]
            child_pairs = tuple(stems[k][0] for k in child_ids)
            if not child_ids:
                kind = "HAIRPIN"
            elif len(child_ids) > 1:
                kind = "MULTILOOP"
            else:
                child = stems[child_ids[0]][0]
                left, right = child[0] - stem[0][0] - 1, stem[0][1] - child[1] - 1
                kind = "INTERNAL" if left > 0 and right > 0 else "BULGE"
            loops.append((stem[0], child_pairs, kind))
            node = Node(kind)
            child_signatures = []
            for child_id in child_ids:
                child_node, child_signature = visit(child_id)
                node.addkid(child_node)
                child_signatures.append(child_signature)
            signatures[idx] = (kind, tuple(child_signatures))
            return node, signatures[idx]

        root = Node("EXTERNAL")
        root_signatures = []
        for idx in roots:
            child_node, child_signature = visit(idx)
            root.addkid(child_node)
            root_signatures.append(child_signature)
        loops.append((None, tuple(stems[idx][0] for idx in roots), "EXTERNAL"))
        return stems, loops, root, ("EXTERNAL", tuple(root_signatures))

    pred_stems, pred_loops_raw, pred_root, pred_signature = components(pred)
    truth_stems, truth_loops_raw, truth_root, truth_signature = components(truth)
    # Loop identity includes all loop instances, including the external loop.
    pred_loop_counter = Counter((closing, children) for closing, children, kind in pred_loops_raw)
    truth_loop_counter = Counter((closing, children) for closing, children, kind in truth_loops_raw)
    pred_kind = {kind: Counter((closing, children) for closing, children, k in pred_loops_raw if k == kind)
                 for kind in ("EXTERNAL", "HAIRPIN", "INTERNAL", "BULGE", "MULTILOOP")}
    truth_kind = {kind: Counter((closing, children) for closing, children, k in truth_loops_raw if k == kind)
                  for kind in ("EXTERNAL", "HAIRPIN", "INTERNAL", "BULGE", "MULTILOOP")}
    pred_multi = pred_kind["MULTILOOP"]
    truth_multi = truth_kind["MULTILOOP"]
    pred_junction = Counter((closing, child) for (closing, children), count in pred_multi.items()
                            for child in children for _ in range(count))
    truth_junction = Counter((closing, child) for (closing, children), count in truth_multi.items()
                             for child in children for _ in range(count))
    pred_wiring = Counter((closing, child) for (closing, children), count in pred_loop_counter.items()
                          for child in children for _ in range(count))
    truth_wiring = Counter((closing, child) for (closing, children), count in truth_loop_counter.items()
                           for child in children for _ in range(count))
    ged = int(simple_distance(pred_root, truth_root))
    n_nodes = max(1 + len(pred_stems), 1 + len(truth_stems))
    return {
        "pair": _prf_sets(pred, truth),
        "stem": _prf_counters(Counter(pred_stems), Counter(truth_stems)),
        "loop": _prf_counters(pred_loop_counter, truth_loop_counter),
        "loop_types": {kind: _prf_counters(pred_kind[kind], truth_kind[kind]) for kind in pred_kind},
        "multiloop": _prf_counters(pred_multi, truth_multi),
        "junction": _prf_counters(pred_junction, truth_junction),
        "wiring": _prf_counters(pred_wiring, truth_wiring),
        "pair_span_confusion": independent_pair_span_confusion(pred, truth),
        "topology": {"ged": ged, "normalized_ged": ged / n_nodes, "f1": 1 - ged / n_nodes,
                     "exact": pred_signature == truth_signature, "n_pred_nodes": 1 + len(pred_stems),
                     "n_truth_nodes": 1 + len(truth_stems)},
        "n_pred_pairs": len(pred), "n_truth_pairs": len(truth),
        "n_pred_stems": len(pred_stems), "n_truth_stems": len(truth_stems),
        "pred_loop_counts": dict(Counter(kind for closing, children, kind in pred_loops_raw)),
        "truth_loop_counts": dict(Counter(kind for closing, children, kind in truth_loops_raw)),
    }


def _compare_values(original, recomputed, path, mismatches, max_delta):
    if isinstance(recomputed, dict):
        for key, value in recomputed.items():
            _compare_values(original.get(key), value, f"{path}.{key}", mismatches, max_delta)
    elif isinstance(recomputed, bool):
        if bool(original) != recomputed:
            mismatches.append({"path": path, "original": original, "recomputed": recomputed})
    elif isinstance(recomputed, (int, float)):
        if original is None or not math.isclose(float(original), float(recomputed), rel_tol=1e-10, abs_tol=1e-10):
            mismatches.append({"path": path, "original": original, "recomputed": recomputed})
        else:
            max_delta[0] = max(max_delta[0], abs(float(original) - float(recomputed)))
    elif original != recomputed:
        mismatches.append({"path": path, "original": original, "recomputed": recomputed})


def verify_task(spec, data_path, original_rows):
    truth, pred = load_truth(Path(data_path)), load_predictions(Path(spec["path"]))
    positional = verified_positional_predictions(truth, pred)
    aligned = positional[0] if positional else None
    mismatches, max_delta = [], [0.0]
    counts = Counter()
    metric_keys = ("pair", "stem", "loop", "loop_types", "multiloop", "junction", "wiring", "pair_span_confusion", "topology",
                   "n_pred_pairs", "n_truth_pairs", "n_pred_stems", "n_truth_stems")
    for row in original_rows:
        name = row["name"]
        counts["rows"] += 1
        pred_missing = name not in aligned if positional else len(pred.get(name, [])) != 1
        if len(truth.get(name, [])) != 1 or pred_missing:
            mismatches.append({"path": f"{spec['prediction_id']}:{name}", "error": "ambiguous raw identity"})
            continue
        t = truth[name][0]
        p = aligned[name] if positional else pred[name][0]
        expected_sha = hashlib.sha256(t["seq"].encode()).hexdigest()
        if row.get("sequence_sha256") != expected_sha:
            mismatches.append({"path": f"{spec['prediction_id']}:{name}.sequence_sha256",
                               "original": row.get("sequence_sha256"), "recomputed": expected_sha})
        try:
            recomputed = independent_metrics(p["pairs"], t["pairs"], len(t["seq"]))
        except Exception as exc:
            if row["status"] == "OK":
                mismatches.append({"path": f"{spec['prediction_id']}:{name}.status", "original": "OK", "recomputed": str(exc)})
            elif row.get("pair"):
                pred_set = {tuple(sorted(map(int, x))) for x in p["pairs"]}
                truth_set = {tuple(sorted(map(int, x))) for x in t["pairs"]}
                pair = _prf_sets(pred_set, truth_set)
                _compare_values(row["pair"], pair, f"{spec['prediction_id']}:{name}.pair_only", mismatches, max_delta)
                span = independent_pair_span_confusion(pred_set, truth_set)
                _compare_values(row.get("pair_span_confusion"), span, f"{spec['prediction_id']}:{name}.pair_span_only", mismatches, max_delta)
                counts["pair_only"] += 1
            else:
                counts["invalid"] += 1
        else:
            if row["status"] != "OK":
                mismatches.append({"path": f"{spec['prediction_id']}:{name}.status", "original": row["status"], "recomputed": "OK"})
                continue
            original_metric = {key: row[key] for key in metric_keys}
            recomputed_metric = {key: recomputed[key] for key in metric_keys}
            _compare_values(original_metric, recomputed_metric, f"{spec['prediction_id']}:{name}", mismatches, max_delta)
            counts["structure"] += 1
        if len(mismatches) > 100:
            break
    return {"prediction_id": spec["prediction_id"], "counts": dict(counts),
            "n_mismatches": len(mismatches), "mismatches": mismatches[:100], "max_abs_float_delta": max_delta[0]}


def _common_intersections(rows):
    by_split_model = {}
    for row in rows:
        if row["split"] not in {"bprna_new", "bprna_ts0"} or row["status"] != "OK" or row.get("repeat") is not None:
            continue
        by_split_model.setdefault((row["split"], row["model"]), {})[row["name"]] = row
    output = []
    for split in ("bprna_new", "bprna_ts0"):
        models = sorted(model for s, model in by_split_model if s == split)
        common = None
        for model in models:
            names = set(by_split_model[(split, model)])
            common = names if common is None else common & names
        common = sorted(common or [])
        run_rows = []
        for model in models:
            selected = [by_split_model[(split, model)][name] for name in common]
            spec = {"prediction_id": f"common/{split}/{model}", "model": model, "split": split, "repeat": None}
            summary = summarize_rows(selected, spec)
            run_rows.append({"model": model, "n_common_rows": len(selected), "macro": summary["macro"], "micro": summary["micro"]})
        output.append({"split": split, "models": models, "n_common_sequences": len(common), "runs": run_rows,
                       "filter": "exact name join; status=OK in every listed system; repeat=None"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-predictions", type=int, default=0)
    parser.add_argument("--out-phase", default="e0")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.out_phase
    target = out / "independent_verification.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite independent verification")
    registry = json.loads((args.repo / "manifests/foldnar_v1/prediction_registry.json").read_text())["rows"]
    data_registry = json.loads((args.repo / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]
    data_paths = {r["dataset_id"]: r["path"] for r in data_registry if r.get("role") == "test"}
    original_rows = json.loads((out / "per_sequence_error_decomposition.json").read_text())["rows"]
    grouped = {}
    for row in original_rows:
        grouped.setdefault(row["prediction_id"], []).append(row)
    specs = [r for r in registry if r["prediction_id"] in grouped]
    if args.limit_predictions:
        specs = specs[:args.limit_predictions]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify_task, spec, data_paths[spec["split"]], grouped[spec["prediction_id"]]): spec for spec in specs}
        for index, future in enumerate(as_completed(futures), 1):
            report = future.result()
            reports.append(report)
            print(f"VERIFY {index}/{len(specs)} {report['prediction_id']} mismatches={report['n_mismatches']}", flush=True)
    reports.sort(key=lambda x: x["prediction_id"])
    mismatches = sum(r["n_mismatches"] for r in reports)
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "separate maximal-stem/tree implementation; zss.simple_distance; all frozen E0 rows",
        "n_prediction_runs": len(reports), "n_rows": sum(r["counts"].get("rows", 0) for r in reports),
        "n_structure_rows": sum(r["counts"].get("structure", 0) for r in reports),
        "n_pair_only_rows": sum(r["counts"].get("pair_only", 0) for r in reports),
        "n_mismatches": mismatches, "max_abs_float_delta": max((r["max_abs_float_delta"] for r in reports), default=0),
        "status": "PASS" if mismatches == 0 and len(reports) == len(grouped) else "FAIL",
        "runs": reports, "common_system_intersections": _common_intersections(original_rows),
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "n_prediction_runs", "n_rows", "n_mismatches", "max_abs_float_delta")}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
