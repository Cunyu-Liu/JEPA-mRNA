#!/usr/bin/env python3
"""Run the pre-registered E0 local-to-global error decomposition."""
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

import numpy as np
import pandas as pd

from rnajepa.fold_structure import StructureError, loop_diagram, pair_prf, structure_metrics

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
MAIN_SPLITS = {"archiveii_embok_clean", "bprna_new", "bprna_ts0", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_truth(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("name", row.get("id", f"row:{line_number}")))
            out.setdefault(name, []).append({
                "name": name, "seq": str(row.get("seq", row.get("sequence", ""))).upper().replace("T", "U"),
                "pairs": row.get("pairs", []), "family": row.get("family") or row.get("rfam_id") or row.get("rfam"),
                "clan": row.get("clan"), "architecture": row.get("architecture"),
            })
    return out


def load_predictions(path: Path) -> dict[str, list[dict]]:
    obj = json.loads(path.read_text())
    rows = obj.get("per_sequence", obj.get("predictions"))
    if not isinstance(rows, list):
        raise ValueError(f"no per-sequence predictions in {path}")
    out: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        name = str(row.get("name", row.get("id", f"row:{index}")))
        pairs = row.get("pred_pairs", row.get("pairs", []))
        out.setdefault(name, []).append({
            "name": name, "pairs": pairs, "length": row.get("length"),
            "n_gt": row.get("n_gt", row.get("n_gt_pairs")), "f1": row.get("f1"),
        })
    return out


def verified_positional_predictions(truth: dict[str, list[dict]], pred: dict[str, list[dict]]) -> tuple[dict[str, dict], dict] | None:
    """Recover legacy blank-name predictions only after row-wise identity checks."""
    if set(pred) != {""}:
        return None
    predictions = pred[""]
    if any(len(rows) != 1 for rows in truth.values()) or len(predictions) != len(truth):
        raise ValueError("blank prediction names cannot be aligned: duplicate truth IDs or record-count mismatch")
    truth_rows = [rows[0] for rows in truth.values()]
    aligned = {}
    max_f1_delta = 0.0
    n_f1_checked = 0
    for index, (t, p) in enumerate(zip(truth_rows, predictions)):
        if p["length"] is not None and int(p["length"]) != len(t["seq"]):
            raise ValueError(f"blank-name positional length mismatch at row {index}")
        if p["n_gt"] is not None and int(p["n_gt"]) != len(t["pairs"]):
            raise ValueError(f"blank-name positional n_gt mismatch at row {index}")
        if p["f1"] is not None:
            recomputed = pair_prf(p["pairs"], t["pairs"])["f1"]
            delta = abs(float(p["f1"]) - recomputed)
            max_f1_delta = max(max_f1_delta, delta)
            n_f1_checked += 1
            if delta > 1e-8:
                raise ValueError(f"blank-name positional F1 mismatch at row {index}: {delta}")
        aligned[t["name"]] = p
    return aligned, {"method": "positional_verified_by_length_n_gt_and_f1", "n_rows": len(aligned),
                     "n_f1_checked": n_f1_checked, "max_f1_delta": max_f1_delta}


def evaluate_join(pred_spec: dict, data_path: str) -> dict:
    truth = load_truth(Path(data_path))
    pred = load_predictions(Path(pred_spec["path"]))
    truth_ids, pred_ids = set(truth), set(pred)
    positional = verified_positional_predictions(truth, pred)
    join_metadata = positional[1] if positional else {"method": "exact_name"}
    pred_by_name = positional[0] if positional else None
    joined = sorted(truth_ids & set(pred_by_name)) if positional else sorted(truth_ids & pred_ids)
    rows = []
    candidates = []
    counts = Counter()
    for name in joined:
        if len(truth[name]) != 1 or (positional is None and len(pred[name]) != 1):
            counts["ambiguous_id"] += 1
            continue
        t = truth[name][0]
        p = pred_by_name[name] if positional else pred[name][0]
        base = {
            "prediction_id": pred_spec["prediction_id"], "model": pred_spec["model"], "split": pred_spec["split"],
            "repeat": pred_spec.get("repeat"), "name": name, "length": len(t["seq"]), "family": t["family"],
            "clan": t["clan"], "architecture": t["architecture"], "status": "OK", "error": None,
            "join_method": join_metadata["method"],
        }
        try:
            metrics = structure_metrics(p["pairs"], t["pairs"], len(t["seq"]))
            base["pair"] = metrics["pair"]
            base["stem"] = metrics["stem"]
            base["loop"] = metrics["loop"]
            base["loop_types"] = metrics["loop_types"]
            base["multiloop"] = metrics["multiloop"]
            base["junction"] = metrics["junction"]
            base["topology"] = metrics["topology"]
            base["n_pred_pairs"] = metrics["n_pred_pairs"]
            base["n_truth_pairs"] = metrics["n_truth_pairs"]
            base["n_pred_stems"] = metrics["n_pred_stems"]
            base["n_truth_stems"] = metrics["n_truth_stems"]
            base["pair_f1_minus_topology_f1"] = metrics["pair"]["f1"] - metrics["topology"]["f1"]
            base["dominant_pair_error"] = "FN" if metrics["pair"]["fn"] > metrics["pair"]["fp"] else "FP" if metrics["pair"]["fp"] > metrics["pair"]["fn"] else "BALANCED"
            base["dominant_stem_error"] = "FN" if metrics["stem"]["fn"] > metrics["stem"]["fp"] else "FP" if metrics["stem"]["fp"] > metrics["stem"]["fn"] else "BALANCED"
            severity = max(base["pair_f1_minus_topology_f1"], 1 - metrics["topology"]["f1"], 1 - metrics["stem"]["f1"])
            candidates.append({"severity": severity, "split": pred_spec["split"], "model": pred_spec["model"],
                               "prediction_id": pred_spec["prediction_id"], "name": name, "metrics": metrics,
                               "sequence": t["seq"], "truth_pairs": t["pairs"], "pred_pairs": p["pairs"],
                               "truth_diagram": loop_diagram(t["pairs"], len(t["seq"])),
                               "pred_diagram": loop_diagram(p["pairs"], len(t["seq"]))})
            counts["ok"] += 1
        except (StructureError, ValueError) as exc:
            base["status"] = "STRUCTURE_UNDEFINED"
            base["error"] = str(exc)
            try:
                base["pair"] = pair_prf(p["pairs"], t["pairs"])
                base["pair_f1_minus_topology_f1"] = None
                counts["pair_only"] += 1
            except Exception as pair_exc:
                base["status"] = "INVALID_PAIR_INPUT"
                base["error"] = f"{exc}; pair metric: {pair_exc}"
                counts["invalid"] += 1
        rows.append(base)
    summary = summarize_rows(rows, pred_spec)
    joined_pred_ids = set(pred_by_name) if positional else pred_ids
    n_prediction_records = sum(len(value) for value in pred.values())
    summary.update({
        "n_truth_unique_names": len(truth_ids), "n_prediction_unique_names": len(pred_ids),
        "n_prediction_records": n_prediction_records,
        "n_joined_names": len(joined), "n_truth_only": len(truth_ids - joined_pred_ids),
        "n_prediction_only": (n_prediction_records - len(joined)) if positional else len(pred_ids - truth_ids),
        "coverage_of_truth": len(joined) / len(truth_ids) if truth_ids else 0,
        "coverage_of_prediction": len(joined) / n_prediction_records if positional and n_prediction_records else len(joined) / len(pred_ids) if pred_ids else 0,
        "join": join_metadata,
        "join_status_counts": dict(counts),
    })
    candidates.sort(key=lambda x: (-x["severity"], x["name"]))
    return {"summary": summary, "rows": rows, "candidates": candidates[:12]}


def _mean(rows: list[dict], section: str, field: str) -> float | None:
    values = [row[section][field] for row in rows if row.get(section) and row[section].get(field) is not None]
    return float(np.mean(values)) if values else None


def _micro(rows: list[dict], section: str) -> dict | None:
    selected = [row[section] for row in rows if row.get(section)]
    if not selected:
        return None
    tp, fp, fn = (sum(int(x[key]) for x in selected) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def summarize_rows(rows: list[dict], pred_spec: dict) -> dict:
    valid = [r for r in rows if r["status"] == "OK"]
    pair_rows = [r for r in rows if r.get("pair")]
    macro = {
        "pair_f1": _mean(pair_rows, "pair", "f1"),
        "pair_precision": _mean(pair_rows, "pair", "precision"),
        "pair_recall": _mean(pair_rows, "pair", "recall"),
        "stem_f1": _mean(valid, "stem", "f1"),
        "loop_f1": _mean(valid, "loop", "f1"),
        "multiloop_f1": _mean(valid, "multiloop", "f1"),
        "junction_f1": _mean(valid, "junction", "f1"),
        "topology_f1": _mean(valid, "topology", "f1"),
        "topology_exact": _mean(valid, "topology", "exact"),
        "topology_ged": _mean(valid, "topology", "ged"),
        "topology_normalized_ged": _mean(valid, "topology", "normalized_ged"),
        "pair_f1_minus_topology_f1": _mean(valid, "pair", "f1") - _mean(valid, "topology", "f1") if valid else None,
    }
    for kind in ("external", "hairpin", "internal", "bulge", "multiloop"):
        values = [r["loop_types"][kind.upper()]["f1"] for r in valid]
        macro[f"{kind}_loop_f1"] = float(np.mean(values)) if values else None
    return {
        "prediction_id": pred_spec["prediction_id"], "model": pred_spec["model"], "split": pred_spec["split"],
        "repeat": pred_spec.get("repeat"), "n_rows": len(rows), "n_structure_valid": len(valid),
        "structure_valid_fraction": len(valid) / len(rows) if rows else 0,
        "macro": macro,
        "micro": {section: _micro(valid if section != "pair" else pair_rows, section)
                  for section in ("pair", "stem", "loop", "multiloop", "junction")},
        "dominant_pair_errors": dict(Counter(r.get("dominant_pair_error") for r in valid)),
        "dominant_stem_errors": dict(Counter(r.get("dominant_stem_error") for r in valid)),
    }


def flatten_row(row: dict) -> dict:
    flat = {k: v for k, v in row.items() if k not in {"pair", "stem", "loop", "loop_types", "multiloop", "junction", "topology"}}
    for section in ("pair", "stem", "loop", "multiloop", "junction"):
        for field in ("tp", "fp", "fn", "precision", "recall", "f1"):
            flat[f"{section}_{field}"] = row.get(section, {}).get(field)
    for kind in ("EXTERNAL", "HAIRPIN", "INTERNAL", "BULGE", "MULTILOOP"):
        for field in ("tp", "fp", "fn", "f1"):
            flat[f"loop_{kind.lower()}_{field}"] = row.get("loop_types", {}).get(kind, {}).get(field)
    for field in ("ged", "normalized_ged", "f1", "exact", "n_pred_nodes", "n_truth_nodes"):
        flat[f"topology_{field}"] = row.get("topology", {}).get(field)
    return flat


def decide_gate(runs: list[dict], leakage_path: Path) -> dict:
    primary = [r for r in runs if r["model"] == "xens2" and (r.get("repeat") in (None, 0))]
    split_triggers = []
    for run in primary:
        m = run["macro"]
        if m["topology_f1"] is None:
            continue
        conditions = {
            "topology_f1_lt_0.85": m["topology_f1"] < 0.85,
            "stem_f1_lt_0.80": m["stem_f1"] < 0.80,
            "pair_minus_topology_gte_0.05": m["pair_f1_minus_topology_f1"] >= 0.05,
        }
        rule1 = any(conditions.values())
        junction_values = [x for x in (m["multiloop_f1"], m["junction_f1"]) if x is not None]
        rule2 = (m["pair_f1_minus_topology_f1"] >= 0.05 and bool(junction_values) and
                 min(junction_values) < 0.75)
        if rule1 or rule2:
            split_triggers.append({"split": run["split"], "rule1_any_of_three": rule1, "rule2_gap_and_junction": rule2,
                                   "conditions": conditions, "macro": m, "coverage_of_truth": run["coverage_of_truth"]})
    distinct_rule1 = sum(x["rule1_any_of_three"] for x in split_triggers)
    any_rule2 = any(x["rule2_gap_and_junction"] for x in split_triggers)
    by_split_model = {(r["split"], r["model"]): r for r in runs}
    comparison_splits = ["bprna_new", "bprna_ts0"]
    systems = []
    for model in sorted({r["model"] for r in runs}):
        if all((split, model) in by_split_model and by_split_model[(split, model)]["coverage_of_truth"] >= 0.95
               and by_split_model[(split, model)]["n_structure_valid"] > 0 for split in comparison_splits):
            systems.append(model)
    evidence_sufficient = len(systems) >= 2 and "xens2" in systems
    e0_trigger = distinct_rule1 >= 2 or any_rule2
    if leakage_path.exists():
        leakage = json.loads(leakage_path.read_text())
        leakage_status = leakage.get("overall_status", "UNKNOWN")
        execution = "HOLD_FOR_LEAKAGE_UNKNOWN" if "BLOCKED" in leakage_status else "AUTHORIZED_BY_E0_AND_PHASE1"
        leakage_evidence = str(leakage_path)
    else:
        leakage_status = "PENDING"
        execution = "PENDING_PHASE1_LEAKAGE_AUDIT"
        leakage_evidence = None
    return {
        "schema_version": 1,
        "primary_system": "xens2",
        "evidence": {"comparison_splits": comparison_splits, "independent_systems_with_complete_raw_coverage": systems,
                     "sufficient": evidence_sufficient,
                     "requirement": ">=2 independent raw-prediction systems; bpRNA-new plus another main split; same joins and filters reported"},
        "trigger": {"distinct_rule1_splits": distinct_rule1, "any_rule2": any_rule2, "e0_trigger": e0_trigger,
                    "split_details": split_triggers},
        "e0_decision": "GO_E1" if evidence_sufficient and e0_trigger else "NO_GO" if evidence_sufficient else "BLOCKED_INSUFFICIENT_EVIDENCE",
        "phase1_leakage_status": leakage_status,
        "phase1_leakage_evidence": leakage_evidence,
        "execution_decision": execution if evidence_sufficient and e0_trigger else "DO_NOT_TRAIN",
        "alternative_causes": {
            "label_noise": "NOT_ESTABLISHED_BY_E0",
            "pretraining_exposure": leakage_status,
            "train_topology_coverage": "DESCRIPTIVE_ONLY; shared simple topology alone does not negate a local-to-global error",
        },
        "interpretation": "GO_E1 is a scientific trigger, not authorization to ignore blocked leakage/pretraining layers.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--out-phase", default="e0")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-predictions", type=int, default=0, help="Development smoke test only; not used for frozen E0")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.out_phase
    final_names = ("error_decomposition.parquet", "per_sequence_error_decomposition.json", "e0_error_decomposition.json", "e0_gate_decision.json")
    if any((out / name).exists() for name in final_names):
        raise FileExistsError("refusing to overwrite frozen E0 outputs")
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((args.repo / "manifests/foldnar_v1/prediction_registry.json").read_text())["rows"]
    data_registry = json.loads((args.repo / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]
    data_paths = {r["dataset_id"]: r["path"] for r in data_registry if r.get("role") == "test"}
    predictions = [r for r in registry if r["status"] == "FROZEN" and r["split"] in data_paths]
    if args.limit_predictions:
        predictions = predictions[:args.limit_predictions]
    print(f"E0 tasks={len(predictions)} workers={args.workers}", flush=True)
    runs, rows, candidates = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_join, spec, data_paths[spec["split"]]): spec for spec in predictions}
        for index, future in enumerate(as_completed(futures), 1):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"E0 failed for {spec['prediction_id']}") from exc
            runs.append(result["summary"])
            rows.extend(result["rows"])
            candidates.extend(result["candidates"])
            print(f"E0 {index}/{len(predictions)} {spec['prediction_id']} rows={len(result['rows'])}", flush=True)
    runs.sort(key=lambda r: r["prediction_id"])
    rows.sort(key=lambda r: (r["prediction_id"], r["name"]))
    pd.DataFrame([flatten_row(r) for r in rows]).to_parquet(out / "error_decomposition.parquet", index=False)
    per_sequence = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "coordinate_system": "zero-based inclusive (i,j), i<j", "n_rows": len(rows), "rows": rows}
    (out / "per_sequence_error_decomposition.json").write_text(json.dumps(per_sequence, ensure_ascii=False, separators=(",", ":")) + "\n")

    # One difficult case per unique (split, sequence), with the most severe model representation.
    best_by_sequence = {}
    for case in candidates:
        key = (case["split"], case["name"])
        if key not in best_by_sequence or case["severity"] > best_by_sequence[key]["severity"]:
            best_by_sequence[key] = case
    difficult = sorted(best_by_sequence.values(), key=lambda x: (-x["severity"], x["split"], x["name"]))[:30]
    for rank, case in enumerate(difficult, 1):
        case["case_rank"] = rank
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "code_sha256": sha256(Path(__file__)), "structure_code_sha256": sha256(args.repo / "src/rnajepa/fold_structure.py"),
               "prediction_registry_sha256": sha256(args.repo / "manifests/foldnar_v1/prediction_registry.json"),
               "n_prediction_runs": len(runs), "n_per_sequence_rows": len(rows), "runs": runs,
               "difficult_cases": difficult, "n_difficult_cases": len(difficult)}
    (out / "e0_error_decomposition.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    gate = decide_gate(runs, args.artifact_root / "artifacts/foldnar_v1/phase1/leakage_matrix.json")
    gate["code_sha256"] = payload["code_sha256"]
    gate["e0_summary_sha256"] = sha256(out / "e0_error_decomposition.json")
    (out / "e0_gate_decision.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "runs": len(runs), "rows": len(rows), "difficult_cases": len(difficult),
                      "e0_decision": gate["e0_decision"], "execution_decision": gate["execution_decision"],
                      "out": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
