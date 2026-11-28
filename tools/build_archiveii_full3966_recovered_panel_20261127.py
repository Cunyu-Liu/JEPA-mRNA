#!/usr/bin/env python3
"""Merge validated CPU recovery checkpoints and rescore ArchiveII dual panels."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/archiveii_official_reference_20261127"
SOURCE = ROOT / "source_data/archiveii_full3966_cpu_recovery_20261127"
OUT = ROOT / "artifacts/archiveii_full3966_cpu_recovery_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-archiveii-full3966-recovered-panel.md"
WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_full3966_cpu_20261127")
VIENNA_CLEAN_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_vienna_clean_cpu_20261127")
XENS2_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_queue_20261127/production/xens2")
XENS2_EVIDENCE_PATH = ROOT / "source_data/archiveii_vienna_full3966_cpu_20261127/xens2_queue_evidence.json"
XENS2_LIVE_STATUS_PATH = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_queue_20261127/queue_status.json")
VIENNA_PLAN_PATH = ROOT / "source_data/archiveii_vienna_full3966_cpu_20261127/clean_recovery_plan.json"
FULL_PANEL = ROOT / "source_data/archiveii_official_reference_20261127/official_reference_3966.jsonl.gz"
CLEAN_PANEL = ROOT / "source_data/archiveii_official_reference_20261127/official_reference_clean_2544.jsonl.gz"
FULL_ID = "archiveii_official_reference_3966"
CLEAN_ID = "archiveii_official_reference_clean_2544"
XENS2 = "xens2/archiveii_embok_clean"
BASELINES = ("contrafold", "eternafold", "mxfold2", "vienna_mfe", "vienna_centroid", "vienna_mea",
             "rnastructure_fold", "rnastructure_maxexpect", "xens2")
SCHEMA = "archiveii.full3966.cpu_recovery.panel.v1"
SEED = 20261127


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical(row) + "\n")
    os.replace(temporary, path)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_set(values, length):
    result, occupied = set(), set()
    for value in values:
        i, j = map(int, value)
        if not 0 <= i < j < length or i in occupied or j in occupied:
            raise ValueError(f"invalid pair {(i, j)} for length {length}")
        occupied.update((i, j))
        result.add((i, j))
    return result


def prf(truth, predicted):
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    if not truth and not predicted:
        return tp, fp, fn, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1


def score_present(panel, predictions):
    totals = [0, 0, 0]
    values, records, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in predictions:
            missing.append(name)
            continue
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(predictions[name]["pairs"], len(label["seq"]))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        a, b, c, p, r, f = prf(truth, predicted)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        values.append((p, r, f))
        records[name] = {"tp": a, "fp": b, "fn": c, "precision": p, "recall": r, "f1": f,
                         "truth_empty": not truth, "prediction_empty": not predicted,
                         "empty_empty": not truth and not predicted}
    n = len(values)
    tp, fp, fn = totals
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
               "truth_empty_records": sum(row["truth_empty"] for row in records.values()),
               "empty_empty_records": sum(row["empty_empty"] for row in records.values())}
    extra = sorted(set(predictions) - {row["name"] for row in panel})
    return metrics, records, missing, extra


def bootstrap(deltas):
    rng = np.random.default_rng(SEED)
    array = np.asarray(deltas)
    values = []
    for start in range(0, 10000, 200):
        size = min(200, 10000 - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        values.extend(array[indices].mean(axis=1).tolist())
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def original_predictions(baseline, rows):
    if baseline.startswith("vienna_"):
        coverage_path = VIENNA_CLEAN_WORK / baseline / "coverage.json"
        if not coverage_path.is_file():
            return {}
        coverage = json.loads(coverage_path.read_text())
        prediction_path = VIENNA_CLEAN_WORK / baseline / "predictions.jsonl.gz"
        if coverage["status"] != "COMPLETE" or coverage["n_pass"] != 2544 or sha256(prediction_path) != coverage["predictions_sha256"]:
            return {}
        return {row["name"]: {"pairs": row["pairs"], "origin": "CPU_CLEAN_2544_RECOVERY",
                              "source_prediction_id": f"{baseline}/archiveii_clean_2544/cpu_recovery_v1",
                              "source_sha256": coverage["predictions_sha256"]} for row in read_gzip(prediction_path)}
    prediction_id = baseline + "/archiveii_embok_clean"
    source = next(row for row in rows if row["prediction_id"] == prediction_id)
    document = json.loads(Path(source["path"]).read_text())
    if baseline.startswith("rnastructure_fold"):
        records = document["results"]["fold"]["per_sequence"]
    elif baseline.startswith("rnastructure_maxexpect"):
        records = document["results"]["maxexpect"]["per_sequence"]
    else:
        records = document.get("per_sequence", document.get("predictions"))
    if records is None:
        raise ValueError(f"missing clean predictions {baseline}")
    output = {}
    for row in records:
        pairs = row.get("pred_pairs", row.get("pairs"))
        output[row["name"]] = {"pairs": pairs, "origin": "FROZEN_CLEAN_2544",
                               "source_prediction_id": prediction_id, "source_sha256": source["sha256"]}
    if len(output) != 2544:
        raise ValueError(f"clean prediction population {baseline}")
    return output


def recovery_predictions(baseline, plan_rows):
    work_root = XENS2_WORK if baseline == "xens2" else WORK / baseline
    coverage_path = work_root / "coverage.json"
    if not coverage_path.is_file():
        return {}, {"status": "NOT_STARTED", "n_completed": 0, "n_pass": 0, "remaining_count": len(plan_rows),
                    "remaining_names": [row["name"] for row in plan_rows], "coverage": 0.0}
    coverage = json.loads(coverage_path.read_text())
    prediction_path = work_root / "predictions.jsonl.gz"
    if sha256(prediction_path) != coverage["predictions_sha256"]:
        raise ValueError(f"recovery coverage SHA mismatch {baseline}")
    records = read_gzip(prediction_path)
    if len(records) != coverage["n_completed"]:
        raise ValueError(f"recovery coverage count mismatch {baseline}")
    expected_names = {row["name"] for row in plan_rows}
    output = {}
    for record in records:
        name = record["name"]
        if name not in expected_names or name in output:
            raise ValueError(f"unexpected/duplicate recovery ID {name}")
        plan = next(row for row in plan_rows if row["name"] == name)
        if record["sequence_sha256"] != plan["sequence_sha256"]:
            raise ValueError(f"recovery sequence SHA mismatch {name}")
        if record["status"] != "PASS":
            continue
        if set(plan["seq"]) - set("ACGU"):
            raise ValueError(f"invalid recovery status {name}")
        pairs = sorted(pair_set(record["pairs"], len(plan["seq"])))
        if record["pair_set_sha256"] != value_sha(pairs):
            raise ValueError(f"recovery pair SHA mismatch {name}")
        output[name] = {"pairs": pairs, "origin": "XENS2_GPU_RECOVERY_1422" if baseline == "xens2" else "CPU_RECOVERY_1422",
                        "checkpoint_sha256": sha256(work_root / "records" / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".json"))}
    if len(output) != coverage["n_pass"]:
        raise ValueError(f"recovery PASS accounting mismatch {baseline}")
    return output, coverage


def recovery_id(baseline):
    if baseline == "xens2":
        return "xens2/archiveii_official_reference_3966/gpu_recovery_v1"
    return f"{baseline}/archiveii_official_reference_3966/cpu_recovery_v1"


def public_strongest(row, panel_id):
    result = row["panels"][panel_id]
    return {"prediction_id": row["prediction_id"], "model_form": row["model_form"],
            "metrics": result["metrics"], "delta_vs_xens2": result["delta_vs_xens2"],
            "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION"}


def main():
    plan = json.loads((SOURCE / "recovery_plan.json").read_text())
    full = read_gzip(FULL_PANEL)
    clean = read_gzip(CLEAN_PANEL)
    base_document = json.loads((BASE / "system_metrics.json").read_text())
    base_rows = base_document["rows"]
    new_rows, metric_rows, recovery_summary = [], [], {}
    project_f1 = {}
    for baseline in BASELINES:
        prediction_id = recovery_id(baseline)
        recovered, coverage = recovery_predictions(baseline, plan["rows"])
        if baseline == "xens2" and coverage["status"] == "NOT_STARTED":
            recovery_summary[baseline] = {**coverage, "checkpoint_root": str(XENS2_WORK),
                                          "predictor_metadata_sha256": None, "pause": None,
                                          "queue_evidence": json.loads((XENS2_EVIDENCE_PATH).read_text()),
                                          "full_composite_records": 2544, "complete_for_full_ranking": False}
            continue
        original = original_predictions(baseline, base_rows)
        if original and set(original) != {row["name"] for row in clean}:
            raise ValueError(f"original clean ID mismatch {baseline}")
        composite = {**original, **recovered}
        for name, prediction in composite.items():
            if name in original and name in recovered:
                raise ValueError(f"duplicate composite origin {name}")
        complete = coverage["status"] == "COMPLETE" and coverage["n_pass"] == 1422 and len(recovered) == 1422
        if coverage["n_completed"] != len(recovered) and coverage.get("n_rejected_non_acgu", coverage.get("n_blocked", 0)) == 0:
            raise ValueError(f"coverage accounting mismatch {baseline}")
        work_root = XENS2_WORK if baseline == "xens2" else WORK / baseline
        pause_path = work_root / "PAUSED.json"
        clean_recovery_path = VIENNA_CLEAN_WORK / baseline / "coverage.json"
        clean_recovery = json.loads(clean_recovery_path.read_text()) if baseline.startswith("vienna_") and clean_recovery_path.is_file() else None
        recovery_summary[baseline] = {**coverage, "checkpoint_root": str(work_root),
                                      "predictor_metadata_sha256": sha256(work_root / "predictor_metadata.json") if (work_root / "predictor_metadata.json").is_file() else None,
                                      "pause": None if complete else (json.loads(pause_path.read_text()) if pause_path.is_file() else None),
                                      "clean_recovery": clean_recovery,
                                      "full_composite_records": len(composite), "complete_for_full_ranking": complete}
        write_gzip_jsonl(OUT / "predictions" / f"{baseline}.jsonl.gz", [
            {"name": row["name"], "sequence_sha256": row["sequence_sha256"], **composite[row["name"]]}
            for row in full if row["name"] in composite])
        row = {"prediction_id": prediction_id,
               "source_group": "XENS2_GPU_RECOVERY_COMPOSITE" if baseline == "xens2" else "CPU_RECOVERY_COMPOSITE",
               "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET",
               "path": str((OUT / "predictions" / f"{baseline}.jsonl.gz").relative_to(ROOT)),
               "available": True, "model_form": "three_checkpoint_ensemble" if baseline == "xens2" else "deterministic_single_predictor",
               "prediction_origin": "FROZEN_CLEAN_2544_PLUS_XENS2_GPU_RECOVERY_1422" if baseline == "xens2" else
                                    "CLEAN_2544_AND_OFFICIAL_1422_CPU_RECOVERY" if baseline.startswith("vienna_") else
                                    "FROZEN_CLEAN_2544_PLUS_CPU_RECOVERY_1422",
               "n_prediction_records": len(composite), "panels": {},
               "recovery_complete_1422": complete}
        if not composite:
            for panel_id, panel in ((FULL_ID, full), (CLEAN_ID, clean)):
                row["panels"][panel_id] = {"status": "BLOCKED_NO_RECOVERY_PREDICTIONS",
                    "n_panel_sequences": len(panel), "n_with_prediction": 0, "coverage": 0.0,
                    "metrics": None, "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None}
            new_rows.append(row)
            continue
        for panel_id, panel in ((FULL_ID, full), (CLEAN_ID, clean)):
            metrics, records, missing, extra = score_present(panel, composite)
            if metrics["n_scored"] == 0:
                row["panels"][panel_id] = {"status": "BLOCKED_NO_PREDICTIONS_IN_PANEL",
                    "n_panel_sequences": len(panel), "n_with_prediction": 0, "coverage": 0.0,
                    "missing_count": len(missing), "missing_names": missing,
                    "extra_count": len(extra), "extra_names": extra, "metrics": None,
                    "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None,
                    "missing_predictions_imputed_as_zero": False}
                continue
            row["panels"][panel_id] = {"status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing), "missing_names": missing,
                "extra_count": len(extra), "extra_names": extra, "metrics": metrics,
                "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None,
                "missing_predictions_imputed_as_zero": False}
            for name, values in records.items():
                metric_rows.append({"panel_id": panel_id, "prediction_id": prediction_id, "name": name, **values})
            if panel_id == CLEAN_ID and not missing:
                project_f1[prediction_id] = {name: values["f1"] for name, values in records.items()}
        new_rows.append(row)
    names = [row["name"] for row in clean]
    xens2_row = next(row for row in base_rows if row["prediction_id"] == XENS2)
    xens2_metrics = xens2_row["panels"][CLEAN_ID]["metrics"]
    xens2_per_record = read_gzip(BASE / "per_record_metrics.jsonl.gz")
    xens2_f1 = {row["name"]: row["f1"] for row in xens2_per_record
                if row["panel_id"] == CLEAN_ID and row["prediction_id"] == XENS2}
    reference_values = np.asarray([xens2_f1[name] for name in names])
    for row in new_rows:
        result = row["panels"][CLEAN_ID]
        if result["status"] != "PASS_FULL_COVERAGE":
            continue
        metrics = result["metrics"]
        result["delta_vs_xens2"] = {key: metrics[key] - xens2_metrics[key] for key in (
            "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        values = np.asarray([project_f1[row["prediction_id"]][name] for name in names])
        deltas = values - reference_values
        result["paired_macro_f1_delta_ci95_vs_xens2"] = {"observed_delta": float(deltas.mean()),
            "ci95": bootstrap(deltas), "method": "paired sequence-level percentile bootstrap",
            "n_bootstrap": 10000, "seed": SEED, "unit": "sequence",
            "applicability": "full common clean 2544 coverage only"}
    all_rows = base_rows + new_rows
    clean_eligible = [row for row in all_rows if row["panels"][CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"]
    full_eligible = [row for row in all_rows if row["panels"][FULL_ID]["status"] == "PASS_FULL_COVERAGE"]
    strongest = {
        "clean_2544": {
            "strongest_overall": public_strongest(max(clean_eligible, key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"]), CLEAN_ID),
            "strongest_ensemble": public_strongest(max((row for row in clean_eligible if "ensemble" in row["model_form"]), key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"]), CLEAN_ID),
            "strongest_single_predictor": public_strongest(max((row for row in clean_eligible if "ensemble" not in row["model_form"]), key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"]), CLEAN_ID),
        },
        "full_3966": {
            "strongest_overall": public_strongest(max(full_eligible, key=lambda row: row["panels"][FULL_ID]["metrics"]["micro_f1"]), FULL_ID),
            "strongest_ensemble": public_strongest(max((row for row in full_eligible if "ensemble" in row["model_form"]), key=lambda row: row["panels"][FULL_ID]["metrics"]["micro_f1"]), FULL_ID),
            "strongest_single_predictor": public_strongest(max((row for row in full_eligible if "ensemble" not in row["model_form"]), key=lambda row: row["panels"][FULL_ID]["metrics"]["micro_f1"]), FULL_ID) if any("ensemble" not in row["model_form"] for row in full_eligible) else "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR",
            "xens2_delta": "UNCHECKED_XENS2_NOT_FULL_COVERAGE",
        },
    }
    write_gzip_jsonl(OUT / "recovered_per_record_metrics.jsonl.gz", metric_rows)
    xens2_gpu_recovery_used = recovery_summary["xens2"]["n_pass"] > 0
    document = {"schema_version": SCHEMA, "scope": "deterministic checkpoint merge and frozen-prediction rescoring; CPU baselines plus gated xens2 GPU recovery; no model selection",
        "base_system_metrics": {"path": str((BASE / "system_metrics.json").relative_to(ROOT)), "sha256": sha256(BASE / "system_metrics.json")},
        "label_panels": base_document["label_panels"], "metric_contract": base_document["metric_contract"],
        "recovery_plan_sha256": sha256(SOURCE / "recovery_plan.json"), "recovery": recovery_summary,
        "coverage_accounting": {"base_identities": len(base_rows), "recovery_identities": len(new_rows),
            "total_identities": len(all_rows), "recovery_complete_1422": sum(row["recovery_complete_1422"] for row in new_rows),
            "clean_2544_full_coverage": len(clean_eligible), "official_3966_full_coverage": len(full_eligible),
            "xens2_official_3966_coverage": max(row["panels"][FULL_ID]["coverage"] for row in all_rows if row["prediction_id"].startswith("xens2/")),
            "xens2_protocol_blocked_records": 16},
        **strongest, "rows": all_rows, "new_prediction_ids": [row["prediction_id"] for row in new_rows],
        "new_per_record_metric_rows": len(metric_rows), "external_test_model_selection": False,
        "new_training_or_inference": True,
        "inference_scope": "official-only 1422 CPU baselines plus frozen xens2 GPU recovery; 16 protocol blocks retained",
        "gpu_used": xens2_gpu_recovery_used, "shared_registry_modified": False}
    write_json(OUT / "system_metrics.json", document)
    recommendation = {"schema_version": SCHEMA + ".registry_recommendation", "recommendation_only": True,
        "apply_by_unified_process_only": True, "shared_registry_modified": False,
        "suggested_prediction_overlay": {"path": "artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json",
            "sha256": sha256(OUT / "system_metrics.json"), "new_prediction_ids": document["new_prediction_ids"]},
        "recovery_coverage": recovery_summary, "strongest": strongest,
        "legacy_derived_label_disposition": "circular-audit-only; historical values retained",
        "xens2_full_3966_delta": "UNCHECKED_XENS2_NOT_FULL_COVERAGE"}
    write_json(OUT / "registry_recommendation.json", recommendation)
    write_json(OUT / "source_manifest.json", {"schema_version": SCHEMA + ".source_manifest",
        "recovery_plan": str((SOURCE / "recovery_plan.json").relative_to(ROOT)), "recovery_plan_sha256": sha256(SOURCE / "recovery_plan.json"),
        "work_root": str(WORK), "base_system_metrics_sha256": sha256(BASE / "system_metrics.json"),
        "full_panel_sha256": sha256(FULL_PANEL), "clean_panel_sha256": sha256(CLEAN_PANEL),
        "xens2_queue_evidence_sha256": sha256(XENS2_EVIDENCE_PATH),
        "xens2_live_queue_status": str(XENS2_LIVE_STATUS_PATH),
        "xens2_live_queue_status_sha256": sha256(XENS2_LIVE_STATUS_PATH),
        "predictor_metadata": {baseline: {"path": str((XENS2_WORK if baseline == "xens2" else WORK / baseline) / "predictor_metadata.json"),
            "sha256": recovery_summary[baseline]["predictor_metadata_sha256"]} for baseline in BASELINES},
        "shared_registry_modified": False, "gpu_used": xens2_gpu_recovery_used, "external_test_model_selection": False})
    lines = ["# ArchiveII full3966 恢复双面板", "",
             "新预测仅来自官方 clean 2544 之外的 1422 条，原始 sequence/SHA 未改写；新 identity 为 clean-frozen + recovery composite。缺预测不补 0，部分覆盖不排名。xens2 已恢复 1406 条、16 条协议阻断，composite 为 3950/3966，故 full3966 的 xens2 差距仍保持 `UNCHECKED_XENS2_NOT_FULL_COVERAGE`。", "",
             "| Baseline | checkpoint 完成 | PASS | coverage | full3966 可排名 |",
             "|---|---:|---:|---:|---|"]
    for baseline in BASELINES:
        item = recovery_summary[baseline]
        lines.append(f"| {baseline} | {item['n_completed']} / 1422 | {item['n_pass']} | {item['coverage']:.10f} | {item['complete_for_full_ranking']} |")
    lines += ["", "## full3966 全覆盖指标", "",
              "| Prediction identity | Model form | n | micro F1 | macro F1 |",
              "|---|---|---:|---:|---:|"]
    for row in sorted(full_eligible, key=lambda item: item["panels"][FULL_ID]["metrics"]["micro_f1"], reverse=True):
        metric = row["panels"][FULL_ID]["metrics"]
        lines.append(f"| `{row['prediction_id']}` | {row['model_form']} | 3966 | {metric['micro_f1']:.10f} | {metric['macro_f1']:.10f} |")
    lines += ["", "## 新增 composite 在 clean 2544 的 xens2 差距", "",
              "仅列 clean 全覆盖的新身份；Vienna 没有旧 clean 记录级预测，故不在此表补 0 或排名。", "",
              "| Prediction identity | n | micro F1 | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in new_rows:
        result = row["panels"][CLEAN_ID]
        if result["status"] != "PASS_FULL_COVERAGE":
            continue
        metric, delta = result["metrics"], result["delta_vs_xens2"]
        lines.append(f"| `{row['prediction_id']}` | 2544 | {metric['micro_f1']:.10f} | {metric['macro_f1']:.10f} | {delta['micro_f1']:+.10f} | {delta['macro_f1']:+.10f} |")
    lines += ["", "## 最强结果", ""]
    for panel in ("clean_2544", "full_3966"):
        for kind in ("strongest_ensemble", "strongest_single_predictor"):
            item = strongest[panel][kind]
            if isinstance(item, str):
                lines.append(f"- {panel} {kind}: `{item}`")
            else:
                lines.append(f"- {panel} {kind}: `{item['prediction_id']}`，micro-F1={item['metrics']['micro_f1']:.16f}，macro-F1={item['metrics']['macro_f1']:.16f}。")
    queue_evidence = recovery_summary["xens2"].get("queue_evidence")
    if queue_evidence:
        queue_status = queue_evidence["status"]
        lines += ["", f"xens2 探针 peak={queue_status['measured_peak_mib']} MiB，生产门限={queue_status['required_free_mib']} MiB，快照 free={queue_status['device']['free_mib']} MiB，状态=`{queue_status['state']}`，排队 PID=`{queue_evidence['queue_pid']}`。未启动生产，未声称 xens2 full3966。"]
    if xens2_gpu_recovery_used:
        live_queue = json.loads(XENS2_LIVE_STATUS_PATH.read_text())
        lines += ["", f"xens2 live queue=`{live_queue['state']}`，处理 {live_queue['coverage']['n_completed']}/1422，pass={live_queue['coverage']['n_pass']}，protocol-blocked={live_queue['coverage']['n_blocked']}，预测 SHA256=`{live_queue['coverage']['predictions_sha256']}`。恢复完成不等于 full coverage；3950/3966 仍不排名。"]
    paused = recovery_summary["rnastructure_maxexpect"].get("pause")
    if paused:
        lines += ["", f"MaxExpect 已安全暂停在 {recovery_summary['rnastructure_maxexpect']['n_completed']} 条，原 PID `{paused['pid']}`。恢复命令：`{paused['resume_command']}`。其 coverage 仍为部分，不参与 full3966 排名。"]
    lines += ["", "完整逐 ID coverage、指标、delta/CI 和 registry 建议见 `system_metrics.json`、`recovered_per_record_metrics.jsonl.gz`、`official_coverage_gaps.json` 与 `registry_recommendation.json`。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "official_coverage_gaps.json", {"schema_version": SCHEMA + ".coverage_gaps",
        "recovery": recovery_summary, "new_prediction_panels": {row["prediction_id"]: row["panels"] for row in new_rows}})
    output_files = [OUT / "system_metrics.json", OUT / "registry_recommendation.json", OUT / "source_manifest.json",
                    OUT / "official_coverage_gaps.json", OUT / "recovered_per_record_metrics.jsonl.gz", REPORT,
                    VIENNA_PLAN_PATH, XENS2_EVIDENCE_PATH]
    output_files += [OUT / "predictions" / f"{baseline}.jsonl.gz" for baseline in BASELINES
                     if (OUT / "predictions" / f"{baseline}.jsonl.gz").is_file()]
    write_json(OUT / "build_receipt.json", {"schema_version": SCHEMA + ".build_receipt", "status": "PASS",
        "new_per_record_metric_rows": len(metric_rows), "recovery_complete_baselines": sum(row["recovery_complete_1422"] for row in new_rows),
        "outputs": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files}})
    print(json.dumps({"status": "PASS", "recovery_complete": document["coverage_accounting"],
                      "full_strongest": strongest["full_3966"]}, sort_keys=True))


if __name__ == "__main__":
    main()
