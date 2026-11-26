#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests/foldnar_v1"
OUTPUT = ROOT / "source_data/bprna_official_reference_5388"
OFFICIAL = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
PROJECT = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
RAW_DATA_SHA = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
OFFICIAL_DATA_SHA = "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403"
PANEL_DATASET = "bpfold_official_bprna_new_reference_5388"
OVERLAY_NAME = "bprna_new_official_reference_5388_prediction_registry.json"
SEED = 20261127
BOOTSTRAP = 10000
MODEL_FORMS = {
    "xens2/bprna_new": "three_checkpoint_ensemble",
    "rinalmo_2d_s0/bprna_new": "single_model",
    "mxfold2/bprna_new": "deterministic_single_predictor",
    "eternafold/bprna_new": "deterministic_single_predictor",
    "rnaformer_interfamily/bprna_new": "single_model",
    "rnaformer_intrafamily/bprna_new": "single_model",
    "rnaformer_biophysical/bprna_new": "single_model",
    "kirigami_fast/bprna_new": "single_model",
    "structrfm_official_v0.0.8/bprna_new": "single_model",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pair_set(values, length):
    result = set()
    occupied = set()
    for value in values:
        left, right = map(int, value)
        assert 0 <= left < right < length, (left, right, length)
        assert left not in occupied and right not in occupied, (left, right)
        occupied.update((left, right))
        result.add((left, right))
    return result


def parse_bpseq(payload):
    sequence = []
    partners = []
    for raw in payload.decode().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        index, base, partner = int(fields[0]), fields[1].upper().replace("T", "U"), int(fields[2])
        assert index == len(sequence) + 1
        sequence.append(base)
        partners.append(partner)
    pairs = set()
    for index, partner in enumerate(partners):
        if partner:
            assert 1 <= partner <= len(sequence)
            assert partners[partner - 1] == index + 1
            pairs.add(tuple(sorted((index, partner - 1))))
    return "".join(sequence), pair_set(pairs, len(sequence))


def load_panel(output_dir):
    project_rows = [json.loads(line) for line in PROJECT.read_text().splitlines() if line.strip()]
    assert sha256(PROJECT) == RAW_DATA_SHA and len(project_rows) == 5388
    names = [row["name"] for row in project_rows]
    assert len(set(names)) == len(names)
    official = {}
    with tarfile.open(OFFICIAL, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.startswith("BPfold_data/test_data/bpRNAnew/") and member.name.endswith(".bpseq"):
                sequence, pairs = parse_bpseq(archive.extractfile(member).read())
                official[Path(member.name).name] = (sequence, pairs)
    assert sha256(OFFICIAL) == OFFICIAL_DATA_SHA and len(official) == 5401
    panel = []
    for name in names:
        assert name in official
        sequence, pairs = official[name]
        core = {
            "name": name,
            "seq": sequence,
            "pairs": [list(pair) for pair in sorted(pairs)],
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "pair_set_sha256": hash_value(sorted(pairs)),
        }
        panel.append({**core, "record_sha256": hash_value(core)})
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "official_reference_5388.jsonl.gz"
    payload = "".join(canonical(row) + "\n" for row in panel).encode()
    panel_path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return panel, panel_path


def load_predictions(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        document = json.loads(path.read_text())
        if "per_sequence" in document:
            rows = document["per_sequence"]
        elif "predictions" in document:
            rows = document["predictions"]
        else:
            raise ValueError("NO_PER_SEQUENCE_PREDICTIONS")
    result = {}
    for row in rows:
        name = row["name"]
        assert name not in result, name
        pairs = row.get("pairs", row.get("pred_pairs"))
        if pairs is None:
            raise ValueError("NO_PER_SEQUENCE_PAIRS")
        result[name] = {"pairs": pairs, "seq": row.get("seq")}
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
    by_name = {row["name"]: row for row in panel}
    missing = sorted(set(by_name) - set(predictions))
    extra = sorted(set(predictions) - set(by_name))
    tp = fp = fn = 0
    precisions, recalls, f1_values = [], [], []
    per_sequence = {}
    for label in panel:
        name = label["name"]
        if name not in predictions:
            continue
        prediction = predictions[name]
        if prediction.get("seq") is not None:
            assert prediction["seq"].upper().replace("T", "U") == label["seq"].upper().replace("T", "U"), name
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(prediction["pairs"], len(label["seq"]))
        a, b, c, p, r, f = prf(truth, predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        precisions.append(p)
        recalls.append(r)
        f1_values.append(f)
        per_sequence[name] = {"tp": a, "fp": b, "fn": c, "precision": p, "recall": r, "f1": f}
    n = len(f1_values)
    metrics = {
        "n_scored": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_precision": math.fsum(precisions) / n if n else None,
        "macro_recall": math.fsum(recalls) / n if n else None,
        "macro_f1": math.fsum(f1_values) / n if n else None,
    }
    return metrics, missing, extra, per_sequence


def inventory(recovered, predictions):
    entries = []
    seen = set()
    for row in recovered["rows"]:
        entries.append({
            "prediction_id": row["prediction_id"], "asset_id": row["asset_id"],
            "baseline_id": row["baseline_id"], "model_form": row["model_form"],
            "prediction_origin": row["prediction_origin"], "path": row["prediction_path"],
            "sha256": row["prediction_sha256"], "registry_status": row["status"],
            "source_group": "RECOVERED_20",
        })
        seen.add(row["prediction_id"])
    for row in predictions["rows"]:
        in_scope = row.get("dataset_id") == "bprna_new" or row.get("split") == "bprna_new"
        if not in_scope or row["prediction_id"] in seen:
            continue
        if row.get("n_sequences") not in (5388, 5386) and row["prediction_id"] not in {"best_plan_a/bprna_new", "h0/bprna_new"}:
            continue
        entries.append({
            "prediction_id": row["prediction_id"], "asset_id": None,
            "baseline_id": row.get("baseline_id", row.get("model")),
            "model_form": MODEL_FORMS.get(row["prediction_id"], "single_model"),
            "prediction_origin": "registered_frozen_prediction",
            "path": row.get("path"), "sha256": row.get("sha256"),
            "registry_status": row.get("status"), "source_group": "ADDITIONAL_REGISTERED",
        })
        seen.add(row["prediction_id"])
    return entries


def bootstrap_interval(deltas):
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = []
    for start in range(0, BOOTSTRAP, 200):
        size = min(200, BOOTSTRAP - start)
        indices = rng.integers(0, n, size=(size, n))
        means.extend(np.asarray(deltas)[indices].mean(axis=1).tolist())
    return [float(value) for value in np.percentile(np.asarray(means), [2.5, 97.5])]


def public_strongest(row):
    return {
        "prediction_id": row["prediction_id"], "asset_id": row["asset_id"],
        "baseline_id": row["baseline_id"], "model_form": row["model_form"],
        "metrics": row["metrics"], "delta_vs_xens2": row["delta_vs_xens2"],
        "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION",
    }


def evaluate(entries, panel):
    rows = []
    per_sequence_f1 = {}
    for entry in entries:
        row = dict(entry)
        row.update({
            "label_dataset_id": PANEL_DATASET, "n_panel_sequences": len(panel),
            "n_prediction_records": 0, "n_with_prediction": 0, "coverage": 0.0,
            "missing_count": len(panel), "missing_names": [], "missing_names_enumerated": False, "extra_names": [],
            "metrics": None, "status": None,
        })
        if not row["path"]:
            row["status"] = "BLOCKED_MISSING_PREDICTION_PATH"
        elif not Path(row["path"]).exists():
            row["status"] = "BLOCKED_PREDICTION_PATH_NOT_FOUND"
        elif not row["sha256"] or sha256(row["path"]) != row["sha256"]:
            row["status"] = "BLOCKED_PREDICTION_SHA_MISMATCH"
        else:
            try:
                predictions = load_predictions(row["path"])
            except (ValueError, KeyError, TypeError) as error:
                row["status"] = "BLOCKED_NO_PER_SEQUENCE_PREDICTIONS"
                row["load_error"] = str(error)
            else:
                metrics, missing, extra, sequence_metrics = score_present(panel, predictions)
                row.update({
                    "n_prediction_records": len(predictions),
                    "n_with_prediction": len(panel) - len(missing),
                    "coverage": (len(panel) - len(missing)) / len(panel),
                    "missing_count": len(missing), "missing_names": missing, "missing_names_enumerated": True, "extra_names": extra,
                    "metrics": metrics,
                    "status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                })
                if not missing:
                    per_sequence_f1[row["prediction_id"]] = {name: values["f1"] for name, values in sequence_metrics.items()}
        rows.append(row)
    assert len([row for row in rows if row["source_group"] == "RECOVERED_20"]) == 20
    assert len([row for row in rows if row["status"] == "PASS_FULL_COVERAGE"]) == 28
    xens2 = next(row for row in rows if row["prediction_id"] == "xens2/bprna_new")
    assert xens2["status"] == "PASS_FULL_COVERAGE"
    names = [row["name"] for row in panel]
    xens2_f1 = np.asarray([per_sequence_f1[xens2["prediction_id"]][name] for name in names])
    for row in rows:
        if row["status"] != "PASS_FULL_COVERAGE":
            row["delta_vs_xens2"] = None
            row["paired_macro_f1_delta_ci95_vs_xens2"] = None
            continue
        row["delta_vs_xens2"] = {
            key: row["metrics"][key] - xens2["metrics"][key]
            for key in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")
        }
        deltas = np.asarray([per_sequence_f1[row["prediction_id"]][name] for name in names]) - xens2_f1
        row["paired_macro_f1_delta_ci95_vs_xens2"] = {
            "observed_delta": float(deltas.mean()), "ci95": bootstrap_interval(deltas),
            "method": "paired sequence-level percentile bootstrap", "n_bootstrap": BOOTSTRAP,
            "seed": SEED, "unit": "sequence", "applicability": "full common 5388 coverage only",
        }
    eligible = [row for row in rows if row["status"] == "PASS_FULL_COVERAGE"]
    ensemble = max((row for row in eligible if "ensemble" in row["model_form"]), key=lambda row: row["metrics"]["micro_f1"])
    single = max((row for row in eligible if "ensemble" not in row["model_form"]), key=lambda row: row["metrics"]["micro_f1"])
    overall = max(eligible, key=lambda row: row["metrics"]["micro_f1"])
    return rows, {"strongest_overall": public_strongest(overall), "strongest_ensemble": public_strongest(ensemble), "strongest_single_predictor": public_strongest(single)}


def render_report(document):
    strongest = document["strongest_overall"]
    single = document["strongest_single_predictor"]
    lines = [
        "# bpRNA-new official-reference 5388 协议修复面板",
        "",
        "## 范围",
        "",
        "本面板使用经认证 BPfold v0.2 `BPfold_data/test_data/bpRNAnew` 官方参考标签，按冻结项目 5,388 个 ID 的原顺序取交集。只进行 CPU 冻结预测复算，不做新推理、训练、阈值调整或模型选择。旧派生标签及历史数值保留，但其结果仅为 circular-audit-only，不再作为独立 SOTA 证据。0.9617 项目标签误配值不进入本面板或排名。",
        "",
        f"标签面板 SHA256：`{document['label_panel']['sha256']}`；ID 顺序 SHA256：`{document['label_panel']['id_order_sha256']}`。每条记录含 ID、sequence、pairs、sequence/pair-set/record SHA256。",
        "",
        "## 指标与覆盖",
        "",
        "严格 zero-based inclusive 精确配对；不追加 crossing、canonical 或 min-loop 过滤。micro 汇总 TP/FP/FN；macro 为逐序列 P/R/F1 的非加权平均。真值与预测均空时单序列 P=R=F1=1。排名仅使用 5,388 全覆盖身份，按 micro-F1 描述冻结结果，不构成选模。",
        "",
        f"清单共 {document['coverage_accounting']['inventory_entries']} 个身份：20 recovered + {document['coverage_accounting']['additional_registered']} additional registered；28 个全覆盖，1 个 5386 部分覆盖，14 个 blocked/summary-only。",
        "",
        "| Prediction identity | Model form | n | micro P | micro R | micro F1 | macro P | macro R | macro F1 | Δmicro F1 vs xens2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    full = sorted((row for row in document["rows"] if row["status"] == "PASS_FULL_COVERAGE"), key=lambda row: row["metrics"]["micro_f1"], reverse=True)
    for row in full:
        metric = row["metrics"]
        lines.append(f"| `{row['prediction_id']}` | {row['model_form']} | 5388 | {metric['micro_precision']:.10f} | {metric['micro_recall']:.10f} | {metric['micro_f1']:.10f} | {metric['macro_precision']:.10f} | {metric['macro_recall']:.10f} | {metric['macro_f1']:.10f} | {row['delta_vs_xens2']['micro_f1']:+.10f} |")
    lines.extend(["", "## 最强单预测器与集成", ""])
    for title, row in (("最强集成/总体", strongest), ("最强单预测器", single)):
        ci = next(item for item in document["rows"] if item["prediction_id"] == row["prediction_id"])["paired_macro_f1_delta_ci95_vs_xens2"]
        lines.append(f"- **{title}**：`{row['prediction_id']}`，micro-F1={row['metrics']['micro_f1']:.16f}，macro-F1={row['metrics']['macro_f1']:.16f}；相对 xens2 Δmicro={row['delta_vs_xens2']['micro_f1']:+.16f}、Δmacro={row['delta_vs_xens2']['macro_f1']:+.16f}，配对 95% CI [{ci['ci95'][0]:+.16f}, {ci['ci95'][1]:+.16f}]。")
    lines.extend([
        "",
        "配对区间为 10,000 次序列级 percentile bootstrap，固定 seed=20261127；仅适用于共同 5,388 全覆盖记录。每个系统相对 xens2 的 macro-F1 配对 CI 均在 `system_metrics.json`。",
        "",
        "## 部分覆盖与不可评分",
        "",
    ])
    for row in document["rows"]:
        if row["status"] == "PARTIAL_COVERAGE_NOT_RANKED":
            lines.append(f"- `{row['prediction_id']}`：{row['n_with_prediction']}/5388，缺失 `{row['missing_names']}`；指标仅对现有记录复算，不进入排名。")
    blocked = [row for row in document["rows"] if row["status"].startswith("BLOCKED")]
    lines.append("- Blocked/summary-only（不补值、不排名）：" + "、".join(f"`{row['prediction_id']}`" for row in blocked) + "。")
    lines.extend([
        "",
        "## Registry 修复",
        "",
        "- 项目旧标签标为 `DERIVED_FROM_BPFOLD_PREDICTIONS`，谱系指向 v0.2 release 预测及 min-loop/crossing 投影，旧 raw/clean 派生结果均为 circular-audit-only。",
        "- BPfold 官方 release 与项目六 checkpoint replay 保持不同 prediction identity；官方 5,401 指标保留，新面板只添加同一官方标签的 5,388 交集结果。",
        "- `data_registry.json`、`prediction_registry.json`、`baseline_registry.json` 和 recovered registry 只增加/修正谱系与比较边界，历史 metric 字段通过逐字段不变断言。",
        "",
        "机器可读完整结果：`source_data/bprna_official_reference_5388/system_metrics.json`；独立第二实现：`tools/verify_bprna_official_reference_panel_20261127.py`。",
        "",
    ])
    return "\n".join(lines)


def lineage():
    return {
        "label_source": "DERIVED_FROM_BPFOLD_PREDICTIONS",
        "source_release": "BPfold v0.2 BPfold_test_results.tar.gz",
        "source_release_sha256": "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073",
        "projection": "data/ss/prepare_decision_data.py project_to_legal; drop span<=3 then greedy crossing removal",
        "disposition": "circular-audit-only",
        "independent_sota_eligible": False,
        "historical_numeric_values_retained": True,
    }


def repair_registries(recovered, predictions, data, baselines, rows, strongest, panel_path):
    by_id = {row["prediction_id"]: row for row in rows}
    old_recovered_metrics = {row["prediction_id"]: json.dumps(row.get("metrics"), sort_keys=True) for row in recovered["rows"]}
    old_prediction_metrics = {row["prediction_id"]: json.dumps(row.get("metrics"), sort_keys=True) for row in predictions["rows"]}
    old_raw = next(row for row in data["rows"] if row["dataset_id"] == "bprna_new")
    old_raw.setdefault("legacy_label_source_before_20261127", old_raw.get("label_source"))
    old_raw["label_source"] = "DERIVED_FROM_BPFOLD_PREDICTIONS"
    old_raw["label_provenance_status"] = "CONFIRMED_DERIVED_FROM_BPFOLD_RELEASE_PREDICTIONS; circular-audit-only"
    old_raw["label_lineage"] = lineage()
    old_raw["independent_sota_eligible"] = False
    old_raw["official_reference_replacement_panel"] = PANEL_DATASET
    panel_row = {
        "dataset_id": PANEL_DATASET, "role": "external_protocol_reference",
        "description": "BPfold official reference labels restricted to frozen project 5388 IDs for same-label comparisons",
        "scope": "EXTERNAL_PROTOCOL_ONLY", "label_source": "BPFOLD_OFFICIAL_REFERENCE",
        "path": str(panel_path.relative_to(ROOT)), "exists": True,
        "bytes": panel_path.stat().st_size, "sha256": sha256(panel_path), "hash_scope": "full_content",
        "n_sequences": 5388, "status": "FROZEN",
        "source_archive": str(OFFICIAL), "source_archive_sha256": OFFICIAL_DATA_SHA,
        "selection_rule": "exact frozen project IDs only; coverage alignment, not model selection",
        "id_order_sha256": None,
        "prediction_ids": [],
        "comparability": "same official labels and IDs for all scored systems; full coverage required for ranking",
    }
    panel_rows = [json.loads(line) for line in gzip.decompress(panel_path.read_bytes()).decode().splitlines()]
    panel_row["id_order_sha256"] = hash_value([row["name"] for row in panel_rows])
    panel_row["prediction_ids"] = [row["prediction_id"] for row in rows if row["status"] == "PASS_FULL_COVERAGE"]
    data["rows"] = [row for row in data["rows"] if row["dataset_id"] != PANEL_DATASET] + [panel_row]
    data["n_rows"] = len(data["rows"])

    for row in predictions["rows"]:
        if row["prediction_id"] in by_id:
            audit = by_id[row["prediction_id"]]
            row["official_reference_5388"] = {
                "registry": OVERLAY_NAME, "label_dataset_id": PANEL_DATASET,
                "status": audit["status"], "coverage": audit["coverage"],
                "missing_names": audit["missing_names"], "metrics": audit["metrics"],
            }
        if row.get("data_path") == str(PROJECT) or row.get("dataset_id") == "bprna_new":
            if "label_protocol" in row and row["label_protocol"] != "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_5388":
                row.setdefault("legacy_label_protocol_before_20261127", row["label_protocol"])
            row["label_protocol"] = "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_5388"
            row["label_lineage"] = lineage()
            row["independent_sota_eligible"] = False
            row["historical_metrics_retained"] = True
            row["comparability"] = "DERIVED_LABEL_CIRCULAR_AUDIT_ONLY; not independent SOTA; use official_reference_5388 overlay"
        if "recovered_bprna_new" in row:
            nested = row["recovered_bprna_new"]
            if nested.get("label_protocol") == "PROJECT_DATABASE_RAW_5388" or nested.get("label_protocol") == "PROJECT_TR1C_CLEAN_5386":
                nested.setdefault("legacy_label_protocol_before_20261127", nested["label_protocol"])
                nested["label_protocol"] = "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_5388"
                nested["label_lineage"] = lineage()
                nested["independent_sota_eligible"] = False
                nested["sota_use"] = "circular-audit-only; historical value retained; not independent SOTA"
    predictions["official_reference_5388_overlay"] = f"manifests/foldnar_v1/{OVERLAY_NAME}"

    for row in recovered["rows"]:
        if row["label_protocol"] in {"PROJECT_DATABASE_RAW_5388", "PROJECT_TR1C_CLEAN_5386"}:
            row.setdefault("legacy_label_protocol_before_20261127", row["label_protocol"])
            row["label_protocol"] = "DERIVED_FROM_BPFOLD_PREDICTIONS_PROJECTED_5388"
            row["label_lineage"] = lineage()
            row["independent_sota_eligible"] = False
            row["sota_use"] = "circular-audit-only; historical value retained; not independent SOTA"
        audit = by_id[row["prediction_id"]]
        row["official_reference_5388"] = {
            "status": audit["status"], "coverage": audit["coverage"],
            "missing_names": audit["missing_names"], "metrics": audit["metrics"],
        }
    recovered["scope_contract"]["derived_project_labels_are_not_independent_sota"] = True
    recovered["scope_contract"]["official_reference_5388_full_coverage_assets"] = 28
    recovered["historical_raw_ranking_disposition"] = "CIRCULAR_AUDIT_ONLY_NOT_INDEPENDENT_SOTA; values retained"
    recovered["official_reference_5388"] = {
        "dataset_id": PANEL_DATASET, "panel_path": str(panel_path.relative_to(ROOT)),
        "panel_sha256": sha256(panel_path), "registry": OVERLAY_NAME, **strongest,
    }
    for key in ("strongest_project_raw", "strongest_project_raw_single_predictor", "xens2_project_raw_reference"):
        recovered[key]["independent_sota_eligible"] = False
        recovered[key]["disposition"] = "historical circular-audit-only; not independent SOTA"

    baseline_ids = {row["baseline_id"] for row in rows}
    for row in baselines["rows"]:
        if row["baseline_id"] not in baseline_ids:
            continue
        related = [item for item in rows if item["baseline_id"] == row["baseline_id"]]
        row["bprna_new_official_reference_5388"] = {
            "registry": OVERLAY_NAME, "label_dataset_id": PANEL_DATASET,
            "prediction_ids": [item["prediction_id"] for item in related],
            "ranking_scope": "descriptive frozen protocol audit only; not model selection",
        }
        comparability = row.setdefault("comparability", {})
        status = comparability.get("status")
        if ((status and "PROJECT_RAW" in status) or row["baseline_id"] == "bpfold") and "legacy_status_before_label_lineage_repair" not in comparability:
            comparability["legacy_status_before_label_lineage_repair"] = status
            comparability["status"] = "DERIVED_LABEL_CIRCULAR_AUDIT_ONLY; not independent SOTA; official-reference identities separated"
        if "bprna_new_recovery" in row:
            row["bprna_new_recovery"]["raw_clean_sota_claim"] = False
            row["bprna_new_recovery"]["derived_raw_disposition"] = "circular-audit-only; not independent SOTA"
    baselines["official_reference_5388_overlay"] = f"manifests/foldnar_v1/{OVERLAY_NAME}"

    assert old_recovered_metrics == {row["prediction_id"]: json.dumps(row.get("metrics"), sort_keys=True) for row in recovered["rows"]}
    assert old_prediction_metrics == {row["prediction_id"]: json.dumps(row.get("metrics"), sort_keys=True) for row in predictions["rows"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--no-registry-update", action="store_true")
    args = parser.parse_args()
    panel, panel_path = load_panel(args.output_dir)
    recovered = json.loads((MANIFESTS / "bprna_new_recovered_prediction_registry.json").read_text())
    predictions = json.loads((MANIFESTS / "prediction_registry.json").read_text())
    data = json.loads((MANIFESTS / "data_registry.json").read_text())
    baselines = json.loads((MANIFESTS / "baseline_registry.json").read_text())
    entries = inventory(recovered, predictions)
    rows, strongest = evaluate(entries, panel)
    document = {
        "schema_version": "bprna.official_reference_5388.prediction_registry.v1",
        "audit_date": "2026-11-27", "base_protocol_commit": "fe8349915714491096c0af059f5ff4fba876ae6d",
        "scope": "CPU-only frozen-prediction rescoring on one official-reference panel; no inference, training, threshold changes, or model selection",
        "label_panel": {
            "dataset_id": PANEL_DATASET, "path": str(panel_path.relative_to(ROOT)),
            "sha256": sha256(panel_path), "bytes": panel_path.stat().st_size,
            "n_sequences": len(panel), "source_archive": str(OFFICIAL),
            "source_archive_sha256": OFFICIAL_DATA_SHA,
            "id_order_sha256": hash_value([row["name"] for row in panel]),
            "record_schema": "name/seq/pairs/sequence_sha256/pair_set_sha256/record_sha256",
        },
        "metric_contract": {
            "pair_matching": "strict zero-based inclusive exact pairs; no additional crossing/canonical/min-loop filtering",
            "micro": "pool TP/FP/FN; zero-denominator precision or recall=0",
            "macro": "unweighted mean per-sequence P/R/F1; P=R=F1=1 only when truth and prediction are both empty",
            "ranking": "micro-F1, full 5388 coverage only; descriptive audit, not model selection",
            "paired_ci": "10000 paired sequence-level percentile bootstraps of macro-F1 delta versus xens2; seed 20261127",
        },
        "coverage_accounting": {
            "inventory_entries": len(rows), "recovered_20": 20,
            "additional_registered": len(rows) - 20,
            "full_coverage_scored": sum(row["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "partial_coverage_not_ranked": sum(row["status"] == "PARTIAL_COVERAGE_NOT_RANKED" for row in rows),
            "blocked_or_summary_only": sum(row["status"].startswith("BLOCKED") for row in rows),
        },
        **strongest,
        "xens2_reference": next(row for row in rows if row["prediction_id"] == "xens2/bprna_new"),
        "circular_project_label_score_excluded": True,
        "rows": rows,
    }
    write_json(args.output_dir / "system_metrics.json", document)
    report_path = ROOT / "docs/plans/2026-11-27-bprna-official-reference-5388-panel.md"
    report_path.write_text(render_report(document))
    if not args.no_registry_update:
        repair_registries(recovered, predictions, data, baselines, rows, strongest, panel_path)
        overlay = {key: value for key, value in document.items() if key != "rows"}
        overlay["rows"] = [{key: value for key, value in row.items() if key not in {"delta_vs_xens2", "paired_macro_f1_delta_ci95_vs_xens2"}} for row in rows]
        write_json(MANIFESTS / OVERLAY_NAME, overlay)
        write_json(MANIFESTS / "bprna_new_recovered_prediction_registry.json", recovered)
        write_json(MANIFESTS / "prediction_registry.json", predictions)
        write_json(MANIFESTS / "data_registry.json", data)
        write_json(MANIFESTS / "baseline_registry.json", baselines)
    print(json.dumps({"status": "PASS", "panel_sha256": sha256(panel_path), "coverage": document["coverage_accounting"], "strongest": strongest, "system_metrics": str(args.output_dir / "system_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
