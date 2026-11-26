#!/usr/bin/env python3
"""Rescore frozen ArchiveII predictions on unfiltered official references, CPU only."""
from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/archiveii_official_reference_20261127"
OUT = ROOT / "artifacts/archiveii_official_reference_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-archiveii-official-reference-panel.md"
J = Path("/root/rna_jepa_data/ss_data/jsonl")
PROJECT = J / "archiveii_embok_clean.jsonl"
PROJECT_SHA = "6c1fbaa4684cbe4644acdac2f31ae95e5f9b0eb36871f23d8148006f3d571399"
BD = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
BR = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
MX = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2/archiveII.tar.gz")
BD_SHA = "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403"
BR_SHA = "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073"
MX_SHA = "633bc50f8e9381f737b680fa6b4c79061a0a781773a5c38b342856eb1a6ac8a4"
SCHEMA = "archiveii.official_reference.panel.v1"
SEED = 20261127
BOOTSTRAP = 10000
FULL_ID = "archiveii_official_reference_3966"
CLEAN_ID = "archiveii_official_reference_clean_2544"
XENS2 = "xens2/archiveii_embok_clean"
RELEASE_ID = "bpfold_official_release/archiveii"
R2D_S1 = "rinalmo_2d_s1/archiveii_embok_clean"
R2D_S2 = "rinalmo_2d_s2/archiveii_embok_clean"
TRUNC_ID = "rinalmo_2d_s0/archiveii_embok_clean_trunc598"
ENSEMBLES = {XENS2, "bpfold/archiveii_embok_clean", RELEASE_ID}
DETERMINISTIC = {
    "contrafold/archiveii_embok_clean", "eternafold/archiveii_embok_clean",
    "mxfold2/archiveii_embok_clean", "rnastructure_fold/archiveii_embok_clean",
    "rnastructure_maxexpect/archiveii_embok_clean",
}


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


def pair_set(values, length):
    result = set()
    occupied = set()
    for value in values:
        i, j = map(int, value)
        if not 0 <= i < j < length:
            raise ValueError(f"illegal pair {(i, j)} for length {length}")
        if i in occupied or j in occupied:
            raise ValueError(f"partner conflict {(i, j)}")
        occupied.update((i, j))
        result.add((i, j))
    return result


def parse_bpseq(payload):
    bases, partners = [], []
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or int(fields[0]) != len(bases) + 1:
            raise ValueError("invalid BPSEQ")
        bases.append(fields[1].upper().replace("T", "U"))
        partners.append(int(fields[2]))
    pairs = set()
    for i, partner in enumerate(partners):
        if partner:
            if not 1 <= partner <= len(bases) or partners[partner - 1] != i + 1:
                raise ValueError("BPSEQ partner mismatch")
            if partner != i + 1:
                pairs.add(tuple(sorted((i, partner - 1))))
    return "".join(bases), pair_set(pairs, len(bases))


def tar_records(path, prefix):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            payload = archive.extractfile(member).read()
            sequence, pairs = parse_bpseq(payload)
            name = Path(member.name).name
            if name in result:
                raise ValueError(f"duplicate BPSEQ {name}")
            result[name] = {"seq": sequence, "pairs": pairs, "member": member.name,
                            "member_sha256": hashlib.sha256(payload).hexdigest()}
    return result


def build_panels():
    if sha256(BD) != BD_SHA or sha256(MX) != MX_SHA or sha256(PROJECT) != PROJECT_SHA:
        raise ValueError("reference/project input identity changed")
    official = tar_records(BD, "BPfold_data/test_data/archiveII/")
    mirror = tar_records(MX, "data/archiveII/")
    if len(official) != 3966 or official.keys() != mirror.keys():
        raise ValueError("official reference population mismatch")
    project = [json.loads(line) for line in PROJECT.read_text().splitlines() if line.strip()]
    if len(project) != 2544 or len({row["name"] for row in project}) != 2544:
        raise ValueError("project population mismatch")
    full, clean = [], []
    for name in sorted(official):
        left, right = official[name], mirror[name]
        if left["seq"] != right["seq"] or left["pairs"] != right["pairs"]:
            raise ValueError(f"reference package mismatch {name}")
        core = {
            "name": name, "seq": left["seq"], "pairs": [list(pair) for pair in sorted(left["pairs"])],
            "sequence_sha256": hashlib.sha256(left["seq"].encode()).hexdigest(),
            "pair_set_sha256": value_sha(sorted(left["pairs"])),
        }
        row = {**core, "record_sha256": value_sha(core),
               "bpfold_member": left["member"], "bpfold_member_sha256": left["member_sha256"],
               "mxfold2_member": right["member"], "mxfold2_member_sha256": right["member_sha256"]}
        full.append(row)
    by_name = {row["name"]: row for row in full}
    for project_row in project:
        name = project_row["name"]
        if name not in by_name or by_name[name]["seq"] != project_row["seq"].upper().replace("T", "U"):
            raise ValueError(f"project/reference sequence mismatch {name}")
        clean.append(by_name[name])
    if sum(not row["pairs"] for row in full) != 5 or sum(not row["pairs"] for row in clean) != 4:
        raise ValueError("official empty-label accounting changed")
    write_gzip_jsonl(SOURCE / "official_reference_3966.jsonl.gz", full)
    write_gzip_jsonl(SOURCE / "official_reference_clean_2544.jsonl.gz", clean)
    return full, clean


def inventory():
    registry = json.loads((ROOT / "manifests/foldnar_v1/prediction_registry.json").read_text())
    rows = []
    for row in sorted((item for item in registry["rows"] if item.get("dataset_id") == "archiveii_embok_clean"), key=lambda item: item["prediction_id"]):
        available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
        rows.append({
            "prediction_id": row["prediction_id"], "source_group": "REGISTERED",
            "registry_status": row.get("status"), "path": row.get("path"), "sha256": row.get("sha256"),
            "data_path": row.get("data_path"), "available": available,
            "model_form": "three_checkpoint_ensemble" if row["prediction_id"] == XENS2 else
                          "six_checkpoint_ensemble" if row["prediction_id"] == "bpfold/archiveii_embok_clean" else
                          "deterministic_single_predictor" if row["prediction_id"] in DETERMINISTIC else "single_model",
            "prediction_origin": "project_bpfold_replay" if row["prediction_id"] == "bpfold/archiveii_embok_clean" else "registered_frozen_prediction",
        })
    additions = [
        (RELEASE_ID, str(BR), BR_SHA, "six_checkpoint_ensemble", "bpfold_author_release_v0.2"),
        (R2D_S1, "/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000/archiveii_embok_clean/result.json", "a8a0875fb63cdeb4782d6b6cfa28f0d87766d2e5989085d8f0f743b6827d8151", "single_model", "additional_frozen_r2d_seed1"),
        (R2D_S2, "/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000/archiveii_embok_clean/result.json", "376278c46e29733d611a9343381bdd6f2b5ae4cbb4d593cf32bf0cfee5459caa", "single_model", "additional_frozen_r2d_seed2"),
        (TRUNC_ID, "/root/rna_jepa_data/eval_decision/nb_jev_r2dtr1c_s0_step20000/archiveii_embok_clean_trunc598/result.json", "89990cecfaf8c5c98cdf6608044d089a5b9236285067605ccb78d53ae76f19d8", "single_model", "first598_protocol_not_full_length"),
    ]
    for prediction_id, path, digest, form, origin in additions:
        rows.append({"prediction_id": prediction_id, "source_group": "INDEPENDENT_AUDIT_ASSET",
                     "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET", "path": path,
                     "sha256": digest, "data_path": str(PROJECT) if prediction_id != TRUNC_ID else str(J / "archiveii_embok_clean_trunc598.jsonl"),
                     "available": prediction_id != TRUNC_ID, "model_form": form, "prediction_origin": origin})
    if len(rows) != 36 or sum(row["available"] for row in rows) != 20:
        raise ValueError("prediction inventory changed")
    return rows


def load_predictions(entry):
    if entry["prediction_id"] == RELEASE_ID:
        return tar_records(BR, "BPfold_test_results/archiveII/"), {"sequence_source": "BPSEQ sequence in author release"}
    path = Path(entry["path"])
    document = json.loads(path.read_text())
    if document.get("data") and document["data"] != entry["data_path"]:
        raise ValueError(f"prediction data path mismatch {entry['prediction_id']}")
    if document.get("data_sha256") and document["data_sha256"] != PROJECT_SHA:
        raise ValueError(f"prediction data SHA mismatch {entry['prediction_id']}")
    if entry["prediction_id"].startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    elif "per_sequence" in document:
        source = document["per_sequence"]
    elif "predictions" in document:
        source = document["predictions"]
    else:
        raise ValueError("NO_RECORD_LEVEL_PREDICTIONS")
    result = {}
    for row in source:
        name = row["name"]
        pairs = row.get("pred_pairs", row.get("pairs"))
        if name in result or pairs is None:
            raise ValueError(f"missing pairs or duplicate ID {name}")
        result[name] = {"pairs": pairs, "seq": row.get("seq")}
    return result, {"sequence_source": "frozen source data path/SHA plus per-ID reference join; direct seq checked when present"}


def prf(truth, predicted):
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    if not truth and not predicted:
        return tp, fp, fn, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1


def score_present(panel, predictions):
    tp = fp = fn = 0
    values, per_sequence, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in predictions:
            missing.append(name)
            continue
        prediction = predictions[name]
        if prediction.get("seq") is not None and prediction["seq"].upper().replace("T", "U") != label["seq"]:
            raise ValueError(f"prediction sequence mismatch {name}")
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(prediction["pairs"], len(label["seq"]))
        a, b, c, p, r, f = prf(truth, predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        values.append((p, r, f))
        per_sequence[name] = {
            "tp": a, "fp": b, "fn": c, "precision": p, "recall": r, "f1": f,
            "truth_empty": not truth, "prediction_empty": not predicted,
            "empty_empty": not truth and not predicted,
        }
    n = len(values)
    metrics = {
        "n_scored": n, "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
        "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
        "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
        "empty_empty_records": sum(row["empty_empty"] for row in per_sequence.values()),
        "truth_empty_records": sum(row["truth_empty"] for row in per_sequence.values()),
    }
    extra = sorted(set(predictions) - {row["name"] for row in panel})
    return metrics, per_sequence, missing, extra


def bootstrap_interval(deltas):
    rng = np.random.default_rng(SEED)
    values = []
    array = np.asarray(deltas)
    for start in range(0, BOOTSTRAP, 200):
        size = min(200, BOOTSTRAP - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        values.extend(array[indices].mean(axis=1).tolist())
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def panel_identity(panel, path, dataset_id):
    return {
        "dataset_id": dataset_id, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "bytes": path.stat().st_size, "n_sequences": len(panel),
        "id_order_sha256": value_sha([row["name"] for row in panel]),
        "empty_reference_records": sum(not row["pairs"] for row in panel),
        "source_archives": {"BPfold_data": {"path": str(BD), "sha256": BD_SHA},
                            "MXfold2_archiveII": {"path": str(MX), "sha256": MX_SHA}},
        "pair_protocol": "unfiltered official BPSEQ; no min-loop/crossing/canonical projection",
    }


def public_strongest(row):
    clean = row["panels"][CLEAN_ID]
    return {"prediction_id": row["prediction_id"], "model_form": row["model_form"],
            "metrics": clean["metrics"], "delta_vs_xens2": clean["delta_vs_xens2"],
            "ranking_scope": "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION"}


def evaluate(entries, full, clean):
    panels = ((FULL_ID, full), (CLEAN_ID, clean))
    output, record_rows, clean_f1 = [], [], {}
    for entry in entries:
        row = dict(entry)
        row["panels"] = {}
        if not entry["available"]:
            status = "BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH" if entry["prediction_id"] == TRUNC_ID else (
                "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED" if entry["registry_status"] == "SUMMARY_ONLY" else "BLOCKED_REGISTERED_PREDICTION_MISSING")
            for panel_id, panel in panels:
                row["panels"][panel_id] = {"status": status, "n_panel_sequences": len(panel),
                                           "n_with_prediction": 0, "coverage": 0.0, "metrics": None,
                                           "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None}
            output.append(row)
            continue
        if not Path(entry["path"]).is_file() or sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"prediction SHA missing/mismatch {entry['prediction_id']}")
        predictions, sequence_join = load_predictions(entry)
        row["sequence_join"] = sequence_join
        row["n_prediction_records"] = len(predictions)
        for panel_id, panel in panels:
            metrics, per_sequence, missing, extra = score_present(panel, predictions)
            coverage = metrics["n_scored"] / len(panel)
            status = "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED"
            row["panels"][panel_id] = {
                "status": status, "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                "coverage": coverage, "missing_count": len(missing), "missing_names": missing,
                "extra_count": len(extra), "extra_names": extra, "metrics": metrics,
                "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None,
                "missing_predictions_imputed_as_zero": False,
            }
            for name, values in per_sequence.items():
                record_rows.append({"panel_id": panel_id, "prediction_id": entry["prediction_id"], "name": name, **values})
            if panel_id == CLEAN_ID and not missing:
                clean_f1[entry["prediction_id"]] = {name: values["f1"] for name, values in per_sequence.items()}
        output.append(row)
    if len(clean_f1) != 20 or XENS2 not in clean_f1:
        raise ValueError("clean full-coverage population changed")
    names = [row["name"] for row in clean]
    xens2_values = np.asarray([clean_f1[XENS2][name] for name in names])
    for row in output:
        clean_result = row["panels"][CLEAN_ID]
        if clean_result["status"] != "PASS_FULL_COVERAGE":
            continue
        metrics = clean_result["metrics"]
        reference = next(item for item in output if item["prediction_id"] == XENS2)["panels"][CLEAN_ID]["metrics"]
        clean_result["delta_vs_xens2"] = {key: metrics[key] - reference[key] for key in (
            "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        values = np.asarray([clean_f1[row["prediction_id"]][name] for name in names])
        deltas = values - xens2_values
        clean_result["paired_macro_f1_delta_ci95_vs_xens2"] = {
            "observed_delta": float(deltas.mean()), "ci95": bootstrap_interval(deltas),
            "method": "paired sequence-level percentile bootstrap", "n_bootstrap": BOOTSTRAP,
            "seed": SEED, "unit": "sequence", "applicability": "full common clean 2544 coverage only",
        }
    eligible = [row for row in output if row["panels"][CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"]
    strongest = {
        "strongest_overall": public_strongest(max(eligible, key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"])),
        "strongest_ensemble": public_strongest(max((row for row in eligible if "ensemble" in row["model_form"]), key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"])),
        "strongest_single_predictor": public_strongest(max((row for row in eligible if "ensemble" not in row["model_form"]), key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"])),
    }
    full_eligible = [row for row in output if row["panels"][FULL_ID]["status"] == "PASS_FULL_COVERAGE"]
    if [row["prediction_id"] for row in full_eligible] != [RELEASE_ID]:
        raise ValueError("only the author release should have official full coverage")
    return output, record_rows, strongest


def render_report(document):
    rows = document["rows"]
    clean_rows = sorted((row for row in rows if row["panels"][CLEAN_ID]["status"] == "PASS_FULL_COVERAGE"),
                        key=lambda row: row["panels"][CLEAN_ID]["metrics"]["micro_f1"], reverse=True)
    lines = [
        "# ArchiveII 未过滤 official-reference 3966 / clean 2544 重算面板", "",
        "本面板仅对冻结预测做 CPU 复算，不做训练、新推理、阈值调整或模型选择。旧 `DERIVED_FROM_BPFOLD_PREDICTIONS` 标签及旧分数只保留 circular-audit-only。共享 registry 未修改；`registry_recommendation.json` 供后续统一整合。", "",
        "BPfold_data 与 MXfold2 的 3966 条官方参考 BPSEQ 已逐条验证 ID、sequence、pairs 和 member SHA 一致。严格 zero-based inclusive 配对计分，不追加 min-loop/crossing/canonical 过滤；truth-empty 记录全部计入 macro，仅 empty/empty 的单序列 P/R/F1=1，truth-empty 而预测非空仍按普通规则计分。官方全集 5 条、clean 交集 4 条 truth-empty；release 对这些记录均产生非空预测，因此实际 empty/empty=0。缺失预测不补 0。", "",
        "## clean 2544 全覆盖结果（描述性，不用于选模）", "",
        "| Prediction identity | Model form | n | micro P | micro R | micro F1 | macro P | macro R | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in clean_rows:
        result = row["panels"][CLEAN_ID]
        metric, delta = result["metrics"], result["delta_vs_xens2"]
        lines.append(f"| `{row['prediction_id']}` | {row['model_form']} | 2544 | {metric['micro_precision']:.10f} | {metric['micro_recall']:.10f} | {metric['micro_f1']:.10f} | {metric['macro_precision']:.10f} | {metric['macro_recall']:.10f} | {metric['macro_f1']:.10f} | {delta['micro_f1']:+.10f} | {delta['macro_f1']:+.10f} |")
    lines += ["", "## 最强单预测器与集成", ""]
    for title, key in (("最强集成/总体", "strongest_ensemble"), ("最强单预测器", "strongest_single_predictor")):
        row = document[key]
        metric, delta = row["metrics"], row["delta_vs_xens2"]
        ci = next(item for item in rows if item["prediction_id"] == row["prediction_id"])["panels"][CLEAN_ID]["paired_macro_f1_delta_ci95_vs_xens2"]["ci95"]
        lines.append(f"- **{title}**：`{row['prediction_id']}`，micro-F1={metric['micro_f1']:.16f}，macro-F1={metric['macro_f1']:.16f}；相对 xens2 Δmicro={delta['micro_f1']:+.16f}、Δmacro={delta['macro_f1']:+.16f}，配对 macro Δ95% CI [{ci[0]:+.16f}, {ci[1]:+.16f}]。")
    lines += ["", "BPfold release 与项目 replay 是不同 prediction identity；release 可覆盖全集 3966，replay 只覆盖 clean 2544。配对区间采用固定 seed=20261127 的 10,000 次序列级 percentile bootstrap，仅适用于 clean 2544 共同全覆盖。", "",
              "## 官方全集 3966 覆盖与 present-only 指标", "",
              "只有全覆盖身份可参与全集最强判断。部分覆盖指标只对实际存在的预测计分，不把 1422 条缺失记录补 0，不计算全集相对 xens2 差距，也不进入排名。", "",
              "| Prediction identity | n / 3966 | coverage | status | present-only micro F1 | present-only macro F1 |",
              "|---|---:|---:|---|---:|---:|"]
    for row in rows:
        result = row["panels"][FULL_ID]
        metric = result["metrics"]
        lines.append(f"| `{row['prediction_id']}` | {result['n_with_prediction']} / 3966 | {result['coverage']:.10f} | `{result['status']}` | {metric['micro_f1']:.10f} | {metric['macro_f1']:.10f} |" if metric else
                     f"| `{row['prediction_id']}` | 0 / 3966 | 0.0000000000 | `{result['status']}` | UNCHECKED | UNCHECKED |")
    release = next(row for row in rows if row["prediction_id"] == RELEASE_ID)["panels"][FULL_ID]
    lines += ["", f"全集唯一全覆盖身份为 `{RELEASE_ID}`（ensemble），micro-F1={release['metrics']['micro_f1']:.16f}，macro-F1={release['metrics']['macro_f1']:.16f}，empty/empty={release['metrics']['empty_empty_records']}。全集没有其他全覆盖单预测器，因此最强单预测器为 `UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR`，不以部分覆盖排名替代。", "",
              "## 阻塞与边界", "",
              "R2D seed 0 全长 raw 仍缺失；first-598 资产标为 `BLOCKED_PROTOCOL_MISMATCH_FIRST598_NOT_FULL_LENGTH`，未冒充全长预测。seed 1/2 是独立 seed 的全长冻结资产，不用于替代 seed 0。", "",
              "其余 summary-only 或缺失身份均保持 blocked、metrics=null、不补 0。完整清单见 `system_metrics.json` 和 `registry_recommendation.json`。", ""]
    return "\n".join(lines)


def registry_recommendation(document):
    return {
        "schema_version": "archiveii.official_reference.registry_recommendation.v1",
        "recommendation_only": True, "apply_by_unified_process_only": True, "shared_registry_modified": False,
        "legacy_project_labels": {
            "dataset_id": "archiveii_embok_clean", "label_source": "DERIVED_FROM_BPFOLD_PREDICTIONS",
            "disposition": "circular-audit-only", "independent_sota_eligible": False,
            "historical_numeric_values_retained": True,
        },
        "suggested_data_rows": [document["label_panels"][FULL_ID], document["label_panels"][CLEAN_ID]],
        "suggested_prediction_overlay": {
            "path": "artifacts/archiveii_official_reference_20261127/system_metrics.json",
            "sha256": sha256(OUT / "system_metrics.json"),
            "rows": document["rows"],
        },
        "suggested_baseline_overlay": {
            "scope": "descriptive frozen official-reference audit only; not model selection",
            "strongest_clean_2544": {key: document[key] for key in ("strongest_overall", "strongest_ensemble", "strongest_single_predictor")},
            "official_full_3966": {
                "only_full_coverage_identity": RELEASE_ID,
                "strongest_single_predictor": "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR",
            },
        },
        "identity_boundaries": {
            "bpfold_official_release/archiveii": "author release BPSEQ predictions",
            "bpfold/archiveii_embok_clean": "project six-checkpoint replay predictions",
            "rinalmo_2d_s0/archiveii_embok_clean": "full-length seed0 remains blocked",
            TRUNC_ID: "different first-598 input protocol; not a full-length substitute",
        },
    }


def main():
    registry_paths = [ROOT / f"manifests/foldnar_v1/{name}.json" for name in ("data_registry", "prediction_registry", "baseline_registry")]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in registry_paths}
    full, clean = build_panels()
    entries = inventory()
    rows, record_rows, strongest = evaluate(entries, full, clean)
    document = {
        "schema_version": SCHEMA, "audit_date": "2026-11-27",
        "scope": "CPU-only frozen-prediction rescoring on unfiltered official references; no inference, training, threshold changes or model selection",
        "label_panels": {
            FULL_ID: panel_identity(full, SOURCE / "official_reference_3966.jsonl.gz", FULL_ID),
            CLEAN_ID: panel_identity(clean, SOURCE / "official_reference_clean_2544.jsonl.gz", CLEAN_ID),
        },
        "metric_contract": {
            "pair_matching": "strict zero-based inclusive exact pairs; no additional crossing/canonical/min-loop filtering",
            "micro": "pool TP/FP/FN; zero-denominator precision or recall=0",
            "macro": "unweighted mean per-sequence P/R/F1; P=R=F1=1 only when truth and prediction are both empty",
            "ranking": "micro-F1, full coverage within the stated panel only; descriptive audit, not model selection",
            "missing_predictions": "score present records only for coverage accounting; never impute missing as zero; partial coverage not ranked",
            "paired_ci": "10000 paired sequence-level percentile bootstraps of macro-F1 delta versus xens2 on clean 2544; seed 20261127",
        },
        "coverage_accounting": {
            "inventory_entries": len(rows), "scored_identities": sum(row["available"] for row in rows),
            "blocked_or_protocol_mismatch": sum(not row["available"] for row in rows),
            "clean_2544_full_coverage": sum(row["panels"][CLEAN_ID]["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "official_3966_full_coverage": sum(row["panels"][FULL_ID]["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "official_3966_partial_not_ranked": sum(row["panels"][FULL_ID]["status"] == "PARTIAL_COVERAGE_NOT_RANKED" for row in rows),
        },
        **strongest,
        "official_3966_full_coverage_summary": {
            "only_full_coverage_identity": RELEASE_ID,
            "strongest_single_predictor": "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR",
        },
        "xens2_reference": next(row for row in rows if row["prediction_id"] == XENS2),
        "legacy_derived_label_disposition": "circular-audit-only; historical values retained; not independent SOTA",
        "rows": rows,
        "external_test_model_selection": False, "new_training_or_inference": False, "gpu_used": False,
        "shared_registry_sha256_before": before,
    }
    write_gzip_jsonl(OUT / "per_record_metrics.jsonl.gz", record_rows)
    write_json(OUT / "system_metrics.json", document)
    write_json(OUT / "registry_recommendation.json", registry_recommendation(document))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(document), encoding="utf-8")
    after = {str(path.relative_to(ROOT)): sha256(path) for path in registry_paths}
    if before != after:
        raise ValueError("shared registry modified")
    source_manifest = {
        "schema_version": SCHEMA + ".source_manifest", "official_archives": [
            {"role": "official reference BPfold_data", "path": str(BD), "sha256": BD_SHA},
            {"role": "official reference MXfold2 mirror", "path": str(MX), "sha256": MX_SHA},
            {"role": "author predicted structures; not labels", "path": str(BR), "sha256": BR_SHA},
            {"role": "legacy prediction-derived project labels", "path": str(PROJECT), "sha256": PROJECT_SHA},
        ],
        "prediction_inputs": [{"prediction_id": row["prediction_id"], "path": row["path"], "sha256": row["sha256"],
                               "available": row["available"]} for row in rows],
        "shared_registry_sha256_before_after": after,
        "shared_registry_modified": False, "new_training_or_inference": False,
        "gpu_used": False, "external_test_model_selection": False,
    }
    write_json(SOURCE / "source_manifest.json", source_manifest)
    write_json(OUT / "build_receipt.json", {
        "schema_version": SCHEMA + ".build_receipt", "status": "PASS",
        "per_record_metric_rows": len(record_rows), "prediction_identities": len(rows),
        "outputs": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in (
            SOURCE / "official_reference_3966.jsonl.gz", SOURCE / "official_reference_clean_2544.jsonl.gz",
            SOURCE / "source_manifest.json", OUT / "per_record_metrics.jsonl.gz", OUT / "system_metrics.json",
            OUT / "registry_recommendation.json", REPORT)},
    })
    print(json.dumps({"status": "PASS", "coverage": document["coverage_accounting"],
                      "strongest_clean": strongest, "per_record_metric_rows": len(record_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
