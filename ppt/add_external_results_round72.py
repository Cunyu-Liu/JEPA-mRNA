#!/usr/bin/env python3
"""Append external raw and long-sequence slides after a verified round-71 slide 34."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
MARGIN_VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round71_build_verification.json"
STRUCT_RESULTS = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structrfm_official_v0.0.8/reports/structrfm_official_bprna_new_metrics_20261126.json")
STRUCT_REGISTRY = ROOT / "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json"
RAW_REGISTRY = ROOT / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
BASELINE_REGISTRY = ROOT / "manifests/foldnar_v1/baseline_registry.json"
LONGSEQ_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126")
BURGUNDY = RGBColor(122, 32, 44)
DARK_RED = RGBColor(145, 36, 48)
GREEN = RGBColor(36, 105, 78)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
PASS_PALE = RGBColor(235, 247, 240)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(222, 192, 184)
SANS = "Source Han Sans SC"
SERIF = "Source Han Serif SC"
SPANS = ("0-49", "50-199", "200-999", ">=1000")


class EligibilityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EligibilityError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: str | Path) -> dict:
    path = Path(path)
    require(path.is_file(), f"required frozen JSON missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EligibilityError(f"invalid frozen JSON {path}: {error}") from error
    require(isinstance(value, dict), f"frozen JSON must be an object: {path}")
    return value


def require_identity(record: dict, path: str | Path, label: str) -> dict:
    path = Path(path).resolve()
    actual = identity(path)
    require(record.get("path") == str(path), f"{label} path mismatch")
    require(record.get("bytes") == actual["bytes"] and record.get("sha256") == actual["sha256"],
            f"{label} byte/hash mismatch")
    return actual


def finite_metric(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"invalid metric: {label}")
    return float(value)


def slide_text(slide) -> str:
    values = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            values.append(shape.text_frame.text)
        if shape.has_table:
            values.extend(cell.text for row in shape.table.rows for cell in row.cells)
    return "\n".join(values)


def validate_margin(deck: str | Path, verification_path: str | Path) -> dict:
    deck = Path(deck).resolve()
    verification_path = Path(verification_path).resolve()
    report = load_json(verification_path)
    require(deck.is_file(), f"margin deck missing: {deck}")
    prs = Presentation(deck)
    require(len(prs.slides) == 15, "round72 requires exactly 15 slides including real margin page 34")
    require(report.get("status") == "PASS_BUILD_REOPEN_14_PRIOR_XML_ATOMIC_REPLACE_HASH_GATE",
            "margin page 34 lacks a successful round71 hash-gated build report")
    require(report.get("slide_count") == 15 and report.get("deck_slide_index") == 15
            and report.get("added_slide_number") == 34
            and report.get("prior_slide_xml_preserved") == 14
            and report.get("atomic_replace") is True,
            "margin page 34 build geometry/order contract is not satisfied")
    require(Path(report.get("deck", "")).resolve() == deck,
            "round71 verification belongs to a different deck")
    deck_identity = identity(deck)
    require(report.get("deck_sha256") == deck_identity["sha256"]
            and report.get("deck_bytes") == deck_identity["bytes"],
            "margin deck SHA-256/bytes differ from the round71 verification")
    require(report.get("scientific_status") in {
        "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL", "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"},
        "margin page 34 is not a final scientific result")
    builder_path = Path(report.get("builder", ""))
    require(builder_path.name == "add_phase0_structured_margin_result_round71.py"
            and builder_path.is_file(), "round71 builder identity is missing")
    require(sha256(builder_path) == report.get("builder_sha256"),
            "round71 builder hash differs from the page34 build report")
    backup = Path(report.get("backup", "")).resolve()
    require(backup.is_file() and backup != deck, "round71 14-slide backup is missing")
    require(sha256(backup) == report.get("backup_sha256") == report.get("prior_deck_sha256"),
            "round71 source backup hash mismatch")
    require(len(Presentation(backup).slides) == 14, "round71 source backup is not 14 slides")
    with zipfile.ZipFile(backup) as before, zipfile.ZipFile(deck) as after:
        for index in range(1, 15):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"margin build changed prior XML: {part}")
    margin_text = slide_text(prs.slides[14])
    require("34 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT" in margin_text
            and "MODEL-DEV ONLY" in margin_text and report["scientific_status"] in margin_text,
            "deck page 34 is not the verified structured-margin result")
    final_identities = []
    for path_text, recorded in report.get("final_json", {}).items():
        final_identities.append(require_identity(recorded, path_text, "margin final JSON"))
    require(len(final_identities) == 3, "margin final evaluation/summary/manifest identities incomplete")
    return {"deck": deck, "verification_path": verification_path, "report": report,
            "deck_identity": deck_identity, "backup": backup, "final_identities": final_identities}


def validate_external_raw(struct_results_path: str | Path, struct_registry_path: str | Path,
                          raw_registry_path: str | Path, baseline_registry_path: str | Path) -> dict:
    struct_results_path = Path(struct_results_path).resolve()
    struct_registry_path = Path(struct_registry_path).resolve()
    raw_registry_path = Path(raw_registry_path).resolve()
    baseline_registry_path = Path(baseline_registry_path).resolve()
    official = load_json(struct_results_path)
    registered_official = load_json(struct_registry_path)
    raw = load_json(raw_registry_path)
    baseline = load_json(baseline_registry_path)
    require(official == registered_official, "official structRFM artifact and frozen registry copy differ")
    require(official.get("status") == "PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS"
            and official.get("full_ready") is True
            and official.get("prediction_id") == "structrfm_official_v0.0.8/bprna_new"
            and official.get("raw_not_clean") is True
            and official.get("summary_only_substitution") is False
            and official.get("label_scope") == "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED",
            "structRFM result is not the complete official raw 5388 artifact")
    require_identity(official["predictions"], official["predictions"]["path"], "structRFM predictions")
    metrics = official.get("metrics", {})
    require(metrics.get("n_dataset_sequences") == metrics.get("n_with_prediction")
            == official.get("dataset", {}).get("n_sequences") == 5388
            and metrics.get("coverage") == 1.0 and not metrics.get("missing_names")
            and not metrics.get("extra_names"), "structRFM raw coverage is incomplete")
    for name in ("micro_f1", "macro_f1"):
        finite_metric(metrics.get(name), f"structRFM {name}")
    rows = {row.get("asset_id"): row for row in raw.get("rows", [])}
    asset_ids = ("bpfold_project_six_checkpoint_raw", "contrafold_mea_single_raw")
    comparators = []
    for asset_id in asset_ids:
        row = rows.get(asset_id, {})
        require(row.get("status") == "PASS_SHA_ID_METRICS"
                and row.get("raw_clean_scope") == "raw"
                and row.get("label_protocol") == "PROJECT_DATABASE_RAW_5388"
                and row.get("n_sequences") == 5388
                and row.get("label_sha256") == official["dataset"]["sha256"]
                and row.get("dataset_id") == "bprna_new"
                and row.get("metrics", {}).get("coverage") == 1.0,
                f"raw comparator scope/coverage mismatch: {asset_id}")
        require(row.get("prediction_sha256") == sha256(row["prediction_path"]),
                f"raw comparator prediction hash mismatch: {asset_id}")
        comparators.append(row)
    for key, row in (("strongest_project_raw", comparators[0]),
                     ("strongest_project_raw_single_predictor", comparators[1])):
        summary = raw.get(key, {})
        require(summary.get("asset_id") == row["asset_id"]
                and summary.get("micro_f1") == row["metrics"]["micro_f1"]
                and summary.get("macro_f1") == row["metrics"]["macro_f1"],
                f"registry summary differs from raw row: {key}")
    xens2 = raw.get("xens2_project_raw_reference", {})
    for name in ("micro_f1", "macro_f1"):
        finite_metric(xens2.get(name), f"xens2 {name}")
    bp_metrics = comparators[0]["metrics"]
    require(xens2.get("gap_to_strongest_ensemble_micro")
            == bp_metrics["micro_f1"] - xens2["micro_f1"]
            and xens2.get("gap_to_strongest_ensemble_macro")
            == bp_metrics["macro_f1"] - xens2["macro_f1"],
            "xens2 raw reference gaps differ from frozen registry metrics")
    require(raw.get("scope_contract", {}).get("raw_is_not_clean") is True,
            "raw registry no longer states raw_is_not_clean")
    baseline_rows = {row.get("baseline_id"): row for row in baseline.get("rows", [])}
    for baseline_id in ("structrfm", "bpfold", "contrafold", "xens2"):
        comparability = baseline_rows.get(baseline_id, {}).get("comparability", {})
        require("raw is not clean" in comparability.get("status", "")
                and comparability.get("clean_subset_policy") == "baseline-specific; no shared TR1c-clean assumed",
                f"raw/clean boundary missing for {baseline_id}")
    struct_baseline = baseline_rows["structrfm"]
    require(struct_baseline.get("official_weights", {}).get("sha256")
            == official["execution"]["checkpoint_sha256"]
            and struct_baseline.get("status") == "OFFICIAL_V008_RAW_BPRNA_NEW_COMPLETE",
            "structRFM official checkpoint/registry identity mismatch")
    return {"official": official, "comparators": comparators, "xens2": xens2,
            "struct_results_path": struct_results_path,
            "paths": (struct_results_path, struct_registry_path, raw_registry_path,
                      baseline_registry_path, Path(official["predictions"]["path"]),
                      *(Path(row["prediction_path"]) for row in comparators))}


def validate_longseq(root: str | Path) -> dict:
    root = Path(root).resolve()
    train_summary_path = root / "runs/linearpartition_c_7224_train_cpu_20261126/summary.json"
    train_delivery_path = root / "runs/linearpartition_c_7224_train_cpu_20261126/pilot_delivery_verification.json"
    holdout_root = root / "runs/linearpartition_c_19000_holdout_cpu_20261126"
    holdout_delivery_path = holdout_root / "holdout_delivery_verification.json"
    train = load_json(train_summary_path)
    train_delivery = load_json(train_delivery_path)
    holdout_delivery = load_json(holdout_delivery_path)
    require_identity(train_delivery.get("summary", {}), train_summary_path, "7224 summary")
    require(train.get("status") == "PASS_TWO_FULL_LENGTH_TRAIN_ONLY_CPU_PILOTS"
            and train.get("train_only") is True
            and train.get("independent_scientific_evaluation") is False
            and train.get("execution_column") == "APPROXIMATE_SPARSE"
            and len(train.get("chains", [])) == 2,
            "7224 train-only pilot contract changed")
    chains = []
    verified_paths = [train_summary_path, train_delivery_path, holdout_delivery_path]
    for chain in train["chains"]:
        require(chain.get("full_length") == 7224 and chain.get("split") == "train"
                and chain.get("train_only") is True
                and chain.get("independent_scientific_evaluation") is False
                and chain.get("execution_column") == "APPROXIMATE_SPARSE"
                and chain.get("status") == "PASS_INDEPENDENT_RAW_OUTPUT_HASH_RESOURCE_AND_METRIC_VERIFICATION",
                f"7224 chain boundary changed: {chain.get('name')}")
        run_report_path = train_summary_path.parent / chain["name"] / "run_report.json"
        independent_path = train_summary_path.parent / chain["name"] / "independent_verification.json"
        require(sha256(run_report_path) == chain.get("run_report_sha256"),
                f"7224 run report hash changed: {chain['name']}")
        independent = load_json(independent_path)
        require(independent.get("recomputed") == chain.get("recomputed")
                and independent.get("run_report_sha256") == chain.get("run_report_sha256")
                and independent.get("status") == chain.get("status"),
                f"7224 independent recomputation differs: {chain['name']}")
        chains.append(chain)
        verified_paths.extend((run_report_path, independent_path))
    holdout = load_json(holdout_root / "7AM2_1/run_report.json")
    holdout_metrics_path = holdout_root / "7AM2_1/metrics.json"
    holdout_independent_path = holdout_root / "7AM2_1/independent_verification.json"
    metrics = load_json(holdout_metrics_path)
    independent = load_json(holdout_independent_path)
    require_identity(holdout_delivery.get("run_report", {}), holdout_root / "7AM2_1/run_report.json",
                     "19000 run report")
    require_identity(holdout_delivery.get("metrics", {}), holdout_metrics_path, "19000 metrics")
    require_identity(holdout_delivery.get("independent_verification", {}), holdout_independent_path,
                     "19000 independent verification")
    require(holdout_delivery.get("status") == "PASS_FULL_19000_HOLDOUT_CPU_PILOT_AND_INDEPENDENT_VERIFICATION"
            and holdout_delivery.get("coverage") == 1.0
            and holdout_delivery.get("truncated") is False
            and holdout_delivery.get("holdout_descriptive_only") is True
            and holdout_delivery.get("not_for_model_selection") is True
            and holdout_delivery.get("execution_column") == "APPROXIMATE_SPARSE",
            "19000 holdout delivery boundary changed")
    require(holdout.get("input", {}).get("length") == 19000
            and holdout.get("split") == "holdout" and holdout.get("gpu_execution") is False
            and holdout.get("execution", {}).get("variant") == "LinearPartition-C"
            and holdout.get("execution", {}).get("column") == "APPROXIMATE_SPARSE"
            and holdout.get("execution", {}).get("beam_size") == 100
            and holdout.get("execution", {}).get("bpp_cutoff") == 0.0
            and holdout.get("holdout_descriptive_only") is True
            and holdout.get("not_for_model_selection") is True,
            "19000 execution column or scope changed")
    recomputed = independent.get("recomputed", {})
    require(independent.get("status") == "PASS_INDEPENDENT_HOLDOUT_OUTPUT_HASH_RESOURCE_AND_METRIC_VERIFICATION"
            and independent.get("run_report_sha256") == sha256(holdout_root / "7AM2_1/run_report.json")
            and recomputed.get("bpp_candidate_pair_recall") == metrics.get("bpp_candidate_pair_recall")
            and recomputed.get("mea_pair_recall") == metrics.get("mea_pair_recall")
            and recomputed.get("coverage") == metrics.get("coverage")
            and recomputed.get("wall_seconds") == holdout["resources"]["latency"]["wall_seconds"]
            and recomputed.get("peak_cpu_rss_bytes") == holdout["resources"]["memory"]["peak_cpu_rss_bytes"],
            "19000 independent metrics/resources differ from frozen raw outputs")
    chains.append({"name": holdout["name"], "full_length": 19000, "split": "holdout",
                   "train_only": False, "independent_scientific_evaluation": False,
                   "execution_column": "APPROXIMATE_SPARSE", "recomputed": recomputed})
    for chain in chains:
        recomputed = chain["recomputed"]
        require(recomputed.get("coverage", {}).get("completed_nucleotide_coverage") == 1.0
                and recomputed.get("coverage", {}).get("output_nucleotides") == chain["full_length"]
                and recomputed.get("coverage", {}).get("truncated") is False
                and recomputed.get("flops_status") == "NOT_MEASURED",
                f"long-sequence coverage/FLOPs contract changed: {chain['name']}")
        for metric in ("bpp_candidate_pair_recall", "mea_pair_recall"):
            require(set(recomputed.get(metric, {})) == set(SPANS) | {"overall"},
                    f"span buckets changed: {chain['name']} {metric}")
    verified_paths.extend((holdout_root / "7AM2_1/run_report.json", holdout_metrics_path,
                           holdout_independent_path))
    return {"chains": chains, "paths": verified_paths,
            "holdout_execution": holdout["execution"], "holdout_resources": holdout["resources"]}


def fmt(value: float) -> str:
    return f"{value:.6f}"


def signed(value: float) -> str:
    return f"{value:+.6f}"


def text(slide, x, y, w, h, value, size, color=INK, bold=False, font="Arial", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = str(value)
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def table(slide, x, y, widths, row_height, headers, rows, font_size=9.0, emphasis_row=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_height * (len(rows) + 1)))
    tbl = shape.table
    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)
    for row_index in range(len(rows) + 1):
        tbl.rows[row_index].height = Inches(row_height)
        values = headers if row_index == 0 else rows[row_index - 1]
        for column_index, value in enumerate(values):
            cell = tbl.cell(row_index, column_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (BURGUNDY if row_index == 0 else
                                        PASS_PALE if row_index == emphasis_row else
                                        ROSE if row_index % 2 == 0 else WHITE)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.04)
            cell.margin_top = cell.margin_bottom = Inches(0.005)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
                paragraph.space_before = paragraph.space_after = Pt(0)
                paragraph.font.name = SANS if column_index == 0 else "Arial"
                paragraph.font.size = Pt(font_size)
                paragraph.font.bold = row_index == 0 or row_index == emphasis_row
                paragraph.font.color.rgb = WHITE if row_index == 0 else INK
    return tbl


def new_slide(prs: Presentation, number: int, section: str, title: str, subtitle: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.25, 11.8, 0.28, f"{number} · {section}", 10.5, BURGUNDY, True)
    text(slide, 0.78, 0.63, 11.9, 0.52, title, 25, INK, True, SERIF)
    text(slide, 0.78, 1.20, 11.9, 0.30, subtitle, 10.5, MUTED, font=SANS)
    return slide


def add_external_slide(prs: Presentation, bundle: dict) -> None:
    official = bundle["official"]
    sm = official["metrics"]
    rows = [["structRFM official v0.0.8 · single", fmt(sm["micro_f1"]), "reference",
             fmt(sm["macro_f1"]), "reference"]]
    labels = (("BPfold · six-checkpoint ensemble", bundle["comparators"][0]["metrics"]),
              ("CONTRAfold · deterministic MEA", bundle["comparators"][1]["metrics"]),
              ("xens2 · frozen project ensemble", bundle["xens2"]))
    for label, metrics in labels:
        rows.append([label, fmt(metrics["micro_f1"]), signed(sm["micro_f1"] - metrics["micro_f1"]),
                     fmt(metrics["macro_f1"]), signed(sm["macro_f1"] - metrics["macro_f1"])])
    slide = new_slide(prs, 35, "FOLDNAR EXTERNAL RESULTS / UNIFIED RAW COMPARISON",
                      "structRFM官方结果：放回统一raw绝对差距语境",
                      f"PROJECT_RAW_GT_ONLY · bpRNA-new {sm['n_dataset_sequences']:,} / coverage {sm['coverage']:.3f} · Δ = structRFM − 对照（绝对F1差）")
    table(slide, 0.78, 1.65, (4.18, 1.80, 2.00, 1.80, 2.00), 0.46,
          ("同一raw标签域", "micro-F1", "Δ micro", "macro-F1", "Δ macro"), rows,
          font_size=10.2, emphasis_row=1)
    rect(slide, 0.78, 4.18, 11.78, 0.90, PALE, BURGUNDY)
    text(slide, 1.00, 4.29, 11.32, 0.28, "RAW ≠ CLEAN ≠ SOTA", 17, DARK_RED, True)
    text(slide, 1.00, 4.62, 11.30, 0.26,
         "仅描述同一PROJECT_RAW 5388标签域的绝对差距；不去污、不外推clean、不宣称SOTA，也不把六checkpoint集成当单模型。",
         10.3, INK, True, SANS)
    rect(slide, 0.78, 5.30, 5.72, 0.84, PASS_PALE, GREEN)
    text(slide, 1.00, 5.40, 5.28, 0.24, "官方身份已闭环", 12, GREEN, True, SANS)
    text(slide, 1.00, 5.69, 5.28, 0.27,
         f"v0.0.8 · checkpoint {official['execution']['checkpoint_sha256'][:12]} · official MixedFold constrained DP",
         8.7, INK)
    rect(slide, 6.74, 5.30, 5.82, 0.84, ROSE, LINE)
    text(slide, 6.96, 5.40, 5.38, 0.24, "协议差异保留", 12, BURGUNDY, True, SANS)
    text(slide, 6.96, 5.69, 5.38, 0.27,
         "BPfold六checkpoint / CONTRAfold单预测器 / xens2项目集成；模型形式不作等同。",
         8.9, INK, font=SANS)
    text(slide, 0.78, 6.46, 11.70, 0.18,
         f"official metrics SHA-256 {sha256(bundle['struct_results_path'])[:12]} · raw registry + baseline registry逐行读取 · raw_not_clean=true",
         7.7, MUTED)
    text(slide, 0.78, 6.72, 11.70, 0.18,
         "structRFM作者训练语料含bpRNA1m；bpRNA-new暴露未清除，故本页不构成独立外部测试或clean SOTA证据。",
         7.7, MUTED, font=SANS)


def recall_cell(metric: dict) -> str:
    overall = metric["overall"]
    return f"{overall['true_positive']}/{overall['reference']} · {overall['recall'] * 100:.2f}%"


def span_cell(bpp: dict, mea: dict, span: str) -> str:
    b, m = bpp[span], mea[span]
    if b["reference"] == 0 and m["reference"] == 0:
        return "n=0 · —"
    require(b["reference"] == m["reference"] and b["recall"] is not None and m["recall"] is not None,
            "non-empty span bucket has incompatible references")
    return f"{b['recall'] * 100:.1f}% / {m['recall'] * 100:.1f}% · n={b['reference']}"


def add_longseq_slide(prs: Presentation, bundle: dict) -> None:
    rows = []
    for chain in bundle["chains"]:
        metrics = chain["recomputed"]
        role = "train-only · 非独立评估" if chain["train_only"] else "holdout · 描述性"
        resources = f"{metrics['wall_seconds']:.2f} s · {metrics['peak_cpu_rss_bytes']:,} B RSS"
        rows.append([chain["name"], role, f"{chain['full_length']:,} nt", "1.000 · full",
                     recall_cell(metrics["bpp_candidate_pair_recall"]),
                     recall_cell(metrics["mea_pair_recall"]), resources])
    slide = new_slide(prs, 36, "FOLDNAR EXTERNAL RESULTS / LONG-SEQUENCE COMPUTE BOUNDARY",
                      "LinearPartition长序列边界：能跑完，不等于精确全局DP",
                      "LinearPartition-C · serial CPU · beam 100 / cutoff 0.0 · APPROXIMATE_SPARSE · BPP候选对 / MEA结构")
    table(slide, 0.78, 1.62, (1.10, 1.70, 0.82, 1.02, 1.70, 1.60, 3.84), 0.48,
          ("target", "数据角色", "全长", "coverage", "BPP overall recall", "MEA overall recall", "wall · peak CPU RSS"),
          rows, font_size=8.7)
    text(slide, 0.78, 3.72, 11.70, 0.20,
         "分span recall：每格 BPP / MEA；span=j−i（zero-based inclusive）；空桶为null，不当作0。",
         9.2, MUTED, font=SANS)
    span_rows = []
    for chain in bundle["chains"]:
        metrics = chain["recomputed"]
        span_rows.append([chain["name"]] + [span_cell(metrics["bpp_candidate_pair_recall"],
                                                     metrics["mea_pair_recall"], span) for span in SPANS])
    table(slide, 0.78, 4.00, (1.58, 2.55, 2.55, 2.55, 2.55), 0.42,
          ("target", "0–49", "50–199", "200–999", "≥1000"), span_rows, font_size=8.9)
    rect(slide, 0.78, 5.88, 11.78, 0.54, PALE, BURGUNDY)
    text(slide, 0.98, 5.96, 11.38, 0.34,
         "APPROXIMATE_SPARSE：beam/cutoff保留候选对；未出现的pair不是精确0，不能归入EXACT_GLOBAL_DP。FLOPs NOT_MEASURED。",
         9.8, DARK_RED, True, SANS)
    text(slide, 0.78, 6.18, 11.70, 0.22,
         "target IDs · 9MR4_5 · 9 Q 7 Q _ 5 · 7AM2_1", 13, INK, True)
    text(slide, 0.78, 6.53, 11.70, 0.18,
         "7224仅train-only工程边界，不作无偏科学评估；19000 holdout仅描述性记录，不用于模型选择；GEC长序列支持NOT_ESTABLISHED。",
         7.7, MUTED, font=SANS)
    text(slide, 0.78, 6.76, 11.70, 0.18,
         "coverage / recall / GNU time RSS均从冻结raw输出与independent_verification重算记录读取；未以短链外推资源。",
         7.7, MUTED, font=SANS)


def verify_built_deck(path: Path, prior_deck: Path, margin_backup: Path,
                      external: dict, longseq: dict) -> dict:
    reopened = Presentation(path)
    prior = Presentation(prior_deck)
    require(len(prior.slides) == 15 and len(reopened.slides) == 17,
            "round72 output must contain exactly 17 slides")
    with zipfile.ZipFile(prior_deck) as before, zipfile.ZipFile(path) as after:
        for index in range(1, 16):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"prior slide part changed: {part}")
    with zipfile.ZipFile(margin_backup) as before, zipfile.ZipFile(path) as after:
        for index in range(1, 15):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"original Phase0 XML changed: {part}")
    for slide in list(reopened.slides)[15:]:
        for shape in slide.shapes:
            require(shape.left >= 0 and shape.top >= 0, "negative shape position")
            require(shape.left + shape.width <= reopened.slide_width + 2, "shape exceeds slide width")
            require(shape.top + shape.height <= reopened.slide_height + 2, "shape exceeds slide height")
    external_text = slide_text(reopened.slides[15])
    sm = external["official"]["metrics"]
    for phrase in ("35 · FOLDNAR EXTERNAL RESULTS", "structRFM官方结果", fmt(sm["micro_f1"]),
                   fmt(sm["macro_f1"]), "BPfold", "CONTRAfold", "xens2", "RAW ≠ CLEAN ≠ SOTA",
                   "不去污", "不宣称SOTA", "绝对差距"): 
        require(phrase in external_text, f"missing external slide text: {phrase}")
    for metrics in [row["metrics"] for row in external["comparators"]] + [external["xens2"]]:
        require(fmt(metrics["micro_f1"]) in external_text and fmt(metrics["macro_f1"]) in external_text,
                "raw comparator metrics missing")
        require(signed(sm["micro_f1"] - metrics["micro_f1"]) in external_text
                and signed(sm["macro_f1"] - metrics["macro_f1"]) in external_text,
                "raw absolute gap missing")
    long_text = slide_text(reopened.slides[16])
    for phrase in ("36 · FOLDNAR EXTERNAL RESULTS", "APPROXIMATE_SPARSE", "EXACT_GLOBAL_DP",
                   "train-only", "holdout", "FLOPs NOT_MEASURED", "NOT_ESTABLISHED",
                   "不用于模型选择", "span=j−i"):
        require(phrase in long_text, f"missing long-sequence boundary: {phrase}")
    for chain in longseq["chains"]:
        metrics = chain["recomputed"]
        require(chain["name"] in long_text and f"{chain['full_length']:,} nt" in long_text,
                "long target missing")
        require(recall_cell(metrics["bpp_candidate_pair_recall"]) in long_text
                and recall_cell(metrics["mea_pair_recall"]) in long_text
                and f"{metrics['wall_seconds']:.2f} s" in long_text
                and f"{metrics['peak_cpu_rss_bytes']:,} B RSS" in long_text,
                f"long target metrics/resources missing: {chain['name']}")
        for span in SPANS:
            require(span_cell(metrics["bpp_candidate_pair_recall"], metrics["mea_pair_recall"], span)
                    in long_text, f"span recall missing: {chain['name']} {span}")
    return {"slide_count": 17, "deck_slide_indices": [16, 17], "added_slide_numbers": [35, 36],
            "prior_slide_xml_preserved": 15, "original_phase0_xml_preserved": 14,
            "shape_bounds_checks": sum(len(slide.shapes) for slide in list(reopened.slides)[15:]),
            "new_slide_text_chars": [len(external_text), len(long_text)]}


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(deck: str | Path = DECK, margin_verification: str | Path = MARGIN_VERIFICATION,
          backup: str | Path | None = None, verification: str | Path | None = None,
          struct_results: str | Path = STRUCT_RESULTS, struct_registry: str | Path = STRUCT_REGISTRY,
          raw_registry: str | Path = RAW_REGISTRY, baseline_registry: str | Path = BASELINE_REGISTRY,
          longseq_root: str | Path = LONGSEQ_ROOT) -> dict:
    deck = Path(deck).resolve()
    backup = Path(backup).resolve() if backup else deck.with_name(deck.stem + ".pre_round72.pptx")
    verification = (Path(verification).resolve() if verification else
                    deck.with_name(deck.stem + ".round72_build_verification.json"))
    margin = validate_margin(deck, margin_verification)
    external = validate_external_raw(struct_results, struct_registry, raw_registry, baseline_registry)
    longseq = validate_longseq(longseq_root)
    require(not backup.exists(), f"refusing to overwrite existing backup: {backup}")
    require(not verification.exists(), f"refusing to overwrite existing verification: {verification}")
    prs = Presentation(deck)
    prior_sha256 = sha256(deck)
    add_external_slide(prs, external)
    add_longseq_slide(prs, longseq)
    descriptor, temporary = tempfile.mkstemp(prefix=deck.stem + ".round72.", suffix=".pptx", dir=deck.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        prs.save(temporary_path)
        layout = verify_built_deck(temporary_path, deck, margin["backup"], external, longseq)
        require(sha256(deck) == prior_sha256, "input deck changed before atomic replacement")
        shutil.copy2(deck, backup)
        os.replace(temporary_path, deck)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    reopened_sha256 = sha256(deck)
    report = {
        "status": "PASS_BUILD_REOPEN_15_PRIOR_XML_SEQUENTIAL_ATOMIC_REPLACE_HASH_GATE",
        "margin_scientific_status": margin["report"]["scientific_status"],
        "deck": str(deck), "deck_sha256": reopened_sha256, "deck_bytes": deck.stat().st_size,
        "prior_deck_sha256": prior_sha256, "backup": str(backup), "backup_sha256": sha256(backup),
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve()),
        "margin_verification": identity(margin["verification_path"]),
        "margin_deck_sha256_verified": margin["deck_identity"]["sha256"],
        "margin_final_json": margin["final_identities"],
        "external_artifacts": [identity(path) for path in external["paths"]],
        "longseq_artifacts": [identity(path) for path in longseq["paths"]],
        "scope": {"raw_not_clean_not_sota": True, "external_raw_n": 5388,
                  "longseq_execution_column": "APPROXIMATE_SPARSE",
                  "train_only_not_unbiased_evaluation": True,
                  "holdout_descriptive_not_model_selection": True,
                  "exact_global_dp_claim": False, "flops": "NOT_MEASURED"},
        "atomic_replace": True, "sequential_append": [35, 36],
        **layout, "render_visual_check": "PENDING_SEPARATE_RENDER_AND_OCR",
    }
    atomic_write_json(verification, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--margin-verification", type=Path, default=MARGIN_VERIFICATION)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--struct-results", type=Path, default=STRUCT_RESULTS)
    parser.add_argument("--struct-registry", type=Path, default=STRUCT_REGISTRY)
    parser.add_argument("--raw-registry", type=Path, default=RAW_REGISTRY)
    parser.add_argument("--baseline-registry", type=Path, default=BASELINE_REGISTRY)
    parser.add_argument("--longseq-root", type=Path, default=LONGSEQ_ROOT)
    args = parser.parse_args(argv)
    try:
        report = build(args.deck, args.margin_verification, args.backup, args.verification,
                       args.struct_results, args.struct_registry, args.raw_registry,
                       args.baseline_registry, args.longseq_root)
    except (EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"round72 build refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
