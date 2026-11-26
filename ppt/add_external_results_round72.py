#!/usr/bin/env python3
"""Append external raw and long-sequence slides after a verified round-71 slide 34."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
MARGIN_INDEPENDENT_VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round71_independent_build_verification.json"
PANEL_REGISTRY = ROOT / "manifests/foldnar_v1/bprna_new_official_reference_5388_prediction_registry.json"
PANEL_SYSTEM_METRICS = ROOT / "source_data/bprna_official_reference_5388/system_metrics.json"
PANEL_INDEPENDENT_VERIFICATION = ROOT / "source_data/bprna_official_reference_5388/independent_verification.json"
PANEL_LABELS = ROOT / "source_data/bprna_official_reference_5388/official_reference_5388.jsonl.gz"
PANEL_COMMIT = "e7ff4e76fe078178ab41d2052349627317b5a3e3"
PANEL_SCHEMA = "bprna.official_reference_5388.prediction_registry.v1"
PANEL_DATASET = "bpfold_official_bprna_new_reference_5388"
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


def validate_margin(deck: str | Path, verification_path: str | Path,
                    independent_verification_path: str | Path = MARGIN_INDEPENDENT_VERIFICATION) -> dict:
    deck = Path(deck).resolve()
    verification_path = Path(verification_path).resolve()
    independent_verification_path = Path(independent_verification_path).resolve()
    report = load_json(verification_path)
    independent_report = load_json(independent_verification_path)
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
    spec = importlib.util.spec_from_file_location("round72_margin_contract_round71", builder_path)
    round71 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(round71)
    backup = Path(report.get("backup", "")).resolve()
    require(backup.is_file() and backup != deck, "round71 14-slide backup is missing")
    require(sha256(backup) == report.get("backup_sha256") == report.get("prior_deck_sha256"),
            "round71 source backup hash mismatch")
    require(len(Presentation(backup).slides) == 14, "round71 source backup is not 14 slides")
    with zipfile.ZipFile(backup) as before, zipfile.ZipFile(deck) as after:
        for index in range(1, 15):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"margin build changed prior XML: {part}")
    final_by_name = {}
    final_identities = []
    for path_text, recorded in report.get("final_json", {}).items():
        final_identities.append(require_identity(recorded, path_text, "margin final JSON"))
        final_by_name[Path(path_text).name] = Path(path_text).resolve()
    require(len(final_identities) == 4
            and {"independent_evaluation.json", "gate_summary.json", "input_manifest.json",
                 "exposure_sensitivity.json"}.issubset(final_by_name),
            "margin final evaluation/summary/manifest/exposure identities incomplete")
    evaluation = load_json(final_by_name["independent_evaluation.json"])
    manifest = load_json(final_by_name["input_manifest.json"])
    exposure_report = report.get("exposure_sensitivity", {})
    require(exposure_report.get("embedded_field") == "preregistered_exposure_sensitivity"
            and exposure_report.get("embedded_matches") is True,
            "margin exposure embedded field/match evidence is absent")
    exposure_path = Path(exposure_report.get("path", "")).resolve()
    require(exposure_path == final_by_name["exposure_sensitivity.json"],
            "margin exposure path differs from final_json identity")
    exposure_identity = require_identity(exposure_report, exposure_path, "margin exposure sensitivity")
    try:
        exposure_bundle = round71.validate_exposure_sensitivity(evaluation, manifest, exposure_path)
        exposure_registry_identities = round71.verify_identities(exposure_bundle["data"])
    except round71.EligibilityError as error:
        raise EligibilityError(f"round71 exposure revalidation failed: {error}") from error
    require(exposure_bundle["identity"] == exposure_identity,
            "round71 exposure revalidation identity differs from margin report")
    exposure = exposure_bundle["data"]
    counts = exposure["registry"]["cluster_dev_counts"]
    require(exposure_report.get("cluster_dev_counts") == counts == round71.EXPOSURE_COUNTS,
            "margin cluster_dev_counts differ from preregistered exposure counts")
    unknown = exposure["strata"]["NOT_DETECTED_UNKNOWN"]["ensemble_vs_controls"]["matched_control"]
    require(exposure_report.get("unknown_ensemble_minus_control") == unknown,
            "margin UNKNOWN ensemble-minus-control evidence differs from exposure artifact")
    require(independent_report.get("status") == "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL"
            and independent_report.get("slide_count") == 15
            and independent_report.get("prior_slide_xml_preserved") == 14
            and independent_report.get("primary_sha256") == deck_identity["sha256"]
            and independent_report.get("scientific_status") == report["scientific_status"]
            and independent_report.get("missing_parts") == []
            and independent_report.get("extra_parts") == []
            and independent_report.get("different_parts") == [],
            "round71 independent rebuild report is incomplete or belongs to another deck")
    require(Path(independent_report.get("exposure_sensitivity_path", "")).resolve() == exposure_path
            and independent_report.get("exposure_sensitivity_sha256") == exposure_identity["sha256"]
            and independent_report.get("exposure_embedded_matches") is True,
            "round71 independent report exposure path/SHA/match evidence is absent or inconsistent")
    margin_text = slide_text(prs.slides[14])
    require("34 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT" in margin_text
            and "MODEL-DEV ONLY" in margin_text and report["scientific_status"] in margin_text
            and "UNKNOWN≠clean" in margin_text and "官方 manifest UNCHECKED" in margin_text
            and "UNKNOWN pair Δ" in margin_text and "UNKNOWN stem Δ" in margin_text,
            "deck page 34 lacks the verified structured-margin or exposure result")
    for state, count in counts.items():
        label = "UNKNOWN" if state == "NOT_DETECTED_UNKNOWN" else state
        require(f"{label} {count:,}" in margin_text, f"page34 exposure count missing: {state}")
    for kind in ("pair", "stem"):
        delta = unknown[kind]["f1"]["delta_ensemble_minus_control"]
        require(f"{delta:+.6f}" in margin_text, f"page34 UNKNOWN {kind} delta missing")
    return {"deck": deck, "verification_path": verification_path,
            "independent_verification_path": independent_verification_path,
            "report": report, "independent_report": independent_report,
            "deck_identity": deck_identity, "backup": backup, "final_identities": final_identities,
            "exposure": exposure, "exposure_identity": exposure_identity,
            "exposure_registry_identities": exposure_registry_identities,
            "cluster_dev_counts": counts, "unknown_ensemble_minus_control": unknown}


def validate_external_raw(panel_registry_path: str | Path, system_metrics_path: str | Path,
                          independent_verification_path: str | Path, panel_labels_path: str | Path) -> dict:
    panel_registry_path = Path(panel_registry_path).resolve()
    system_metrics_path = Path(system_metrics_path).resolve()
    independent_verification_path = Path(independent_verification_path).resolve()
    panel_labels_path = Path(panel_labels_path).resolve()
    registry = load_json(panel_registry_path)
    system = load_json(system_metrics_path)
    independent = load_json(independent_verification_path)
    require(registry.get("schema_version") == system.get("schema_version") == PANEL_SCHEMA,
            "old DERIVED_FROM_BPFOLD_PREDICTIONS lineage or missing official-reference registry")
    panel = registry.get("label_panel", {})
    require(panel == system.get("label_panel")
            and panel.get("dataset_id") == PANEL_DATASET
            and panel.get("n_sequences") == 5388
            and panel.get("sha256", "").startswith("4924e351")
            and "DERIVED_FROM_BPFOLD_PREDICTIONS" not in json.dumps(panel),
            "official-reference 5388 panel identity/lineage changed")
    require(panel_labels_path.is_file(), f"missing official-reference 5388 panel: {panel_labels_path}")
    labels_identity = identity(panel_labels_path)
    require(labels_identity["bytes"] == panel.get("bytes")
            and labels_identity["sha256"] == panel.get("sha256")
            and panel_labels_path.as_posix().endswith(panel.get("path", "")),
            "official-reference panel path/bytes/SHA mismatch")
    coverage = registry.get("coverage_accounting", {})
    require(coverage == system.get("coverage_accounting")
            and coverage.get("full_coverage_scored") == 28
            and coverage.get("inventory_entries") == 43
            and registry.get("circular_project_label_score_excluded") is True
            and "descriptive audit, not model selection" in registry.get("metric_contract", {}).get("ranking", ""),
            "official-reference panel must contain 28 full-coverage descriptive systems")
    rows = registry.get("rows", [])
    system_rows = system.get("rows", [])
    require(len(rows) == len(system_rows) == 43
            and sum(row.get("status") == "PASS_FULL_COVERAGE" for row in rows) == 28,
            "official-reference inventory or full-coverage count mismatch")
    by_id = {row.get("prediction_id"): row for row in rows}
    system_by_id = {row.get("prediction_id"): row for row in system_rows}
    for prediction_id, row in by_id.items():
        other = system_by_id.get(prediction_id, {})
        for key in ("baseline_id", "prediction_origin", "path", "sha256", "label_dataset_id",
                    "n_panel_sequences", "n_with_prediction", "coverage", "status", "metrics"):
            require(row.get(key) == other.get(key), f"registry/system_metrics mismatch: {prediction_id}/{key}")
    ids = ("structrfm_official_v0.0.8/bprna_new", "bpfold/bprna_new",
           "contrafold/bprna_new", "xens2/bprna_new",
           "bpfold_official_release/bprna_new_official_reference")
    selected = []
    for prediction_id in ids:
        row = by_id.get(prediction_id, {})
        require(row.get("label_dataset_id") == PANEL_DATASET
                and row.get("status") == "PASS_FULL_COVERAGE"
                and row.get("n_panel_sequences") == row.get("n_with_prediction") == 5388
                and row.get("coverage") == 1.0 and row.get("missing_count") == 0
                and row.get("metrics", {}).get("n_scored") == 5388,
                f"selected system lacks full new-panel coverage: {prediction_id}")
        require(row.get("sha256") == sha256(row["path"]),
                f"selected prediction SHA mismatch: {prediction_id}")
        for metric in ("micro_f1", "macro_f1"):
            finite_metric(row["metrics"].get(metric), f"{prediction_id} {metric}")
        selected.append(row)
    official, replay, contrafold, xens2, author_release = selected
    require(official.get("prediction_origin") == "registered_frozen_prediction"
            and official.get("registry_status") == "FROZEN"
            and replay.get("prediction_origin") == "project_replay"
            and author_release.get("prediction_origin") == "author_release"
            and replay.get("path") != author_release.get("path")
            and replay.get("sha256") != author_release.get("sha256"),
            "BPfold project replay and author release identities must remain distinct")
    require(registry.get("strongest_overall", {}).get("prediction_id") == replay["prediction_id"]
            and registry["strongest_overall"].get("metrics") == replay["metrics"]
            and registry.get("strongest_single_predictor", {}).get("prediction_id") == contrafold["prediction_id"]
            and registry["strongest_single_predictor"].get("metrics") == contrafold["metrics"]
            and registry.get("xens2_reference", {}).get("metrics") == xens2["metrics"],
            "official-reference descriptive summary differs from selected rows")
    require(independent.get("status") == "PASS_INDEPENDENT_SECOND_IMPLEMENTATION"
            and independent.get("panel_records_verified") == 5388
            and independent.get("full_coverage_systems_verified") == 28
            and independent.get("paired_ci_verified") == 28
            and independent.get("new_inference") is False
            and independent.get("model_selection") is False,
            "official-reference independent verification is incomplete")
    deliverables = {Path(row.get("path", "")).name: row for row in independent.get("deliverables", [])}
    for path in (panel_labels_path, system_metrics_path, panel_registry_path):
        recorded = deliverables.get(path.name, {})
        actual = identity(path)
        require(recorded.get("bytes") == actual["bytes"] and recorded.get("sha256") == actual["sha256"],
                f"independent report does not bind new-panel deliverable: {path.name}")
    return {"official": official, "comparators": [replay, contrafold], "xens2": xens2,
            "author_release": author_release, "panel": panel, "coverage": coverage,
            "system_metrics_path": system_metrics_path,
            "paths": (panel_registry_path, system_metrics_path, independent_verification_path,
                      panel_labels_path, *(Path(row["path"]) for row in selected))}


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


def fmt10(value: float) -> str:
    return f"{value:.10f}"


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
    rows = [["structRFM official v0.0.8 · single", fmt10(sm["micro_f1"]), "reference",
             fmt10(sm["macro_f1"]), "reference"]]
    labels = (("BPfold project replay · six-checkpoint", bundle["comparators"][0]["metrics"]),
              ("CONTRAfold · deterministic MEA", bundle["comparators"][1]["metrics"]),
              ("xens2 · frozen project ensemble", bundle["xens2"]["metrics"]))
    for label, metrics in labels:
        rows.append([label, fmt10(metrics["micro_f1"]), signed(sm["micro_f1"] - metrics["micro_f1"]),
                     fmt10(metrics["macro_f1"]), signed(sm["macro_f1"] - metrics["macro_f1"])])
    slide = new_slide(prs, 35, "FOLDNAR EXTERNAL RESULTS / OFFICIAL-REFERENCE RAW-EXPOSURE",
                      "structRFM官方结果：新参考面板统一重算",
                      f"OFFICIAL_REFERENCE_5388 · {bundle['coverage']['full_coverage_scored']} full-coverage systems · raw-exposure descriptive · Δ = structRFM − 对照")
    table(slide, 0.78, 1.65, (4.18, 1.80, 2.00, 1.80, 2.00), 0.46,
          ("同一官方参考标签面板", "micro-F1", "Δ micro", "macro-F1", "Δ macro"), rows,
          font_size=10.2, emphasis_row=1)
    rect(slide, 0.78, 4.18, 11.78, 0.90, PALE, BURGUNDY)
    text(slide, 1.00, 4.29, 11.32, 0.28, "RAW-EXPOSURE DESCRIPTIVE ≠ CLEAN SOTA", 16, DARK_RED, True)
    text(slide, 1.00, 4.62, 11.30, 0.26,
         "28个全覆盖系统仅作raw-exposure描述与绝对差距审计；不去污、不用于模型选择、不宣称clean SOTA。",
         10.3, INK, True, SANS)
    rect(slide, 0.78, 5.30, 5.72, 0.84, PASS_PALE, GREEN)
    text(slide, 1.00, 5.40, 5.28, 0.24, "BPfold项目replay ≠ 官方release", 12, GREEN, True, SANS)
    text(slide, 1.00, 5.69, 5.28, 0.27,
         "不同prediction_id / path / SHA；作者release另行注册，禁止把项目replay身份写成作者发布结果。",
         8.6, INK, font=SANS)
    rect(slide, 6.74, 5.30, 5.82, 0.84, ROSE, LINE)
    text(slide, 6.96, 5.40, 5.38, 0.24, "面板身份与独立重算", 12, BURGUNDY, True, SANS)
    text(slide, 6.96, 5.69, 5.38, 0.27,
         f"official-reference SHA {bundle['panel']['sha256'][:8]}… · 5388 records · 28 full coverage · second implementation PASS",
         8.5, INK)
    text(slide, 0.78, 6.46, 11.70, 0.18,
         f"panel commit {PANEL_COMMIT[:8]} · system_metrics SHA {sha256(bundle['system_metrics_path'])[:12]} · registry/system rows + prediction SHA verified",
         7.7, MUTED)
    text(slide, 0.78, 6.72, 11.70, 0.18,
         "旧DERIVED_FROM_BPFOLD_PREDICTIONS项目raw排名已弃用；本页不构成去污外部测试、训练暴露清除或clean SOTA证据。",
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
    for phrase in ("35 · FOLDNAR EXTERNAL RESULTS", "structRFM官方结果", fmt10(sm["micro_f1"]),
                   fmt10(sm["macro_f1"]), "BPfold project replay", "官方release", "CONTRAfold", "xens2",
                   "RAW-EXPOSURE DESCRIPTIVE ≠ CLEAN SOTA", "28 full-coverage systems",
                   "不去污", "不宣称clean SOTA", "DERIVED_FROM_BPFOLD_PREDICTIONS"):
        require(phrase in external_text, f"missing external slide text: {phrase}")
    for metrics in [row["metrics"] for row in external["comparators"]] + [external["xens2"]["metrics"]]:
        require(fmt10(metrics["micro_f1"]) in external_text and fmt10(metrics["macro_f1"]) in external_text,
                "official-reference comparator metrics missing")
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
          margin_independent_verification: str | Path = MARGIN_INDEPENDENT_VERIFICATION,
          backup: str | Path | None = None, verification: str | Path | None = None,
          panel_registry: str | Path = PANEL_REGISTRY,
          system_metrics: str | Path = PANEL_SYSTEM_METRICS,
          panel_independent_verification: str | Path = PANEL_INDEPENDENT_VERIFICATION,
          panel_labels: str | Path = PANEL_LABELS,
          longseq_root: str | Path = LONGSEQ_ROOT) -> dict:
    deck = Path(deck).resolve()
    backup = Path(backup).resolve() if backup else deck.with_name(deck.stem + ".pre_round72.pptx")
    verification = (Path(verification).resolve() if verification else
                    deck.with_name(deck.stem + ".round72_build_verification.json"))
    margin = validate_margin(deck, margin_verification, margin_independent_verification)
    external = validate_external_raw(panel_registry, system_metrics,
                                     panel_independent_verification, panel_labels)
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
        "margin_independent_verification": identity(margin["independent_verification_path"]),
        "margin_deck_sha256_verified": margin["deck_identity"]["sha256"],
        "margin_final_json": margin["final_identities"],
        "exposure_sensitivity": {**margin["exposure_identity"],
                                 "embedded_field": "preregistered_exposure_sensitivity",
                                 "embedded_matches": True,
                                 "cluster_dev_counts": margin["cluster_dev_counts"],
                                 "unknown_ensemble_minus_control": margin["unknown_ensemble_minus_control"]},
        "verified_exposure_registry_identities": margin["exposure_registry_identities"],
        "external_artifacts": [identity(path) for path in external["paths"]],
        "longseq_artifacts": [identity(path) for path in longseq["paths"]],
        "scope": {"raw_not_clean_not_sota": True, "external_raw_n": 5388,
                  "official_reference_descriptive_only": True,
                  "official_reference_full_coverage_systems": 28,
                  "old_derived_label_ranking_used": False,
                  "bpfold_replay_author_release_conflated": False,
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
    parser.add_argument("--margin-independent-verification", type=Path,
                        default=MARGIN_INDEPENDENT_VERIFICATION)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--panel-registry", type=Path, default=PANEL_REGISTRY)
    parser.add_argument("--system-metrics", type=Path, default=PANEL_SYSTEM_METRICS)
    parser.add_argument("--panel-independent-verification", type=Path,
                        default=PANEL_INDEPENDENT_VERIFICATION)
    parser.add_argument("--panel-labels", type=Path, default=PANEL_LABELS)
    parser.add_argument("--longseq-root", type=Path, default=LONGSEQ_ROOT)
    args = parser.parse_args(argv)
    try:
        report = build(args.deck, args.margin_verification, args.margin_independent_verification,
                       args.backup, args.verification, args.panel_registry, args.system_metrics,
                       args.panel_independent_verification, args.panel_labels, args.longseq_root)
    except (EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"round72 build refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
