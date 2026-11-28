#!/usr/bin/env python3
"""Append slide 37 to a hash-verified round72 17-slide delivery copy."""
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
ROUND72_ROOT = Path("/home/rna_jev_external_result_ppt")
SOURCE_DECK = ROUND72_ROOT / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.round72_delivery.pptx"
SOURCE_VERIFICATION = ROUND72_ROOT / ".round72_work/final_publication/publication_success.json"
DATA_REGISTRY = ROOT / "manifests/foldnar_v1/data_registry.json"
ARCHIVE_LINEAGE = ROOT / "artifacts/archive_label_lineage_audit_20261127/summary.json"
ARCHIVE_SYSTEM = ROOT / "artifacts/archiveii_official_reference_20261127/system_metrics.json"
ARCHIVE_RECOMMENDATION = ROOT / "artifacts/archiveii_official_reference_20261127/registry_recommendation.json"
ARCHIVE_FULL_SYSTEM = ROOT / "artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json"
ARCHIVE_FULL_CHECK = ROOT / "artifacts/archiveii_full3966_cpu_recovery_20261127/verification_round3.json"
ARCHIVE_XENS2_QUEUE = ROOT / "source_data/archiveii_vienna_full3966_cpu_20261127/xens2_queue_evidence.json"
BPRNA_SYSTEM = ROOT / "source_data/bprna_official_reference_5388/system_metrics.json"
PRETRAIN_SUMMARY = ROOT / "manifests/pretrain_exposure_v1/summary.json"
PDB_LINEAGE = ROOT / "source_data/pdb_rfam_label_lineage_audit_20261127/panel_summary.json"
PDB_SUMMARY = ROOT / "source_data/pdb_rcsb_rebuild_20261127/round6_combined/rebuild_summary.json"
PDB_VERIFICATION = ROOT / "source_data/pdb_rcsb_rebuild_20261127/round6_combined_verification.json"
TS_COVERAGE = ROOT / "artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json"
TS_VERIFICATION = ROOT / "artifacts/ts0_testsetb_official_reference_20261127/verification_round3.json"
TS0_RECOVERY = ROOT / "artifacts/ts0_official_coverage_recovery_20261127/recovery_audit.json"
TS0_RECOVERY_CHECK = ROOT / "artifacts/ts0_official_coverage_recovery_20261127/verification_round2.json"
TESTSETB_RECOVERY = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127/recovery_audit.json"
TESTSETB_RECOVERY_CHECK = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127/verification_round3.json"
RFAM_REGISTRY = ROOT / "source_data/rfam_release_lineage_20261127/final_registry_proposal/registry_recommendation.json"
RFAM_VERIFICATION = ROOT / "source_data/rfam_release_lineage_20261127/final_registry_verification.json"
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
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EligibilityError(f"invalid frozen JSON {path}: {error}") from error
    require(isinstance(value, dict), f"frozen JSON must be an object: {path}")
    return value


def finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"invalid metric: {label}")
    return float(value)


def fmt10(value: float) -> str:
    return f"{value:.10f}"


def signed10(value: float) -> str:
    return f"{value:+.10f}"


def slide_text(slide) -> str:
    values = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            values.append(shape.text_frame.text)
        if shape.has_table:
            values.extend(cell.text for row in shape.table.rows for cell in row.cells)
    return "\n".join(values)


def validate_source(source_deck: str | Path, source_verification: str | Path,
                    expected_source_sha256: str | None = None) -> dict:
    source_deck = Path(source_deck).resolve()
    source_verification = Path(source_verification).resolve()
    report = load_json(source_verification)
    actual = identity(source_deck)
    require(report.get("status") == "READY_FOR_ATOMIC_PUBLICATION",
            "round72 publication success is absent")
    require(report.get("slide_count") == 17 and len(Presentation(source_deck).slides) == 17,
            "round73 requires the real or synthetic 17-slide round72 delivery")
    require(Path(report.get("delivery_path", "")).resolve() == source_deck,
            "source deck is not the published round72 delivery")
    recorded = report.get("delivery_deck", {})
    require(recorded.get("bytes") == actual["bytes"] and recorded.get("sha256") == actual["sha256"],
            "round72 delivery bytes/SHA-256 differ from publication_success")
    require(report.get("source_slide_xml_preserved") == 14
            and report.get("prior_slide_xml_preserved") == 15,
            "round72 prior-slide XML contract is incomplete")
    require(report.get("scientific_status") == "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
            and report.get("scientific_overall_pass") is False,
            "the four-arm unconverged scientific boundary changed")
    require(report.get("final_marker", {}).get("ready") is True,
            "round72 final marker is not ready")
    steps = report.get("steps", [])
    require(len(steps) == 6 and all(step.get("exit_code") == 0 for step in steps)
            and steps[-1].get("name") == "round72_independent_verification",
            "round72 six-step publication transaction is incomplete")
    if expected_source_sha256 is not None:
        require(expected_source_sha256 == actual["sha256"],
                "explicit round72 delivery SHA-256 does not match")
    return {"deck": source_deck, "verification": source_verification,
            "identity": actual, "report": report}


def registry_row(registry: dict, dataset_id: str) -> dict:
    matches = [row for row in registry.get("rows", []) if row.get("dataset_id") == dataset_id]
    require(len(matches) == 1, f"expected one data-registry row: {dataset_id}")
    return matches[0]


def prediction_row(system: dict, prediction_id: str, panel: str) -> dict:
    matches = [row for row in system.get("rows", []) if row.get("prediction_id") == prediction_id]
    require(len(matches) == 1, f"expected one prediction row: {prediction_id}")
    row = matches[0].get("panels", {}).get(panel, {})
    require(row.get("status") == "PASS_FULL_COVERAGE" and row.get("coverage") == 1.0
            and row.get("n_with_prediction") == row.get("n_panel_sequences"),
            f"full coverage missing: {prediction_id}")
    finite(row.get("metrics", {}).get("micro_f1"), prediction_id)
    return row


def validate_evidence(data_registry: str | Path = DATA_REGISTRY,
                      archive_lineage: str | Path = ARCHIVE_LINEAGE,
                      archive_system: str | Path = ARCHIVE_SYSTEM,
                      archive_recommendation: str | Path = ARCHIVE_RECOMMENDATION,
                      archive_full_system: str | Path = ARCHIVE_FULL_SYSTEM,
                      archive_full_check: str | Path = ARCHIVE_FULL_CHECK,
                      archive_xens2_queue: str | Path = ARCHIVE_XENS2_QUEUE,
                      bprna_system: str | Path = BPRNA_SYSTEM,
                      pretrain_summary: str | Path = PRETRAIN_SUMMARY,
                      pdb_lineage: str | Path = PDB_LINEAGE,
                      pdb_summary: str | Path = PDB_SUMMARY,
                      pdb_verification: str | Path = PDB_VERIFICATION,
                      ts_coverage: str | Path = TS_COVERAGE,
                      ts_verification: str | Path = TS_VERIFICATION,
                      ts0_recovery: str | Path = TS0_RECOVERY,
                      ts0_recovery_check: str | Path = TS0_RECOVERY_CHECK,
                      testsetb_recovery: str | Path = TESTSETB_RECOVERY,
                      testsetb_recovery_check: str | Path = TESTSETB_RECOVERY_CHECK,
                      rfam_registry: str | Path = RFAM_REGISTRY,
                      rfam_verification: str | Path = RFAM_VERIFICATION) -> dict:
    paths = tuple(Path(path).resolve() for path in
                  (data_registry, archive_lineage, archive_system, archive_recommendation,
                   archive_full_system, archive_full_check, archive_xens2_queue,
                   bprna_system, pretrain_summary, pdb_lineage, pdb_summary, pdb_verification,
                   ts_coverage, ts_verification, ts0_recovery, ts0_recovery_check,
                   testsetb_recovery, testsetb_recovery_check, rfam_registry, rfam_verification))
    (registry, lineage, archive, recommendation, archive_full, archive_full_check_report,
     xens2_queue, bprna, pretrain, pdb, pdb_rebuild, pdb_check, coverage, coverage_check, ts0, ts0_check,
     testsetb, testsetb_check, rfam, rfam_check) = map(load_json, paths)
    legacy = {}
    for dataset_id, replacement in (("bprna_new", "bpfold_official_bprna_new_reference_5388"),
                                    ("archiveii_embok_clean", None)):
        row = registry_row(registry, dataset_id)
        label = row.get("label_lineage", {})
        require(row.get("label_source") == label.get("label_source") == "DERIVED_FROM_BPFOLD_PREDICTIONS"
                and row.get("independent_sota_eligible") is False
                and label.get("independent_sota_eligible") is False
                and label.get("disposition") == "circular-audit-only",
                f"prediction-derived legacy label disqualification changed: {dataset_id}")
        if replacement:
            require(row.get("official_reference_replacement_panel") == replacement,
                    "bpRNA-new official-reference replacement changed")
        else:
            require(row.get("official_reference_replacement_panels") == [
                "archiveii_official_reference_3966", "archiveii_official_reference_clean_2544"],
                "ArchiveII official-reference replacements changed")
        legacy[dataset_id] = row
    archive_old = lineage.get("datasets", {}).get("archiveii_embok_clean", {})
    require(archive_old.get("project_records") == 2544
            and archive_old.get("record_status_counts", {}).get(
                "EXACT_DERIVED_FROM_BPFOLD_PREDICTION_PROJECTION") == 2544
            and archive_old.get("external_test_model_selection_allowed") is False,
            "ArchiveII prediction projection lineage is incomplete")
    require(bprna.get("label_panel", {}).get("n_sequences") == 5388
            and bprna.get("circular_project_label_score_excluded") is True
            and bprna.get("coverage_accounting", {}).get("full_coverage_scored") == 28,
            "bpRNA-new official-reference panel contract changed")
    panel = "archiveii_official_reference_clean_2544"
    require(archive.get("label_panels", {}).get(panel, {}).get("n_sequences") == 2544
            and "no min-loop/crossing/canonical projection" in
            archive["label_panels"][panel].get("pair_protocol", "")
            and archive.get("external_test_model_selection") is False,
            "ArchiveII official-reference clean2544 protocol changed")
    ids = ("xens2/archiveii_embok_clean", "bpfold/archiveii_embok_clean",
           "rinalmo_ft/archiveii_embok_clean")
    xens2, replay, rinalmo = [prediction_row(archive, prediction_id, panel) for prediction_id in ids]
    expected = dict(zip(ids, ("0.8156522070", "0.8999331355", "0.8204522395")))
    for prediction_id, row in zip(ids, (xens2, replay, rinalmo)):
        require(fmt10(row["metrics"]["micro_f1"]) == expected[prediction_id],
                f"ArchiveII official-reference metric changed: {prediction_id}")
    strongest = recommendation.get("suggested_baseline_overlay", {}).get(
        "strongest_clean_2544", {}).get("strongest_single_predictor", {})
    require(strongest.get("prediction_id") == ids[2]
            and strongest.get("ranking_scope") ==
            "DESCRIPTIVE_FROZEN_PROTOCOL_AUDIT_ONLY_NOT_MODEL_SELECTION"
            and recommendation.get("legacy_project_labels", {}).get("independent_sota_eligible") is False,
            "ArchiveII strongest-single descriptive boundary changed")
    full = archive_full.get("full_3966", {})
    strongest_full = full.get("strongest_single_predictor", {})
    require(archive_full.get("label_panels", {}).get(
                "archiveii_official_reference_3966", {}).get("n_sequences") == 3966
            and archive_full.get("coverage_accounting", {}).get("official_3966_full_coverage") == 8
            and archive_full.get("coverage_accounting", {}).get("xens2_protocol_blocked_records") == 16
            and archive_full.get("external_test_model_selection") is False
            and strongest_full.get("prediction_id") ==
            "mxfold2/archiveii_official_reference_3966/cpu_recovery_v1"
            and strongest_full.get("model_form") == "deterministic_single_predictor"
            and fmt10(strongest_full.get("metrics", {}).get("micro_f1")) == "0.7023050286"
            and full.get("xens2_delta") == "UNCHECKED_XENS2_NOT_FULL_COVERAGE",
            "ArchiveII full3966 eight-full/MXfold2 contract changed")
    full_rows = []
    xens2_full = None
    for row in archive_full.get("rows", []):
        row_panel = row.get("panels", {}).get("archiveii_official_reference_3966") or {}
        if row_panel.get("status") == "PASS_FULL_COVERAGE":
            require(row_panel.get("coverage") == 1.0
                    and row_panel.get("n_with_prediction") == 3966
                    and row_panel.get("missing_count") == 0,
                    f"ArchiveII full3966 coverage changed: {row.get('prediction_id')}")
            full_rows.append(row.get("prediction_id"))
        if row.get("prediction_id") == "xens2/archiveii_embok_clean":
            xens2_full = row_panel
    vienna_full = {"vienna_mfe/archiveii_official_reference_3966/cpu_recovery_v1",
                   "vienna_centroid/archiveii_official_reference_3966/cpu_recovery_v1",
                   "vienna_mea/archiveii_official_reference_3966/cpu_recovery_v1"}
    require(len(full_rows) == 8 and vienna_full.issubset(full_rows) and xens2_full is not None
            and xens2_full.get("status") == "PARTIAL_COVERAGE_NOT_RANKED"
            and xens2_full.get("n_with_prediction") == 2544
            and xens2_full.get("missing_count") == 1422
            and xens2_full.get("coverage") == 2544 / 3966,
            "ArchiveII Vienna eight-full or xens2 partial/no-ranking boundary changed")
    require(xens2_queue.get("production_started") is False
            and xens2_queue.get("status", {}).get("state") == "WAITING_PRODUCTION"
            and xens2_queue.get("production_blocked_reason_counts", {}).get(
                "BLOCKED_PROTOCOL_LENGTH_GT1024") == 16
            and 2544 + 1422 - 16 == 3950,
            "ArchiveII xens2 GPU-queued/theoretical-max3950 boundary changed")
    require(archive_full_check_report.get("status") == "PASS_CLEAN_CLONE_AND_DELIVERY"
            and archive_full_check_report.get("clean_clone_outputs_byte_identical") is True
            and archive_full_check_report.get("external_test_model_selection") is False
            and archive_full_check_report.get("clean_clone_output_sha256", {}).get(
                "artifacts/archiveii_full3966_cpu_recovery_20261127/system_metrics.json"
            ) == sha256(archive_full_system),
            "ArchiveII full3966 clean-clone verification or artifact identity changed")
    require(pdb_rebuild.get("n_source_records") == 153
            and pdb_rebuild.get("n_rebuilt") == 62
            and pdb_rebuild.get("n_unresolved") == 91
            and pdb_rebuild.get("external_test_policy") == "AUDIT_ONLY_NO_MODEL_SELECTION"
            and sum(pdb_rebuild.get("panel_rebuilt", {}).values()) == 62
            and sum(pdb_rebuild.get("unresolved_status_counts", {}).values()) == 91
            and sum(pdb_rebuild.get("pair_match_counts", {}).values()) == 62,
            "PDB62/91 rebuild conservation or no-ranking contract changed")
    require(pdb_check.get("status") == "PASS_RNAFORMER_DOUBLE_PROTOCOL_DSSR_INDEPENDENT_VERIFIER"
            and pdb_check.get("n_rebuilt") == 62 and pdb_check.get("n_unresolved") == 91
            and pdb_check.get("pair_match_counts") == pdb_rebuild.get("pair_match_counts")
            and pdb_check.get("rebuild_summary_sha256") == sha256(pdb_summary)
            and pdb_check.get("unobserved_negative_policy") == "MASKED_NEVER_NEGATIVE",
            "PDB62/91 independent verification or summary identity changed")
    require(coverage_check.get("status") == "PASS_CLEAN_CLONE_AND_DELIVERY"
            and coverage_check.get("clean_clone_outputs_byte_identical") is True
            and coverage_check.get("external_test_model_selection") is False
            and coverage_check.get("clean_clone_output_sha256", {}).get(
                "artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json"
            ) == sha256(ts_coverage),
            "TS0/TestSetB clean-clone verification or coverage identity changed")
    for dataset_id, project_n, official_n, missing_n in (("bprna_ts0", 1288, 1305, 17),
                                                        ("testsetb", 428, 430, 2)):
        panel = coverage.get("datasets", {}).get(dataset_id, {})
        require(panel.get("project_n") == project_n and panel.get("official_n") == official_n
                and panel.get("official_missing_from_project_count") == missing_n
                and len(panel.get("official_missing_from_project", [])) == missing_n
                and official_n - project_n == missing_n,
                f"TS0/TestSetB projected/full coverage accounting changed: {dataset_id}")
    require(ts0.get("status") == "PASS_7_FULL1305_IDENTITIES_2_PARTIAL"
            and ts0.get("external_test_model_selection") is False
            and len(ts0.get("new_full_coverage_identities", [])) == 7
            and ts0.get("official_panel", {}).get("n_sequences") == 1305
            and ts0.get("project_panel", {}).get("n_sequences") == 1288,
            "TS0 seven-full/projected recovery contract changed")
    require(ts0_check.get("status") == "PASS_INDEPENDENT_RECOVERY_RECONSTRUCTION"
            and set(ts0_check.get("full1305_identities", [])) == set(ts0["new_full_coverage_identities"])
            and len(ts0_check.get("partial1302_identities", [])) == 2
            and ts0_check.get("deliverables", {}).get(
                "artifacts/ts0_official_coverage_recovery_20261127/recovery_audit.json", {}
            ).get("sha256") == sha256(ts0_recovery),
            "TS0 independent recovery verification changed")
    require(testsetb.get("status") == "PASS_7_FULL430_IDENTITIES_AND_2_PARTIAL429_WITH_EXPLICIT_BLOCKS"
            and testsetb.get("external_test_model_selection") is False
            and len(testsetb.get("new_full_coverage_identities", [])) == 7
            and testsetb.get("official_panel", {}).get("n_sequences") == 430
            and testsetb.get("project_panel", {}).get("n_sequences") == 428,
            "TestSetB seven-full/projected recovery contract changed")
    require(testsetb_check.get("status") == "PASS"
            and testsetb_check.get("coverage_accounting", {}).get("official_full_coverage") == 7
            and set(testsetb_check.get("official_full_coverage_identities", [])) ==
            set(testsetb["new_full_coverage_identities"])
            and testsetb_check.get("recovery_manifest_sha256") == sha256(testsetb_recovery)
            and testsetb_check.get("project_428_metrics_unchanged_for_prior_20") is True,
            "TestSetB independent recovery verification changed")
    for recovery, count_key, partial_n in ((ts0, "n_prediction_records", 1302),
                                           (testsetb, "n_sequences", 429)):
        entries = {row.get("prediction_id"): row for row in recovery.get("prediction_entries", [])}
        official_n = recovery["official_panel"]["n_sequences"]
        for prediction_id in recovery["new_full_coverage_identities"]:
            require(entries.get(prediction_id, {}).get(count_key) == official_n,
                    f"official-full identity is incomplete: {prediction_id}")
        for prediction_id in ("rnastructure_fold/" + recovery["dataset_id"],
                              "rnastructure_maxexpect/" + recovery["dataset_id"]):
            require(entries.get(prediction_id, {}).get(count_key) == partial_n,
                    f"RNAstructure partial coverage changed: {prediction_id}")
    require(pdb.get("negative_label_policy", {}).get("unobserved") ==
            "must be masked, never a negative label",
            "PDB observation-mask qualification changed")
    temporal = rfam.get("panels", {}).get("rfam_temporal", {})
    novelty = rfam.get("temporal_novelty_counts", {})
    require(rfam.get("external_test_policy") == "AUDIT_ONLY_NO_MODEL_SELECTION"
            and temporal.get("n_records") == 436
            and novelty.get("NOVEL_SEQUENCE_RELEASE_SUPPORTED") == 269
            and novelty.get("UNCHECKED") == 110
            and sum(novelty.values()) == 436
            and temporal.get("tr1c_exact_overlap") == 33
            and temporal.get("cross_panel_exact_overlap") == 2
            and temporal.get("tr1c_family_overlap") == 321,
            "Rfam temporal novelty or training-overlap accounting changed")
    require(rfam_check.get("status") == "PASS_RFAM_TEMPORAL_LINEAGE_INDEPENDENT_VERIFIER"
            and rfam_check.get("temporal_novelty_counts") == novelty
            and rfam_check.get("tr1c_exact") == 33 and rfam_check.get("cross_panel_exact") == 2
            and rfam_check.get("identity_pairs_used") is False
            and rfam_check.get("registry_recommendation_sha256") == sha256(rfam_registry),
            "Rfam temporal independent verification or artifact identity changed")
    require(pretrain.get("complete_official_pretraining_coverage_state") == "UNCHECKED",
            "complete pretraining exposure must remain UNCHECKED")
    populations = pretrain.get("populations", {})
    for dataset_id, n_rows, exact, unchecked in (("bprna_new", 5388, 5381, 7),
                                                ("archiveii_embok_clean", 2544, 2048, 496)):
        population = populations.get(dataset_id, {})
        require(population.get("n_input_rows") == n_rows
                and population.get("known_source_exact_state", {}).get("EXACT") == exact
                and population.get("complete_official_pretraining_exposure_state", {}).get("UNCHECKED") == unchecked
                and population.get("selection_use") is False,
                f"pretrain exposure boundary changed: {dataset_id}")
    return {"legacy": legacy, "xens2": xens2, "replay": replay, "rinalmo": rinalmo,
            "archive_full": archive_full, "pdb_rebuild": pdb_rebuild, "pretrain": pretrain, "coverage": coverage,
            "ts0_recovery": ts0, "testsetb_recovery": testsetb, "paths": paths}


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


def table(slide, x, y, widths, row_height, headers, rows):
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
            cell.fill.fore_color.rgb = BURGUNDY if row_index == 0 else ROSE if row_index % 2 == 0 else WHITE
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.005)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if column_index in (0, 3) else PP_ALIGN.CENTER
                paragraph.space_before = paragraph.space_after = Pt(0)
                paragraph.font.name = SANS if column_index in (0, 3) else "Arial"
                paragraph.font.size = Pt(9.5)
                paragraph.font.bold = row_index == 0
                paragraph.font.color.rgb = WHITE if row_index == 0 else INK
    return tbl


def add_label_slide(prs: Presentation, evidence: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.25, 11.8, 0.28, "37 · LABEL PROTOCOL / SOTA EVIDENCE ELIGIBILITY", 10.5, BURGUNDY, True)
    text(slide, 0.78, 0.63, 11.9, 0.52, "标签谱系决定SOTA证据资格", 25, INK, True, SERIF)
    text(slide, 0.78, 1.20, 11.9, 0.30,
         "PREDICTION-DERIVED → OFFICIAL-REFERENCE DESCRIPTIVE ONLY · 完整SOTA仍未闭环", 10.5, MUTED, font=SANS)
    rect(slide, 0.78, 1.62, 7.52, 1.14, PALE, BURGUNDY)
    text(slide, 1.00, 1.73, 7.08, 0.25, "① bpRNA-new / ArchiveII：旧标签资格否定", 12.2, BURGUNDY, True, SANS)
    text(slide, 1.00, 2.02, 7.08, 0.59,
         "旧 label_source = DERIVED_FROM_BPFOLD_PREDICTIONS；旧分数仅 circular-audit。\n新 official-reference：bpRNA-new 5388；ArchiveII 3966 → clean 2544（unfiltered BPSEQ）。",
         9.2, INK, font=SANS)
    rect(slide, 8.54, 1.62, 4.02, 1.14, ROSE, DARK_RED)
    text(slide, 8.76, 1.71, 3.58, 0.25, "③ PDB：62 rebuilt / 91 UNCHECKED", 11.0, DARK_RED, True)
    text(slide, 8.76, 1.99, 3.58, 0.70,
         "PDB 153 · mask≠negative · no ranking\nRfam temporal：269 novel-supported\n110 UNCHECKED · TR1c exact 33 · cross-panel exact 2\n训练重叠不豁免",
         6.8, INK, font=SANS)
    xens2 = evidence["xens2"]["metrics"]["micro_f1"]
    rows = []
    for label, key, role in (("xens2 · frozen project ensemble", "xens2", "reference"),
                             ("BPfold replay · six-checkpoint", "replay", "strongest overall*"),
                             ("RiNALMo-ft · single", "rinalmo", "strongest single*")):
        value = evidence[key]["metrics"]["micro_f1"]
        delta = "reference" if key == "xens2" else signed10(value - xens2)
        rows.append([label, fmt10(value), delta, role])
    table(slide, 0.78, 2.96, (4.10, 2.00, 2.12, 3.56), 0.36,
          ("ArchiveII official clean2544", "micro-F1", "Δ vs xens2", "descriptive role"), rows)
    full_single = evidence["archive_full"]["full_3966"]["strongest_single_predictor"]["metrics"]["micro_f1"]
    text(slide, 0.78, 4.43, 11.70, 0.32,
         "* clean2544 新差距仅描述同一全覆盖面板；不是模型选择，也不自动继承 clean SOTA 资格。\nArchiveII full3966：8 full · Vienna MFE/centroid/MEA full · MXfold2 strongest single=" + fmt10(full_single) +
         "\nxens2 GPU queued · 理论最多3950/3966（16 blocked）不排名 · full未完成",
         6.8, MUTED, font=SANS)
    rect(slide, 0.78, 4.78, 4.40, 0.24, PALE, BURGUNDY)
    text(slide, 0.94, 4.80, 4.08, 0.20,
         "PROJECTED · TS0 1288 · TestSetB 428\nfrozen project IDs；与 official 分开报告", 7.1, BURGUNDY, True, SANS)
    rect(slide, 5.40, 4.78, 7.16, 0.24, ROSE, DARK_RED)
    text(slide, 5.56, 4.80, 6.84, 0.20,
         "OFFICIAL · TS0 7×1305 full · TestSetB 7×430 full\nRNAstructure fold/MaxExpect：1302/1305 · 429/430 partial",
         7.0, DARK_RED, True, SANS)
    rect(slide, 0.78, 5.05, 11.78, 0.66, PASS_PALE, GREEN)
    text(slide, 1.00, 5.13, 4.25, 0.22, "② Pretrain exposure 仍否决 clean-OOD", 11.0, GREEN, True, SANS)
    text(slide, 1.00, 5.39, 11.30, 0.20,
         "bpRNA-new EXACT 5381/5388 rows · ArchiveII EXACT 2048/2544 rows · UNKNOWN / UNCHECKED ≠ clean",
         9.0, INK, True)
    rect(slide, 0.78, 5.94, 11.78, 0.70, PALE, DARK_RED)
    text(slide, 1.00, 6.02, 3.60, 0.26, "完整 SOTA 仍未闭环", 16, DARK_RED, True, SANS)
    text(slide, 4.25, 6.02, 8.08, 0.48,
         "official标签 × full coverage × model-specific exposure × PDB coordinate/observation lineage\n四臂未收敛；外部描述值不能替代任何一项证据资格。",
         9.0, INK, True, SANS)
    text(slide, 0.78, 6.80, 11.70, 0.18,
         "LABEL_REFERENCE_REBUILD_ONLY · PDB62/91不得排名 · legacy prediction-derived labels不用于独立SOTA · raw-exposure descriptive ≠ clean SOTA",
         7.6, MUTED, font=SANS)


def verify_built_deck(path: Path, prior_deck: Path, evidence: dict) -> dict:
    reopened = Presentation(path)
    require(len(Presentation(prior_deck).slides) == 17 and len(reopened.slides) == 18,
            "round73 output must contain exactly 18 slides")
    with zipfile.ZipFile(prior_deck) as before, zipfile.ZipFile(path) as after:
        for index in range(1, 18):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"prior slide part changed: {part}")
    for shape in reopened.slides[17].shapes:
        require(shape.left >= 0 and shape.top >= 0, "negative shape position")
        require(shape.left + shape.width <= reopened.slide_width + 2, "shape exceeds slide width")
        require(shape.top + shape.height <= reopened.slide_height + 2, "shape exceeds slide height")
    content = slide_text(reopened.slides[17])
    required = ("37 · LABEL PROTOCOL", "标签谱系决定SOTA证据资格", "DERIVED_FROM_BPFOLD_PREDICTIONS",
                "bpRNA-new 5388", "clean 2544", "0.8156522070", "0.8999331355", "0.8204522395",
                "+0.0842809284", "+0.0048000324", "strongest single", "62 rebuilt / 91 UNCHECKED",
                "ArchiveII full3966：8 full", "Vienna MFE/centroid/MEA full",
                "MXfold2 strongest single=0.7023050286",
                "xens2 GPU queued", "理论最多3950/3966（16 blocked）不排名", "full未完成",
                "PROJECTED · TS0 1288 · TestSetB 428", "与 official 分开报告",
                "OFFICIAL · TS0 7×1305 full · TestSetB 7×430 full",
                "RNAstructure fold/MaxExpect：1302/1305 · 429/430 partial",
                "Rfam temporal", "269 novel-supported", "110 UNCHECKED",
                "TR1c exact 33", "cross-panel exact 2", "训练重叠不豁免",
                "不得排名", "5381/5388", "2048/2544", "UNKNOWN / UNCHECKED ≠ clean",
                "完整 SOTA 仍未闭环", "四臂未收敛", "≠ clean SOTA")
    for phrase in required:
        require(phrase in content, f"missing label-protocol slide text: {phrase}")
    for obsolete in ("58 rebuilt / 95 UNCHECKED", "PDB58/95",
                     "全集无任何 full-coverage prediction", "ArchiveII full3966：5 full"):
        require(obsolete not in content, f"obsolete evidence value remains on slide: {obsolete}")
    return {"slide_count": 18, "deck_slide_index": 18, "added_slide_number": 37,
            "prior_slide_xml_preserved": 17, "shape_bounds_checks": len(reopened.slides[17].shapes),
            "new_slide_text_chars": len(content)}


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


def build(deck: str | Path, source_deck: str | Path = SOURCE_DECK,
          source_verification: str | Path = SOURCE_VERIFICATION,
          expected_source_sha256: str | None = None,
          backup: str | Path | None = None, verification: str | Path | None = None,
          **evidence_paths) -> dict:
    deck = Path(deck).resolve()
    source = validate_source(source_deck, source_verification, expected_source_sha256)
    require(deck != source["deck"], "refusing to modify the real round72 delivery; use a temporary copy")
    require(sha256(deck) == source["identity"]["sha256"],
            "temporary deck is not a byte-identical copy of the verified round72 delivery")
    backup = Path(backup).resolve() if backup else deck.with_name(deck.stem + ".pre_round73.pptx")
    verification = (Path(verification).resolve() if verification else
                    deck.with_name(deck.stem + ".round73_build_verification.json"))
    require(not backup.exists(), f"refusing to overwrite existing backup: {backup}")
    require(not verification.exists(), f"refusing to overwrite existing verification: {verification}")
    evidence = validate_evidence(**evidence_paths)
    prs = Presentation(deck)
    prior_sha256 = sha256(deck)
    add_label_slide(prs, evidence)
    descriptor, temporary = tempfile.mkstemp(prefix=deck.stem + ".round73.", suffix=".pptx", dir=deck.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        prs.save(temporary_path)
        layout = verify_built_deck(temporary_path, deck, evidence)
        require(sha256(deck) == prior_sha256, "temporary input changed before atomic replacement")
        require(sha256(source["deck"]) == source["identity"]["sha256"], "real round72 delivery changed")
        shutil.copy2(deck, backup)
        os.replace(temporary_path, deck)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    report = {
        "status": "PASS_BUILD_REOPEN_17_PRIOR_XML_ATOMIC_REPLACE_HASH_GATE",
        "deck": str(deck), "deck_sha256": sha256(deck), "deck_bytes": deck.stat().st_size,
        "prior_deck_sha256": prior_sha256, "backup": str(backup), "backup_sha256": sha256(backup),
        "source_delivery": source["identity"], "source_verification": identity(source["verification"]),
        "source_delivery_unchanged": sha256(source["deck"]) == source["identity"]["sha256"],
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve()),
        "evidence_artifacts": [identity(path) for path in evidence["paths"]],
        "scope": {"legacy_prediction_derived_labels_sota_eligible": False,
                  "official_reference_descriptive_only": True, "pdb62_91_ranking_allowed": False,
                  "archiveii_full3966_full_coverage_identities": 8,
                  "archiveii_full3966_xens2_ranking_allowed": False,
                  "archiveii_full3966_xens2_gpu_queued": True,
                  "archiveii_full3966_xens2_theoretical_max": 3950,
                  "ts0_testsetb_official_full_ranking_allowed": False,
                  "ts0_official_full_coverage_identities": 7,
                  "testsetb_official_full_coverage_identities": 7,
                  "rfam_temporal_novel_supported": 269, "rfam_temporal_unchecked": 110,
                  "rfam_temporal_blanket_qualified": False,
                  "pretrain_unknown_or_unchecked_clean": False, "four_arm_converged": False,
                  "complete_sota_closed": False, "model_selection": False},
        "atomic_replace": True, "temporary_copy_only": True, "real_ppt_modified": False,
        **layout, "render_visual_check": "PENDING_SEPARATE_RENDER_AND_OCR",
    }
    atomic_write_json(verification, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True, help="temporary byte-identical delivery copy")
    parser.add_argument("--source-deck", type=Path, default=SOURCE_DECK)
    parser.add_argument("--source-verification", type=Path, default=SOURCE_VERIFICATION)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--data-registry", type=Path, default=DATA_REGISTRY)
    parser.add_argument("--archive-lineage", type=Path, default=ARCHIVE_LINEAGE)
    parser.add_argument("--archive-system", type=Path, default=ARCHIVE_SYSTEM)
    parser.add_argument("--archive-recommendation", type=Path, default=ARCHIVE_RECOMMENDATION)
    parser.add_argument("--archive-full-system", type=Path, default=ARCHIVE_FULL_SYSTEM)
    parser.add_argument("--archive-full-check", type=Path, default=ARCHIVE_FULL_CHECK)
    parser.add_argument("--archive-xens2-queue", type=Path, default=ARCHIVE_XENS2_QUEUE)
    parser.add_argument("--bprna-system", type=Path, default=BPRNA_SYSTEM)
    parser.add_argument("--pretrain-summary", type=Path, default=PRETRAIN_SUMMARY)
    parser.add_argument("--pdb-lineage", type=Path, default=PDB_LINEAGE)
    parser.add_argument("--pdb-summary", type=Path, default=PDB_SUMMARY)
    parser.add_argument("--pdb-verification", type=Path, default=PDB_VERIFICATION)
    parser.add_argument("--ts-coverage", type=Path, default=TS_COVERAGE)
    parser.add_argument("--ts-verification", type=Path, default=TS_VERIFICATION)
    parser.add_argument("--ts0-recovery", type=Path, default=TS0_RECOVERY)
    parser.add_argument("--ts0-recovery-check", type=Path, default=TS0_RECOVERY_CHECK)
    parser.add_argument("--testsetb-recovery", type=Path, default=TESTSETB_RECOVERY)
    parser.add_argument("--testsetb-recovery-check", type=Path, default=TESTSETB_RECOVERY_CHECK)
    parser.add_argument("--rfam-registry", type=Path, default=RFAM_REGISTRY)
    parser.add_argument("--rfam-verification", type=Path, default=RFAM_VERIFICATION)
    args = parser.parse_args(argv)
    evidence_paths = {name: getattr(args, name) for name in
                      ("data_registry", "archive_lineage", "archive_system", "archive_recommendation",
                       "archive_full_system", "archive_full_check", "archive_xens2_queue",
                       "bprna_system", "pretrain_summary", "pdb_lineage", "pdb_summary",
                       "pdb_verification", "ts_coverage", "ts_verification", "ts0_recovery",
                       "ts0_recovery_check", "testsetb_recovery", "testsetb_recovery_check",
                       "rfam_registry", "rfam_verification")}
    try:
        report = build(args.deck, args.source_deck, args.source_verification,
                       args.expected_source_sha256, args.backup, args.verification, **evidence_paths)
    except (EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"round73 build refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
