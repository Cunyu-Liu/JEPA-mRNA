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
BPRNA_SYSTEM = ROOT / "source_data/bprna_official_reference_5388/system_metrics.json"
PRETRAIN_SUMMARY = ROOT / "manifests/pretrain_exposure_v1/summary.json"
PDB_LINEAGE = ROOT / "source_data/pdb_rfam_label_lineage_audit_20261127/panel_summary.json"
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
                      bprna_system: str | Path = BPRNA_SYSTEM,
                      pretrain_summary: str | Path = PRETRAIN_SUMMARY,
                      pdb_lineage: str | Path = PDB_LINEAGE) -> dict:
    paths = tuple(Path(path).resolve() for path in
                  (data_registry, archive_lineage, archive_system, archive_recommendation,
                   bprna_system, pretrain_summary, pdb_lineage))
    registry, lineage, archive, recommendation, bprna, pretrain, pdb = map(load_json, paths)
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
    pdb_rebuild = registry.get("pdb_raw_source_rebuild", {})
    require(pdb_rebuild.get("status") == "COMPLETE_58_REBUILT_95_UNCHECKED"
            and pdb_rebuild.get("n_source_records") == 153
            and pdb_rebuild.get("n_rebuilt") == 58
            and pdb_rebuild.get("n_unresolved") == 95
            and pdb_rebuild.get("external_test_policy") == "AUDIT_ONLY_NO_MODEL_SELECTION"
            and "no prediction metrics or model ranking authorized" in pdb_rebuild.get("ranking_scope", ""),
            "PDB58/95 rebuild/no-ranking contract changed")
    require(sum(pdb_rebuild.get("unresolved_status_counts", {}).values()) == 95,
            "PDB unresolved status counts do not sum to 95")
    summary = pdb_rebuild.get("summary", {})
    summary_path = ROOT / summary.get("path", "")
    actual_summary = identity(summary_path)
    require(summary.get("bytes") == actual_summary["bytes"]
            and summary.get("sha256") == actual_summary["sha256"],
            "PDB rebuild summary identity mismatch")
    require(pdb.get("related_temporal_pdb_collection", {}).get("status") == "NOT_QUALIFIED"
            and pdb.get("negative_label_policy", {}).get("unobserved") ==
            "must be masked, never a negative label",
            "PDB observation-mask qualification changed")
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
            "pdb_rebuild": pdb_rebuild, "pretrain": pretrain,
            "paths": (*paths, summary_path)}


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
    text(slide, 8.76, 1.73, 3.58, 0.25, "③ PDB：58 rebuilt / 95 UNCHECKED", 11.4, DARK_RED, True)
    text(slide, 8.76, 2.03, 3.58, 0.56,
         "153 candidates · no model ranking\n未观测位点 mask ≠ 负标签；不得排名。", 9.4, INK, font=SANS)
    xens2 = evidence["xens2"]["metrics"]["micro_f1"]
    rows = []
    for label, key, role in (("xens2 · frozen project ensemble", "xens2", "reference"),
                             ("BPfold replay · six-checkpoint", "replay", "strongest overall*"),
                             ("RiNALMo-ft · single", "rinalmo", "strongest single*")):
        value = evidence[key]["metrics"]["micro_f1"]
        delta = "reference" if key == "xens2" else signed10(value - xens2)
        rows.append([label, fmt10(value), delta, role])
    table(slide, 0.78, 3.00, (4.10, 2.00, 2.12, 3.56), 0.43,
          ("ArchiveII official clean2544", "micro-F1", "Δ vs xens2", "descriptive role"), rows)
    text(slide, 0.78, 4.78, 11.70, 0.18,
         "* official-reference 新差距仅描述同一全覆盖面板；不是模型选择，也不自动继承 clean SOTA 资格。", 8.3, MUTED, font=SANS)
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
         "LABEL_REFERENCE_REBUILD_ONLY · PDB58/95不得排名 · legacy prediction-derived labels不用于独立SOTA · raw-exposure descriptive ≠ clean SOTA",
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
                "+0.0842809284", "+0.0048000324", "strongest single", "58 rebuilt / 95 UNCHECKED",
                "不得排名", "5381/5388", "2048/2544", "UNKNOWN / UNCHECKED ≠ clean",
                "完整 SOTA 仍未闭环", "四臂未收敛", "≠ clean SOTA")
    for phrase in required:
        require(phrase in content, f"missing label-protocol slide text: {phrase}")
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
                  "official_reference_descriptive_only": True, "pdb58_95_ranking_allowed": False,
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
    parser.add_argument("--bprna-system", type=Path, default=BPRNA_SYSTEM)
    parser.add_argument("--pretrain-summary", type=Path, default=PRETRAIN_SUMMARY)
    parser.add_argument("--pdb-lineage", type=Path, default=PDB_LINEAGE)
    args = parser.parse_args(argv)
    evidence_paths = {name: getattr(args, name) for name in
                      ("data_registry", "archive_lineage", "archive_system", "archive_recommendation",
                       "bprna_system", "pretrain_summary", "pdb_lineage")}
    try:
        report = build(args.deck, args.source_deck, args.source_verification,
                       args.expected_source_sha256, args.backup, args.verification, **evidence_paths)
    except (EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"round73 build refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
