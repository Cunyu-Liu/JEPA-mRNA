#!/usr/bin/env python3
"""Build Phase 0 slide 33 only from a final hash-verified four-arm evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
BACKUP = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.pre_round70.pptx"
VERIFICATION = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_build_verification.json"
EVALUATION_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_independent_eval_v1/generation_01")
SCHEMA = "foldnar.structured_margin_independent_eval.v1"
ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
CONTROLS = ("matched_control", "destroy_interval_shuffle", "independent_context")
METRICS = (
    "micro_pair_precision", "micro_pair_recall", "micro_pair_f1",
    "macro_pair_precision", "macro_pair_recall", "macro_pair_f1",
    "micro_stem_precision", "micro_stem_recall", "micro_stem_f1",
    "macro_stem_precision", "macro_stem_recall", "macro_stem_f1",
)
CORE_METRICS = ("micro_pair_f1", "macro_pair_f1", "micro_stem_f1", "macro_stem_f1")
TARGETS = ("isolated_pair:isolated", "pair_span:ge128")
ARM_LABELS = {
    "matched_control": "control",
    "ensemble_context": "ensemble",
    "destroy_interval_shuffle": "destroy",
    "independent_context": "independent",
}
CONTROL_LABELS = {
    "matched_control": "ensemble − control",
    "destroy_interval_shuffle": "ensemble − destroy",
    "independent_context": "ensemble − independent",
}
BURGUNDY = RGBColor(122, 32, 44)
DARK_RED = RGBColor(145, 36, 48)
GREEN = RGBColor(36, 105, 78)
INK = RGBColor(30, 25, 28)
MUTED = RGBColor(105, 96, 101)
ROSE = RGBColor(247, 241, 239)
PALE = RGBColor(250, 245, 246)
FAIL_PALE = RGBColor(252, 235, 237)
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
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict:
    require(path.is_file(), f"required final artifact missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EligibilityError(f"invalid final JSON {path}: {error}") from error
    require(isinstance(value, dict), f"final JSON must be an object: {path}")
    return value


def collect_identities(value: object, found: dict[str, dict], location: str = "root") -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = value["path"]
            require(isinstance(path, str) and Path(path).is_absolute(),
                    f"non-absolute artifact identity at {location}")
            require(isinstance(value["bytes"], int) and not isinstance(value["bytes"], bool)
                    and value["bytes"] >= 0, f"invalid byte count at {location}")
            require(isinstance(value["sha256"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
                    f"invalid SHA-256 at {location}")
            recorded = {"path": path, "bytes": value["bytes"], "sha256": value["sha256"]}
            if path in found:
                require(found[path] == recorded, f"conflicting identities for {path}")
            found[path] = recorded
        for key, child in value.items():
            collect_identities(child, found, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_identities(child, found, f"{location}[{index}]")


def verify_identities(*values: dict) -> list[dict]:
    found: dict[str, dict] = {}
    for value in values:
        collect_identities(value, found)
    require(found, "no hash-verifiable source identities in final evaluation")
    verified = []
    for path, recorded in sorted(found.items()):
        source = Path(path)
        require(source.is_file(), f"recorded source missing: {path}")
        require(source.stat().st_size == recorded["bytes"], f"source byte count changed: {path}")
        require(sha256(source) == recorded["sha256"], f"source SHA-256 changed: {path}")
        verified.append(recorded)
    return verified


def validate_final_evaluation(root: str | Path) -> dict:
    root = Path(root).resolve()
    evaluation_path = root / "independent_evaluation.json"
    summary_path = root / "gate_summary.json"
    manifest_path = root / "input_manifest.json"
    evaluation = load_json(evaluation_path)
    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    require(evaluation.get("schema_version") == summary.get("schema_version") == SCHEMA,
            "not a structured-margin independent final evaluation")
    require(evaluation.get("final_verdict_allowed") is True,
            "final verdict is not allowed; IN_PROGRESS/engineering output cannot build slide 33")
    status = evaluation.get("status")
    require(status in {"PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL",
                       "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"},
            f"non-final or engineering status refused: {status}")
    overall = evaluation.get("overall_pass")
    require(type(overall) is bool and summary.get("overall_pass") is overall,
            "four-arm overall verdict is absent or inconsistent")
    require(status == ("PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if overall
                       else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"),
            "status and scientific verdict disagree")
    require(summary.get("status") == status, "gate summary status differs from final evaluation")
    require(set(evaluation.get("arm_metrics", {})) == set(ARMS), "final evaluation is not four-arm")
    require(summary.get("arm_metrics") == evaluation["arm_metrics"],
            "gate summary arm metrics differ from final evaluation")
    for arm in ARMS:
        require(set(evaluation["arm_metrics"][arm]) == set(METRICS), f"incomplete core metrics: {arm}")
        require(all(type(evaluation["arm_metrics"][arm][metric]) in (int, float)
                    and math.isfinite(evaluation["arm_metrics"][arm][metric]) for metric in METRICS),
                f"non-finite core metric: {arm}")
        require(evaluation.get("validation_audit", {}).get(arm, {}).get("status") == "STOPPED"
                and evaluation["validation_audit"][arm].get("pass") is True,
                f"validation is not terminal and qualified: {arm}")
        require(evaluation.get("checkpoint_audit", {}).get(arm, {}).get("pass") is True,
                f"checkpoint qualification did not pass: {arm}")
    gates = evaluation.get("margin_gates", {})
    require(type(gates.get("pass")) is bool and gates["pass"] is overall,
            "scientific gate verdict mismatch")
    require(len(gates.get("gates", {})) == 16
            and all(type(value) is bool for value in gates["gates"].values()),
            "final evaluation must contain exactly 16 boolean scientific gates")
    failed = [name for name, passed in gates["gates"].items() if not passed]
    require(gates.get("failed_gates") == failed and summary.get("failed_margin_gates") == failed,
            "failed-gate lists are inconsistent")
    require(evaluation.get("independence", {}).get("run_meta_read") is False
            and evaluation["independence"].get("official_evaluator_imported") is False
            and evaluation["independence"].get("validation_loss_nll_comparison")
            == "FORBIDDEN_DIFFERENT_OBJECTIVES", "independence contract is not satisfied")
    require(evaluation.get("execution", {}).get("metric_device") == "cpu"
            and evaluation["execution"].get("model_inference_during_evaluation") is False,
            "final evaluation execution contract is not satisfied")
    nll = evaluation.get("qualified_nll_reference", {})
    require(nll.get("same_frozen_mea") is True
            and nll.get("validation_loss_comparison_allowed") is False
            and nll.get("reference_role") == "QUALIFIED_NLL_STRUCTURE_REFERENCE_ONLY",
            "QUALIFIED NLL same-MEA reference contract is not satisfied")
    require(set(evaluation.get("margin_prediction_identities", {})) == set(ARMS),
            "final raw predictions do not cover four arms")
    for arm in ARMS:
        expected = (root / "predictions" / arm / "dev_predictions.jsonl").resolve()
        require(Path(evaluation["margin_prediction_identities"][arm].get("path", "")).resolve() == expected,
                f"unexpected final prediction path: {arm}")
    require(set(evaluation.get("comparisons", {})) == set(CONTROLS),
            "ensemble comparisons do not cover all three controls")
    require(set(evaluation.get("margin_vs_qualified_nll_same_arm", {})) == set(ARMS),
            "same-arm NLL comparisons are incomplete")
    nll_global = {}
    for arm in ARMS:
        comparison = evaluation["margin_vs_qualified_nll_same_arm"][arm]
        require(set(comparison.get("metrics", {})) == set(METRICS), f"incomplete NLL metrics: {arm}")
        require(all(target in comparison.get("subgroups", {}) for target in TARGETS),
                f"incomplete NLL target subgroups: {arm}")
        nll_global[arm] = comparison["metrics"]
    require(summary.get("margin_vs_nll_global") == nll_global,
            "gate summary NLL comparisons differ from final evaluation")
    for control in CONTROLS:
        comparison = evaluation["comparisons"][control]
        require(set(comparison.get("metrics", {})) == set(METRICS), f"incomplete comparison: {control}")
        require(all(target in comparison.get("subgroups", {}) for target in TARGETS),
                f"incomplete target comparison: {control}")
    require(all(target in evaluation.get("targets", {}) for target in TARGETS),
            "isolated/long-span targets are incomplete")
    panel = evaluation.get("panel", {})
    require(panel.get("panel_id") == "tr1c_cluster80_dev_v1"
            and type(panel.get("n_sequences")) is int and panel["n_sequences"] > 0
            and type(panel.get("n_pairs")) is int and panel["n_pairs"] > 0,
            "unexpected or empty model-dev panel")
    required_inputs = {"config", "source_contract", "registry", "raw_tr1c", "panel_records",
                       "dev_ids", "train_ids", "candidate_contract"}
    require(required_inputs.issubset(manifest.get("inputs", {})), "final input manifest is incomplete")
    verified = verify_identities(evaluation, manifest)
    return {"root": root, "evaluation": evaluation, "summary": summary,
            "paths": (evaluation_path, summary_path, manifest_path),
            "verified_identities": verified}


def fmt(value: float) -> str:
    return f"{value:.6f}"


def signed(value: float) -> str:
    return f"{value:+.6f}"


def ci(value: list[float]) -> str:
    require(isinstance(value, list) and len(value) == 2
            and all(type(item) in (int, float) and math.isfinite(item) for item in value),
            "invalid paired CI")
    return f"[{value[0]:+.4f}, {value[1]:+.4f}]"


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
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y),
        Inches(sum(widths)), Inches(row_height * (len(rows) + 1)),
    )
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
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.005)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.font.name = SANS if column_index == 0 else "Arial"
            paragraph.font.size = Pt(font_size)
            paragraph.font.bold = row_index == 0 or row_index == emphasis_row
            paragraph.font.color.rgb = WHITE if row_index == 0 else INK
    return tbl


def slide_text(slide) -> str:
    values = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            values.append(shape.text_frame.text)
        if shape.has_table:
            values.extend(cell.text for row in shape.table.rows for cell in row.cells)
    return "\n".join(values)


def add_slide(prs: Presentation, bundle: dict) -> None:
    evaluation = bundle["evaluation"]
    metrics = evaluation["arm_metrics"]
    passed = evaluation["overall_pass"]
    accent = GREEN if passed else DARK_RED
    panel = evaluation["panel"]
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.78, 0.25, 11.8, 0.28,
         "33 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT", 10.5, accent, True)
    title = ("结构化 margin 通过独立四臂验收" if passed else "结构化 margin 未通过独立四臂验收")
    text(slide, 0.78, 0.63, 11.9, 0.52, title, 25, INK, True, SERIF)
    text(slide, 0.78, 1.20, 11.9, 0.30,
         f"MODEL-DEV ONLY · {panel['n_sequences']:,} sequences / {panel['n_pairs']:,} pairs · seed0 · 固定 MEA · 独立原始 pair 重算",
         10.8, MUTED, font=SANS)
    rows = [[ARM_LABELS[arm]] + [fmt(metrics[arm][metric]) for metric in CORE_METRICS] for arm in ARMS]
    table(slide, 0.78, 1.60, (3.18, 2.15, 2.15, 2.15, 2.15), 0.31,
          ("四臂核心结果", "micro pair-F1", "macro pair-F1", "micro stem-F1", "macro stem-F1"),
          rows, font_size=9.5, emphasis_row=2)
    text(slide, 0.78, 3.29, 11.7, 0.20,
         "ensemble 对三个对照：Δ = ensemble − 对照；isolated / span≥128 均为 micro pair recall。",
         9.1, MUTED, font=SANS)
    delta_rows = []
    for control in CONTROLS:
        comparison = evaluation["comparisons"][control]
        delta_rows.append([
            CONTROL_LABELS[control],
            signed(comparison["metrics"]["micro_pair_f1"]["delta"]),
            signed(comparison["metrics"]["micro_stem_f1"]["delta"]),
            signed(comparison["subgroups"]["isolated_pair:isolated"]["recall"]["delta"]),
            signed(comparison["subgroups"]["pair_span:ge128"]["recall"]["delta"]),
        ])
    table(slide, 0.78, 3.54, (3.0, 2.10, 2.10, 2.25, 2.33), 0.30,
          ("配对比较", "Δ micro pair-F1", "Δ micro stem-F1", "Δ isolated recall", "Δ span≥128 recall"),
          delta_rows, font_size=9.2)
    text(slide, 0.78, 4.80, 7.85, 0.18,
         "同臂 Δ = margin − QUALIFIED NLL · 相同冻结 MEA · 禁止比较不同目标训练 loss",
         8.8, MUTED, font=SANS)
    nll_rows = []
    for arm in ARMS:
        comparison = evaluation["margin_vs_qualified_nll_same_arm"][arm]["metrics"]
        nll_rows.append([ARM_LABELS[arm], signed(comparison["micro_pair_f1"]["delta_margin_minus_nll"]),
                         ci(comparison["micro_pair_f1"]["paired_ci95"]),
                         signed(comparison["micro_stem_f1"]["delta_margin_minus_nll"])])
    table(slide, 0.78, 5.03, (1.75, 1.60, 2.55, 1.65), 0.225,
          ("同臂", "Δ pair-F1", "paired 95% CI", "Δ stem-F1"), nll_rows, font_size=8.3)
    gates = evaluation["margin_gates"]["gates"]
    passed_count = sum(gates.values())
    rect(slide, 8.58, 4.80, 3.98, 1.36, PASS_PALE if passed else FAIL_PALE, accent)
    text(slide, 8.80, 4.89, 3.54, 0.28, f"科学门控 {passed_count} / 16", 16, accent, True, SANS)
    text(slide, 8.80, 5.22, 3.54, 0.23, evaluation["status"], 8.6, accent, True)
    failed = evaluation["margin_gates"]["failed_gates"]
    verdict_note = ("16 项全部通过；仅限 registered model-dev，不能外推外部测试或 SOTA。" if passed else
                    f"科学 FAIL 如实记录 · {len(failed)} 项失败：{', '.join(failed[:2])}" +
                    (" …" if len(failed) > 2 else ""))
    text(slide, 8.80, 5.52, 3.50, 0.49, verdict_note, 9.0, INK, True, SANS)
    rect(slide, 0.78, 6.42, 11.78, 0.012, RGBColor(212, 165, 154)).line.fill.background()
    text(slide, 0.78, 6.51, 11.65, 0.18,
         f"evaluator {evaluation['code_identity']['git_head'][:12]} · panel={panel['panel_id']} · model-dev only · no external test · no decoder tuning",
         7.5, MUTED)
    text(slide, 0.78, 6.72, 11.65, 0.18,
         "final sources structured_margin_independent_eval_v1/generation_01/{independent_evaluation.json,gate_summary.json,input_manifest.json} · SHA-256 verified",
         7.5, MUTED)


def verify_built_deck(path: Path, prior_deck: Path, evaluation: dict) -> dict:
    reopened = Presentation(path)
    prior = Presentation(prior_deck)
    require(len(prior.slides) == 13, "source deck no longer has 13 slides")
    require(len(reopened.slides) == 14, "built deck must have exactly 14 slides")
    require([slide.part.blob for slide in list(reopened.slides)[:13]]
            == [slide.part.blob for slide in prior.slides], "prior slide XML changed")
    with zipfile.ZipFile(prior_deck) as before, zipfile.ZipFile(path) as after:
        for index in range(1, 14):
            part = f"ppt/slides/slide{index}.xml"
            require(before.read(part) == after.read(part), f"prior slide part changed: {part}")
    new_slide = reopened.slides[13]
    for shape in new_slide.shapes:
        require(shape.left >= 0 and shape.top >= 0, "negative shape position")
        require(shape.left + shape.width <= reopened.slide_width + 2, "shape exceeds slide width")
        require(shape.top + shape.height <= reopened.slide_height + 2, "shape exceeds slide height")
    joined = slide_text(new_slide)
    for arm in ARMS:
        require(ARM_LABELS[arm] in joined, f"missing arm label: {arm}")
        for metric in CORE_METRICS:
            require(fmt(evaluation["arm_metrics"][arm][metric]) in joined, f"missing {arm} {metric}")
    for control in CONTROLS:
        comparison = evaluation["comparisons"][control]
        for value in (comparison["metrics"]["micro_pair_f1"]["delta"],
                      comparison["metrics"]["micro_stem_f1"]["delta"],
                      comparison["subgroups"]["isolated_pair:isolated"]["recall"]["delta"],
                      comparison["subgroups"]["pair_span:ge128"]["recall"]["delta"]):
            require(signed(value) in joined, f"missing ensemble comparison: {control}")
    for phrase in ("MODEL-DEV ONLY", "QUALIFIED NLL", "相同冻结 MEA", "isolated", "span≥128",
                   evaluation["status"], f"科学门控 {sum(evaluation['margin_gates']['gates'].values())} / 16",
                   "SHA-256 verified"):
        require(phrase in joined, f"missing required slide text: {phrase}")
    return {"slide_count": 14, "deck_slide_index": 14, "added_slide_number": 33,
            "prior_slide_xml_preserved": 13, "shape_bounds_checks": len(new_slide.shapes),
            "new_slide_text_chars": len(joined)}


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


def build(deck: str | Path = DECK, evaluation_root: str | Path = EVALUATION_ROOT,
          backup: str | Path | None = None, verification: str | Path | None = None) -> dict:
    deck = Path(deck).resolve()
    backup = Path(backup).resolve() if backup else deck.with_name(deck.stem + ".pre_round70.pptx")
    verification = (Path(verification).resolve() if verification else
                    deck.with_name(deck.stem + ".round70_build_verification.json"))
    bundle = validate_final_evaluation(evaluation_root)
    require(deck.is_file(), f"deck missing: {deck}")
    prs = Presentation(deck)
    require(len(prs.slides) == 13, "expected 13-slide Phase 0 deck; refusing duplicate append")
    require(not backup.exists(), f"refusing to overwrite existing backup: {backup}")
    prior_sha256 = sha256(deck)
    add_slide(prs, bundle)
    descriptor, temporary = tempfile.mkstemp(prefix=deck.stem + ".round70.", suffix=".pptx", dir=deck.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        prs.save(temporary_path)
        layout = verify_built_deck(temporary_path, deck, bundle["evaluation"])
        shutil.copy2(deck, backup)
        os.replace(temporary_path, deck)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    reopened_sha256 = sha256(deck)
    report = {
        "status": "PASS_BUILD_REOPEN_13_PRIOR_XML_ATOMIC_REPLACE_HASH_GATE",
        "scientific_status": bundle["evaluation"]["status"],
        "scientific_overall_pass": bundle["evaluation"]["overall_pass"],
        "deck": str(deck), "deck_sha256": reopened_sha256, "deck_bytes": deck.stat().st_size,
        "prior_deck_sha256": prior_sha256, "backup": str(backup), "backup_sha256": sha256(backup),
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve()),
        "evaluation_root": str(bundle["root"]),
        "final_json": {str(path): identity(path) for path in bundle["paths"]},
        "verified_source_identities": bundle["verified_identities"],
        "evaluator_git_head": bundle["evaluation"]["code_identity"]["git_head"],
        "panel": bundle["evaluation"]["panel"], "atomic_replace": True,
        **layout, "render_visual_check": "PENDING_SEPARATE_RENDER_AND_OCR",
    }
    atomic_write_json(verification, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build(args.deck, args.evaluation_root, args.backup, args.verification)
    except (EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"round70 build refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
