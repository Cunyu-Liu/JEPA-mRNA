#!/usr/bin/env python3
"""Read-only provenance checks for the 2026-11-26 scientific story ledger."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/plans/2026-11-26-gec-margin-scientific-story-ledger.md"
F = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
A = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts")
MARGIN_REPO = Path("/home/rna_jev_margin_candidate")

SOURCES = {
    "capacity": A / "representation_head_mechanism_4arm_v1/capacity_freeze_audit.json",
    "srd": A / "structured_redecoding_v1/seed0_control_matched/srd_summary.json",
    "stacking": A / "stacking_closure_form_v1/pilot/evaluation/pilot_verdict.json",
    "gec": F / "gec_seed0_v3_independent_reeval_v1/independent_reeval.json",
    "residual": F / "gec_residual_error_audit_v1/ensemble_context_summary.json",
    "residual_verification": F / "gec_residual_error_audit_v1/verification.json",
    "residual_comparison": F / "gec_residual_error_audit_v1/comparison.json",
    "crossover": F / "gec_frozen_context_crossover_v1/full/summary.json",
    "crossover_verification": F / "gec_frozen_context_crossover_v1/full/independent_verification.json",
    "ppt33_record": Path("/home/rna_jev_foldnar_integration/ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_build_verification.json"),
    "page34_builder": Path("/home/rna_jev_foldnar_integration/ppt/add_phase0_structured_margin_result_round71.py"),
    "nll": F / "gec_seed0_v3_requalified_v1/qualification_summary.json",
    "margin_startup": F / "structured_margin_full_v1/generation_01/startup_report_v1.json",
    "margin_authorization": F / "structured_margin_full_v1/full_authorization_v1.json",
    "margin_scheduler": F / "structured_margin_full_v1/generation_01/scheduler_state.json",
    "margin_config": MARGIN_REPO / "configs/gec_structured_margin_convergence_v1.json",
}


def pointer(document, path: str):
    value = document
    for part in path.split("/") if path else ():
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def missing_sources() -> list[str]:
    return [f"{name}: {path}" for name, path in SOURCES.items() if not path.is_file()]


def verify_links() -> list[str]:
    errors = []
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", DOC.read_text()):
        if target.startswith(("http://", "https://", "/", "#")):
            continue
        path = target.split("#", 1)[0]
        if path and not (DOC.parent / path).resolve().is_file():
            errors.append(f"broken relative link: {target}")
    return errors


def verify_narrative_guards() -> list[str]:
    text = DOC.read_text()
    required = [
        "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED",
        "STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE",
        "STOP_STACKING_CLOSURE_FORM_SEED0",
        "FAIL_GEC_SEED0_INDEPENDENT_REEVAL",
        "不能解释成随机扰动有益",
        "不证明 pair-Hamming structured margin",
        "original_scientific_fail_unchanged=true",
        "上下文有因果作用",
        "真实 ensemble 一致有益",
        "810441e7480ea06a717097c57a49130c820c9636",
        "03f7f4e14ef906359cbd9d1b519089b6e61116f4",
        "PPT 第 33 页",
        "PPT 第 34 页",
        "margin_superior_to_nll_claim_allowed=false",
        "完成臂 0",
        "scientific_conclusion=null",
        "只有对应**直接、已完成、可回溯**的证据全部存在",
    ]
    return [f"missing narrative guard: {item}" for item in required if item not in text]


def verify_numbers() -> list[str]:
    errors = []
    if missing_sources():
        return [f"missing JSON source: {item}" for item in missing_sources()]
    data = {name: None if name == "page34_builder" else json.loads(path.read_text()) for name, path in SOURCES.items()}
    text = DOC.read_text()
    checks = []

    def check(label: str, actual, expected, snippet: str | None = None):
        if isinstance(expected, float):
            valid = isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = actual == expected
        if not valid:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")
        if snippet is not None and snippet not in text:
            errors.append(f"{label}: ledger lacks rendered value {snippet}")

    check("capacity decision", pointer(data["capacity"], "decision"), "STOP_REPRESENTATION_HEAD_CAPACITY_THREE_SEED")
    for arm, micro, macro in [
        ("head_deep", -0.003938939080463388, -0.009905280737087593),
        ("repr_head", -0.03451471909687084, -0.05192656892120199),
    ]:
        check(f"capacity {arm} micro", pointer(data["capacity"], f"mean_delta_vs_control_matched/{arm}/micro_pair_f1"), micro)
        check(f"capacity {arm} macro", pointer(data["capacity"], f"mean_delta_vs_control_matched/{arm}/macro_pair_f1"), macro)

    check("srd decision", pointer(data["srd"], "decision"), "STOP_SRD_NOT_BETTER_THAN_CURRENT_MEA_OR_CONSTRAINT_FAILURE")
    check("srd micro", pointer(data["srd"], "arms/nested_srd_stacking/micro_pair_f1"), 0.8002104155707522, "0.800210")
    check("mea micro", pointer(data["srd"], "arms/current_mea/micro_pair_f1"), 0.8257126886528787, "0.825713")
    check("srd multimodel", pointer(data["srd"], "multimodel_status"), "NOT_RUN_SINGLE_SOURCE_STOP")

    check("stacking decision", pointer(data["stacking"], "decision"), "STOP_STACKING_CLOSURE_FORM_SEED0")
    check("stacking micro delta", pointer(data["stacking"], "delta_vs_local_control/micro_pair_f1"), 0.003989820889696238, "+0.0039898209")
    check("stacking macro delta", pointer(data["stacking"], "delta_vs_local_control/macro_pair_f1"), -0.005640591986496468, "−0.0056405920")
    check("stacking destroy micro", pointer(data["stacking"], "arms/destroy_offset/micro_pair_f1"), 0.8479153164830417, "0.8479153165")
    check("stacking seeds closed", pointer(data["stacking"], "seed1_seed2_allowed"), False)

    check("GEC status", pointer(data["gec"], "status"), "FAIL_GEC_SEED0_INDEPENDENT_REEVAL")
    check("GEC overall", pointer(data["gec"], "overall_pass"), False)
    check("GEC panel", pointer(data["gec"], "panel/n_sequences"), 9143, "9,143")
    for arm, micro, macro in [
        ("ensemble_context", 0.6812310470423341, 0.6746040481514581),
        ("destroy_interval_shuffle", 0.7077029737291702, 0.7360553532532974),
    ]:
        check(f"GEC {arm} micro", pointer(data["gec"], f"arm_metrics/{arm}/micro_pair_f1"), micro)
        check(f"GEC {arm} macro", pointer(data["gec"], f"arm_metrics/{arm}/macro_pair_f1"), macro)
    check("GEC destroy micro delta", pointer(data["gec"], "comparisons/destroy_interval_shuffle/metrics/micro_pair_f1/delta"), -0.026471926686836178, "−0.0264719267")

    group = pointer(data["residual"], "groups/true_all")
    check("residual truth", group["truth_pairs"], 247013)
    check("residual selected", group["selected"], 144110)
    check("residual nonpositive", group["nonpositive_score_margin"], 101164, "101,164")
    check("residual competition", group["positive_weight_mea_competition"], 1739, "1,739")
    check("residual conservation", group["truth_pairs"] - group["selected"], 102903, "102,903")
    ratio = group["nonpositive_score_margin"] / (group["nonpositive_score_margin"] + group["positive_weight_mea_competition"])
    check("residual ratio", ratio, 0.9831005898759025, "98.31005899%")
    check("residual verification", pointer(data["residual_verification"], "status"), "PASS")
    check("residual failure unchanged", pointer(data["residual_comparison"], "independent_seed0_failure_remains"), True)

    check("crossover commit", pointer(data["crossover"], "git_commit"), "810441e7480ea06a717097c57a49130c820c9636")
    check("crossover status", pointer(data["crossover"], "status"), "COMPLETE")
    check("crossover records", pointer(data["crossover"], "record_count"), 9143)
    check("crossover combinations", len(pointer(data["crossover"], "artifacts")), 16)
    check("crossover GEC pass", pointer(data["crossover"], "gec_pass_claim"), False)
    check("crossover seeds closed", pointer(data["crossover"], "seeds_1_2_authorized"), False)
    for context, expected, snippet in [
        ("ensemble_context", 0.6812310470423341, "0.6812310470"),
        ("matched_control", 0.6993017226005002, "0.6993017226"),
        ("destroy_interval_shuffle", 0.6871289606495534, "0.6871289606"),
    ]:
        check(f"crossover ensemble model {context} input", pointer(data["crossover"], f"metrics/ensemble_context__{context}/micro_pair_f1"), expected, snippet)
    for arm, expected, snippet in [
        ("matched_control", -0.10410194955059948, "−0.1041019496"),
        ("ensemble_context", -0.1932951913581875, "−0.1932951914"),
        ("destroy_interval_shuffle", -0.28977659820091917, "−0.2897765982"),
    ]:
        path = f"comparisons/{arm}/each_context_s1_minus_matched_s1/independent_context/micro_pair_f1"
        check(f"crossover {arm} independent delta", pointer(data["crossover"], path), expected, snippet)
        check(f"crossover verification {arm} independent delta", pointer(data["crossover_verification"], path), expected)
        check(f"crossover native {arm}", pointer(data["crossover"], f"native_prediction_validation/{arm}/pass"), True)
        check(f"crossover native {arm} records", pointer(data["crossover"], f"native_prediction_validation/{arm}/matched_records"), 9143)
    check("crossover native independent", pointer(data["crossover"], "native_prediction_validation/independent_context/pass"), True)
    check("crossover verification status", pointer(data["crossover_verification"], "status"), "PASS")
    check("crossover raw recomputation", pointer(data["crossover_verification"], "independent_raw_truth_recomputation"), True)
    check("crossover no official aggregate", pointer(data["crossover_verification"], "official_aggregate_read"), False)
    check("PPT33 page", pointer(data["ppt33_record"], "added_slide_number"), 33)
    check("PPT33 crossover commit", pointer(data["ppt33_record"], "crossover_commit"), "810441e7480ea06a717097c57a49130c820c9636")
    check("PPT33 native value", pointer(data["ppt33_record"], "source_values/ensemble_model_micro_pair_f1_by_input/ensemble_context"), 0.6812310470423341)
    page34 = SOURCES["page34_builder"].read_text()
    for required in ["final hash-verified four-arm evaluation", "34 · FOLDNAR PHASE 1 / STRUCTURED MARGIN FOUR-ARM VERDICT", "added_slide_number"]:
        if required not in page34:
            errors.append(f"PPT34 builder lacks final-evidence guard: {required}")

    check("NLL qualification", pointer(data["nll"], "qualification_pass"), True)
    check("NLL final commit", pointer(data["nll"], "code_identity/final_head"), "ecc40d3a8ec545bf332199a935e6380b718d415b")
    check("NLL science unchanged", pointer(data["nll"], "reference_use/original_scientific_fail_unchanged"), True)
    check("NLL gates unchanged", pointer(data["nll"], "reference_use/scientific_gate_recomputed_or_changed"), False)
    check("NLL no superiority", pointer(data["nll"], "reference_use/margin_superior_to_nll_claim_allowed"), False)
    for arm in ["matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"]:
        check(f"NLL {arm}", pointer(data["nll"], f"per_arm/{arm}/qualification"), "QUALIFIED_STRUCTURE_REFERENCE")
        check(f"NLL {arm} best", pointer(data["nll"], f"per_arm/{arm}/best_step"), 10500)
        check(f"NLL {arm} stop", pointer(data["nll"], f"per_arm/{arm}/stop_step"), 13000)

    check("margin queued snapshot", pointer(data["margin_startup"], "scheduler_state/status"), "QUEUED_PROBE_MEMORY")
    check("margin scheduler PID", pointer(data["margin_startup"], "scheduler_pid"), 3976280, "3976280")
    check("margin no completed arms", pointer(data["margin_startup"], "scheduler_state/completed_arms"), [], "完成臂 0")
    check("margin no conclusion at startup", pointer(data["margin_startup"], "scientific_conclusion"), None)
    check("margin no authorized conclusion", pointer(data["margin_authorization"], "scientific_conclusion"), None)
    check("margin latest scheduler status", pointer(data["margin_scheduler"], "status"), "QUEUED_ARM_MEMORY")
    check("margin latest completed arms", pointer(data["margin_scheduler"], "completed_arms"), [], "完成臂仍为 0")
    check("margin latest remaining arms", pointer(data["margin_scheduler"], "remaining_arms"), ["destroy_interval_shuffle", "independent_context"])
    check("margin latest conclusion", pointer(data["margin_scheduler"], "scientific_conclusion"), None)
    for arm in ["matched_control", "ensemble_context"]:
        check(f"margin latest {arm} running", pointer(data["margin_scheduler"], f"processes/{arm}/status"), "running")
    check("margin code", pointer(data["margin_authorization"], "code_commit"), "a846d90519e575a79b7db76f63692afb755e8ac6")
    check("margin NLL reference", pointer(data["margin_authorization"], "historical_nll_status"), "QUALIFIED_STRUCTURE_REFERENCE")
    check("margin arms", pointer(data["margin_config"], "arms"), ["matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context"])
    check("margin lambda NLL", pointer(data["margin_config"], "training/lambda_nll"), 0.0)
    check("margin no fixed cap", pointer(data["margin_config"], "training/fixed_step_cap"), None)
    check("margin no decoder search", pointer(data["margin_config"], "model/decoder/search_allowed"), False)

    for token in ["−0.0039389391", "−0.0519265689", "−0.025502", "+0.0064705882", "−0.0194305860", "0.6812310470", "0.6746040482", "0.7077029737", "0.7360553533", "0.5834105897", "0.1090534979"]:
        if token not in text:
            errors.append(f"ledger lacks core rendered number: {token}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-only", action="store_true", help="skip host-specific JSON artifacts")
    args = parser.parse_args()
    errors = verify_links() + verify_narrative_guards()
    if not args.links_only:
        errors += verify_numbers()
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    scope = "links and narrative guards" if args.links_only else "links, narrative guards, and JSON numeric provenance"
    print(f"PASS: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
