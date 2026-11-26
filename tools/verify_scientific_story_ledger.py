#!/usr/bin/env python3
"""Read-only provenance checks for the 2026-11-26 scientific story ledger."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/plans/2026-11-26-gec-margin-scientific-story-ledger.md"
F = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
A = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts")
MARGIN_REPO = Path("/home/rna_jev_margin_candidate")
MARGIN_EVALUATOR_REPO = Path("/home/rna_jev_margin_evaluator")
STRUCTRFM = F / "structrfm_official_v0.0.8"
FOLDNAR_REPO = Path("/home/rna_jev_foldnar_integration")
LABEL_QUALITY_REPO = Path("/home/rna_jev_label_quality")
LABEL_QUALITY = F / "label_quality_v1"
PRETRAIN_EXPOSURE_REPO = Path("/home/rna_jev_pretrain_exposure")
PRETRAIN_EXPOSURE = PRETRAIN_EXPOSURE_REPO / "manifests/pretrain_exposure_v1"
LINEARPARTITION_RUN = F / "longseq_linearpartition_contract_20261126/runs/linearpartition_c_19000_holdout_cpu_20261126/7AM2_1"

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
    "structrfm": STRUCTRFM / "reports/structrfm_official_bprna_new_metrics_20261126.json",
    "structrfm_provenance": STRUCTRFM / "reports/structrfm_official_provenance_20261126.json",
    "structrfm_delivery": STRUCTRFM / "reports/full_delivery_verification_20261126.json",
    "raw_baselines": FOLDNAR_REPO / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json",
    "linearpartition_run": LINEARPARTITION_RUN / "run_report.json",
    "linearpartition_metrics": LINEARPARTITION_RUN / "metrics.json",
    "linearpartition_verification": LINEARPARTITION_RUN / "independent_verification.json",
    "linearpartition_delivery": LINEARPARTITION_RUN.parent / "holdout_delivery_verification.json",
    "label_summary": LABEL_QUALITY / "summary.json",
    "label_conservation": LABEL_QUALITY / "deletion_conservation.json",
    "label_verification": LABEL_QUALITY / "independent_verification.json",
    "label_manifest": LABEL_QUALITY / "artifact_manifest.json",
    "exposure_summary": PRETRAIN_EXPOSURE / "summary.json",
    "exposure_assets": PRETRAIN_EXPOSURE / "asset_registry.json",
    "exposure_verification": PRETRAIN_EXPOSURE / "independent_verification.json",
}

SOURCE_COMMITS = {
    "FoldNAR integration": (FOLDNAR_REPO, "79a2ed45879f1f4e81d6719ed54600bf7f8da807"),
    "label quality": (LABEL_QUALITY_REPO, "3e64df111cffaeaf21add48094434123234ea40f"),
    "pretrain exposure": (PRETRAIN_EXPOSURE_REPO, "4247d1e4a971ceb8c63e3f8338e026a23c18941b"),
    "independent margin evaluator": (MARGIN_EVALUATOR_REPO, "8b1861f9f8c116250aba194799bf5ee7d9a4f39e"),
}


def pointer(document, path: str):
    value = document
    for part in path.split("/") if path else ():
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def missing_sources() -> list[str]:
    return [f"{name}: {path}" for name, path in SOURCES.items() if not path.is_file()]


def verify_source_commits() -> list[str]:
    errors = []
    for name, (repo, expected) in SOURCE_COMMITS.items():
        if not (repo / ".git").exists():
            errors.append(f"missing source repository: {name}: {repo}")
            continue
        actual = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if actual != expected:
            errors.append(f"{name} commit: expected {expected}, got {actual}")
    return errors


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
        "fd497e9819e39a29887ec26698a46c91284d230b",
        "5a7ccbd418f520eb44f02f0fdf6fbdf9f6ff238e",
        "真实 deck 为 14 张、记录至第 33 页",
        "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED",
        "不是我们的候选正向性能，也不是 clean SOTA",
        "0.19321803153859785",
        "0.15297454451260917",
        "这不是相对百分比，也不是我们的候选提升",
        "长序列只有 baseline 回答，GEC unsupported",
        "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION",
        "8,481,693,696 bytes",
        "53/73=0.726027397260274",
        "8/73=0.1095890410958904",
        "margin_superior_to_nll_claim_allowed=false",
        "完成臂 0",
        "scientific_conclusion=null",
        "只有对应**直接、已完成、可回溯**的证据全部存在",
        "3e64df111cffaeaf21add48094434123234ea40f",
        "TR1c `DATABASE` 与 CHANRG `COMPARATIVE_INFERRED` 均不能称为实验真值",
        "95,052 条 noncanonical",
        "10,250 条 short-span",
        "不自动删除",
        "硬冲突记录=0",
        "实际删除=0",
        "4247d1e4a971ceb8c63e3f8338e026a23c18941b",
        "EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626",
        "bpRNA-new EXACT=5,381/5,388",
        "完整官方预训练语料仍 `UNCHECKED`",
        "dev 不能外推 clean-OOD",
        "exposure-aware descriptive performance",
        "下一页必须暴露分层",
        "margin 双训练未完成",
        "/home/rna_jev_margin_evaluator",
        "8b1861f9f8c116250aba194799bf5ee7d9a4f39e",
        "不得干扰 GPU 或改原 16 门",
        "margin 仍无结论",
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

    check("structRFM status", pointer(data["structrfm"], "status"), "PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS")
    check("structRFM full ready", pointer(data["structrfm"], "full_ready"), True)
    check("structRFM sequences", pointer(data["structrfm"], "dataset/n_sequences"), 5388, "5,388")
    check("structRFM coverage", pointer(data["structrfm"], "metrics/coverage"), 1.0)
    check("structRFM TP", pointer(data["structrfm"], "metrics/tp"), 73864, "73,864")
    check("structRFM FP", pointer(data["structrfm"], "metrics/fp"), 62616, "62,616")
    check("structRFM FN", pointer(data["structrfm"], "metrics/fn"), 61320, "61,320")
    check("structRFM micro", pointer(data["structrfm"], "metrics/micro_f1"), 0.5437893868896873, "0.5437893868896873")
    check("structRFM macro", pointer(data["structrfm"], "metrics/macro_f1"), 0.5338105702966888, "0.5338105702966888")
    check("structRFM raw scope", pointer(data["structrfm"], "label_scope"), "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED")
    check("structRFM raw not clean", pointer(data["structrfm"], "raw_not_clean"), True)
    check("structRFM historical unchanged", pointer(data["structrfm"], "historical_prediction/overwritten"), False)
    check("structRFM single model", pointer(data["structrfm_provenance"], "model_identity/model_form"), "single_checkpoint_single_model")
    check("structRFM not ensemble", pointer(data["structrfm_provenance"], "model_identity/ensemble"), False)
    check("structRFM exposure unresolved", pointer(data["structrfm_provenance"], "training_exposure/raw_not_clean"), True)
    check("structRFM delivery status", pointer(data["structrfm_delivery"], "status"), "PASS_FULL_5388_GPU_INFERENCE_AND_INDEPENDENT_DELIVERY")
    check("structRFM delivery raw", pointer(data["structrfm_delivery"], "raw_not_clean"), True)
    check("structRFM delivery micro", pointer(data["structrfm_delivery"], "micro_f1"), 0.5437893868896873)
    check("structRFM delivery macro", pointer(data["structrfm_delivery"], "macro_f1"), 0.5338105702966888)

    check("raw registry status", pointer(data["raw_baselines"], "status"), "PASS_20_EXISTING_ASSETS_SHA_ID_METRICS")
    check("raw registry not clean", pointer(data["raw_baselines"], "scope_contract/raw_is_not_clean"), True)
    check("BPfold model form", pointer(data["raw_baselines"], "strongest_project_raw/model_form"), "six_checkpoint_ensemble")
    check("BPfold coverage", pointer(data["raw_baselines"], "strongest_project_raw/coverage"), 1.0)
    check("BPfold sequences", pointer(data["raw_baselines"], "strongest_project_raw/n_sequences"), 5388)
    check("BPfold micro", pointer(data["raw_baselines"], "strongest_project_raw/micro_f1"), 0.7370074184282851, "0.7370074184282851")
    check("CONTRAfold model form", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/model_form"), "deterministic_single_predictor")
    check("CONTRAfold coverage", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/coverage"), 1.0)
    check("CONTRAfold sequences", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/n_sequences"), 5388)
    check("CONTRAfold micro", pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/micro_f1"), 0.6967639314022964, "0.6967639314022964")
    structrfm_micro = pointer(data["structrfm"], "metrics/micro_f1")
    bpfold_gap = pointer(data["raw_baselines"], "strongest_project_raw/micro_f1") - structrfm_micro
    contrafold_gap = pointer(data["raw_baselines"], "strongest_project_raw_single_predictor/micro_f1") - structrfm_micro
    check("BPfold minus structRFM", bpfold_gap, 0.19321803153859785, "0.19321803153859785")
    check("CONTRAfold minus structRFM", contrafold_gap, 0.15297454451260917, "0.15297454451260917")

    check("LinearPartition run status", pointer(data["linearpartition_run"], "status"), "PASS_FULL_LENGTH_HOLDOUT_EXECUTION_AND_OUTPUT__FLOPS_NOT_MEASURED")
    check("LinearPartition split", pointer(data["linearpartition_run"], "split"), "holdout")
    check("LinearPartition target", pointer(data["linearpartition_run"], "name"), "7AM2_1")
    check("LinearPartition variant", pointer(data["linearpartition_run"], "execution/variant"), "LinearPartition-C")
    check("LinearPartition approximate sparse", pointer(data["linearpartition_run"], "execution/column"), "APPROXIMATE_SPARSE")
    check("LinearPartition CPU threads", pointer(data["linearpartition_run"], "execution/threads"), 1)
    check("LinearPartition full input", pointer(data["linearpartition_run"], "input/length"), 19000, "19,000")
    check("LinearPartition not truncated", pointer(data["linearpartition_run"], "input/truncated"), False)
    check("LinearPartition wall", pointer(data["linearpartition_run"], "resources/latency/wall_seconds"), 362.79, "362.79")
    check("LinearPartition RSS", pointer(data["linearpartition_run"], "resources/memory/peak_cpu_rss_bytes"), 8481693696, "8,481,693,696")
    check("LinearPartition FLOPs", pointer(data["linearpartition_run"], "resources/flops/status"), "NOT_MEASURED")
    check("LinearPartition no FLOPs estimate", pointer(data["linearpartition_run"], "resources/flops/estimated"), False)
    check("LinearPartition holdout only", pointer(data["linearpartition_run"], "holdout_descriptive_only"), True)
    check("LinearPartition not selection", pointer(data["linearpartition_run"], "not_for_model_selection"), True)
    check("LinearPartition metric scope", pointer(data["linearpartition_metrics"], "scope"), "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION")
    check("LinearPartition record coverage", pointer(data["linearpartition_metrics"], "coverage/record_coverage"), 1.0)
    check("LinearPartition nucleotide coverage", pointer(data["linearpartition_metrics"], "coverage/completed_nucleotide_coverage"), 1.0)
    check("LinearPartition output length", pointer(data["linearpartition_metrics"], "coverage/output_nucleotides"), 19000)
    check("LinearPartition coverage not truncated", pointer(data["linearpartition_metrics"], "coverage/truncated"), False)
    for output, tp, recall in [("bpp_candidate_pair_recall", 53, 0.726027397260274), ("mea_pair_recall", 8, 0.1095890410958904)]:
        check(f"LinearPartition {output} TP", pointer(data["linearpartition_metrics"], f"{output}/overall/true_positive"), tp)
        check(f"LinearPartition {output} reference", pointer(data["linearpartition_metrics"], f"{output}/overall/reference"), 73)
        check(f"LinearPartition {output} recall", pointer(data["linearpartition_metrics"], f"{output}/overall/recall"), recall)
        check(f"LinearPartition {output} 200-999 TP", pointer(data["linearpartition_metrics"], f"{output}/200-999/true_positive"), 0)
        check(f"LinearPartition {output} 200-999 reference", pointer(data["linearpartition_metrics"], f"{output}/200-999/reference"), 13)
        check(f"LinearPartition {output} >=1000 reference", pointer(data["linearpartition_metrics"], f"{output}/>=1000/reference"), 0)
        check(f"LinearPartition {output} >=1000 null", pointer(data["linearpartition_metrics"], f"{output}/>=1000/recall"), None)
    check("LinearPartition verification status", pointer(data["linearpartition_verification"], "status"), "PASS_INDEPENDENT_HOLDOUT_OUTPUT_HASH_RESOURCE_AND_METRIC_VERIFICATION")
    check("LinearPartition independent wall", pointer(data["linearpartition_verification"], "recomputed/wall_seconds"), 362.79)
    check("LinearPartition independent RSS", pointer(data["linearpartition_verification"], "recomputed/peak_cpu_rss_bytes"), 8481693696)
    check("LinearPartition independent BPP TP", pointer(data["linearpartition_verification"], "recomputed/bpp_candidate_pair_recall/overall/true_positive"), 53)
    check("LinearPartition independent MEA TP", pointer(data["linearpartition_verification"], "recomputed/mea_pair_recall/overall/true_positive"), 8)
    check("LinearPartition delivery status", pointer(data["linearpartition_delivery"], "status"), "PASS_FULL_19000_HOLDOUT_CPU_PILOT_AND_INDEPENDENT_VERIFICATION")
    check("LinearPartition delivery head", pointer(data["linearpartition_delivery"], "head"), "bbc4366bae1e4643bccb2e960470abcee77a83c2")
    check("LinearPartition delivery coverage", pointer(data["linearpartition_delivery"], "coverage"), 1.0)
    check("LinearPartition delivery holdout only", pointer(data["linearpartition_delivery"], "holdout_descriptive_only"), True)
    check("LinearPartition delivery not selection", pointer(data["linearpartition_delivery"], "not_for_model_selection"), True)

    check("label base commit", pointer(data["label_summary"], "base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("label schema", pointer(data["label_summary"], "schema_version"), "foldnar.label_quality.v1")
    check("label total", pointer(data["label_summary"], "total_records"), 148285, "148,285")
    check("label record counts", pointer(data["label_summary"], "record_counts"), {
        "tr1c_cluster_train": 33421,
        "tr1c_cluster_dev": 9143,
        "chanrg_train": 95181,
        "chanrg_validation": 10540,
    })
    check("label combined records", pointer(data["label_summary"], "combined/records"), 148285)
    check("TR1c train DATABASE", pointer(data["label_summary"], "split_summaries/tr1c_cluster_train/source_categories"), {"DATABASE": 33421})
    check("TR1c dev DATABASE", pointer(data["label_summary"], "split_summaries/tr1c_cluster_dev/source_categories"), {"DATABASE": 9143})
    check("CHANRG train COMPARATIVE_INFERRED", pointer(data["label_summary"], "split_summaries/chanrg_train/source_categories"), {"COMPARATIVE_INFERRED": 95181})
    check("CHANRG validation COMPARATIVE_INFERRED", pointer(data["label_summary"], "split_summaries/chanrg_validation/source_categories"), {"COMPARATIVE_INFERRED": 10540})
    check("label lineage unchecked", pointer(data["label_summary"], "combined/experimental_lineage_status"), {"UNCHECKED": 148285})
    check("label lineage verified", pointer(data["label_summary"], "per_record_experimental_lineage_verified_records"), 0)
    check("label noncanonical records", pointer(data["label_summary"], "combined/issue_record_counts/noncanonical_pairs"), 95052, "95,052")
    check("label short-span records", pointer(data["label_summary"], "combined/issue_record_counts/min_loop_span_lt4"), 10250, "10,250")
    check("label review records", pointer(data["label_summary"], "combined/review_records"), 95783, "95,783")
    check("label hard conflicts", pointer(data["label_summary"], "combined/hard_conflict_records"), 0, "硬冲突记录=0")
    check("label proposed exclusions", pointer(data["label_summary"], "combined/proposed_exclusion_records"), 0)
    check("label actual deletions", pointer(data["label_summary"], "combined/actual_deleted_records"), 0, "实际删除=0")
    check("label soft only", pointer(data["label_summary"], "combined/deletion_conservation/soft_review_only_records"), 95783)
    check("label conservation", pointer(data["label_summary"], "combined/deletion_conservation/conservation_holds"), True)
    check("label deletion unauthorized", pointer(data["label_summary"], "combined/deletion_conservation/deletion_authorized"), False)
    check("label no deletion performed", pointer(data["label_summary"], "actual_deletion_performed"), False)
    check("label IDs unchanged", pointer(data["label_summary"], "training_ids_modified"), False)
    check("label model-dev unchanged", pointer(data["label_summary"], "model_dev_v2_modified"), False)
    check("label no external test", pointer(data["label_summary"], "external_test_accessed"), False)
    check("label no GPU task", pointer(data["label_summary"], "gpu_task_launched"), False)
    check("label soft-review policy", pointer(data["label_conservation"], "policy/soft_review"), "review only; noncanonical, short-loop and crossing labels are not automatically invalid")
    check("label independent pass", pointer(data["label_verification"], "verification_passed"), True)
    check("label independent raw", pointer(data["label_verification"], "independent_raw_recomputation"), True)
    check("label builder not imported", pointer(data["label_verification"], "builder_imported"), False)
    check("label independent conflicts", pointer(data["label_verification"], "hard_conflict_records_verified"), 0)
    check("label independent deletions", pointer(data["label_verification"], "actual_deleted_records"), 0)
    check("label manifest base", pointer(data["label_manifest"], "base_commit"), "79a2ed45879f1f4e81d6719ed54600bf7f8da807")
    check("label manifest pass", pointer(data["label_manifest"], "verification_passed"), True)
    check("label manifest rounds", pointer(data["label_manifest"], "verification_rounds"), [1, 2, 3])

    check("exposure schema", pointer(data["exposure_summary"], "schema_version"), "rinalmo.pretrain_exposure.summary.v1")
    check("complete official corpus UNCHECKED", pointer(data["exposure_summary"], "complete_official_pretraining_coverage_state"), "UNCHECKED", "完整官方预训练语料仍 `UNCHECKED`")
    check("exposure dev rows", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/n_input_rows"), 9143, "9,143")
    check("exposure dev unique", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/n_unique_sequences"), 9143)
    check("exposure dev role", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/role"), "development")
    check("exposure dev selection", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/selection_use"), True)
    check("exposure dev exact states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/known_source_exact_state"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 1165, "UNCHECKED": 0,
    })
    check("exposure dev joint states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/known_source_exact_or_homology_state"), {
        "EXACT": 7978, "HOMOLOGY": 539, "NOT_DETECTED_UNKNOWN": 626, "UNCHECKED": 0,
    }, "EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626")
    check("exposure dev complete official states", pointer(data["exposure_summary"], "populations/tr1c_cluster80_dev/complete_official_pretraining_exposure_state"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 1165,
    })
    check("exposure bpRNA-new rows", pointer(data["exposure_summary"], "populations/bprna_new/n_input_rows"), 5388, "5,388")
    check("exposure bpRNA-new role", pointer(data["exposure_summary"], "populations/bprna_new/role"), "external_test_final_audit_only")
    check("exposure bpRNA-new no selection", pointer(data["exposure_summary"], "populations/bprna_new/selection_use"), False)
    check("exposure bpRNA-new exact states", pointer(data["exposure_summary"], "populations/bprna_new/known_source_exact_state"), {
        "EXACT": 5381, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 7, "UNCHECKED": 0,
    })
    check("exposure bpRNA-new joint states", pointer(data["exposure_summary"], "populations/bprna_new/known_source_exact_or_homology_state"), {
        "EXACT": 5381, "HOMOLOGY": 7, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 0,
    }, "bpRNA-new EXACT=5,381/5,388")
    check("exposure bpRNA-new complete official states", pointer(data["exposure_summary"], "populations/bprna_new/complete_official_pretraining_exposure_state"), {
        "EXACT": 5381, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 0, "UNCHECKED": 7,
    })
    check("official training set UNCHECKED", pointer(data["exposure_assets"], "official_pretraining_declaration/complete_official_training_set_state"), "UNCHECKED")
    check("official nt unpinned", pointer(data["exposure_assets"], "official_pretraining_declaration/nt/status"), "NOT_DOWNLOADED_HISTORICAL_SNAPSHOT_UNPINNED")
    check("official declared sources", pointer(data["exposure_assets"], "official_pretraining_declaration/declared_sources"), ["RNAcentral", "Rfam", "nt", "Ensembl 109", "Ensembl Bacteria 56"])
    check("exposure verification status", pointer(data["exposure_verification"], "status"), "PASS")
    check("exposure registry rows", pointer(data["exposure_verification"], "n_registry_rows"), 157227, "157,227")
    check("exposure verification dev", pointer(data["exposure_verification"], "population_exact_counts/tr1c_cluster80_dev"), {
        "EXACT": 7978, "HOMOLOGY": 0, "NOT_DETECTED_UNKNOWN": 1165, "UNCHECKED": 0,
    })
    check("exposure verification bpRNA-new", pointer(data["exposure_verification"], "bprna_new"), {
        "rows": 5388, "exact": 5381, "exact_not_detected_unknown": 7, "homology_only": 7, "homology_not_detected": 0,
    })
    check("exposure verification official coverage", pointer(data["exposure_verification"], "official_complete_coverage"), "UNCHECKED")
    check("exposure no training", pointer(data["exposure_verification"], "no_model_inference_or_training"), True)
    check("exposure external final audit", pointer(data["exposure_verification"], "external_test_use"), "final exposure/quality audit only")
    sota_limits = pointer(data["exposure_summary"], "sota_limitations")
    for required in ["clean-OOD", "NOT_DETECTED_UNKNOWN is not clean", "no complete exposure rate", "not proof of superiority on unseen RNA families"]:
        if not any(required in item for item in sota_limits):
            errors.append(f"exposure SOTA limitation missing: {required}")

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
        errors += verify_numbers() + verify_source_commits()
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    scope = "links and narrative guards" if args.links_only else "links, narrative guards, JSON numeric provenance, and source commits"
    print(f"PASS: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
