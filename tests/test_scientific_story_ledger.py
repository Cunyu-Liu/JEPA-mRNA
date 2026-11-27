import pytest

from tools import verify_scientific_story_ledger as ledger


def test_relative_markdown_links():
    assert ledger.verify_links() == []


def test_scientific_narrative_guards():
    assert ledger.verify_narrative_guards() == []


def test_json_pointer_and_residual_arithmetic():
    document = {"groups": {"true_all": {"nonpositive_score_margin": 101164, "positive_weight_mea_competition": 1739}}}
    group = ledger.pointer(document, "groups/true_all")
    ratio = group["nonpositive_score_margin"] / (group["nonpositive_score_margin"] + group["positive_weight_mea_competition"])
    assert ratio == pytest.approx(0.9831005898759025)
    assert 247013 - 144110 == 101164 + 1739


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_frozen_json_numeric_provenance():
    assert ledger.verify_numbers() == []


def test_crossover_claim_separation_guard():
    text = ledger.DOC.read_text()
    assert "crossover 直接支持①，但不支持②" in text
    assert "上下文有因果作用”与“真实 ensemble 一致有益”是不同命题" in text
    assert "没有最终 hash-verified 四臂评估可填入" in text


def test_structrfm_official_raw_scope_and_arithmetic():
    text = ledger.DOC.read_text()
    assert "不是我们的候选正向性能，也不是 clean SOTA" in text
    assert "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED" in text
    micro = 2 * 73864 / (2 * 73864 + 62616 + 61320)
    assert micro == pytest.approx(0.5437893868896873)


def test_official_reference_absolute_gap_arithmetic():
    xens2 = 0.48901660903024696
    assert 0.7745179151021447 - xens2 == pytest.approx(0.2855013060718978)
    assert 0.629675588219163 - xens2 == pytest.approx(0.14065897918891607)
    assert 0.6339521964768657 - xens2 == pytest.approx(0.14493558744661877)
    text = ledger.DOC.read_text()
    assert "这不是相对百分比，也不是我们的候选提升" in text
    assert "official release 与项目 replay 是不同 prediction identity" in text
    assert "0.19321803153859785" not in text
    assert "0.15297454451260917" not in text


def test_bpfold_protocol_and_official_panel_provenance():
    text = ledger.DOC.read_text()
    assert "e7ff4e76fe078178ab41d2052349627317b5a3e3" in text
    assert "DERIVED_FROM_BPFOLD_PREDICTIONS" in text
    assert "4924e3512e89fd965233a1a14b327c3d87e6c583764b7682c2d293b1454159e4" in text
    assert "106 条" in text and "5,282 条" in text
    assert "0.9617 仅循环审计不得排名" in text
    assert "bprna_new_official_reference_5388_prediction_registry.json" in text


def test_bpfold_sota_and_circular_exclusion_contract():
    text = ledger.DOC.read_text()
    assert "19/19 派生标签资产" in text
    assert "28/28 official-reference scored rows 均不含 0.9617" in text
    assert "旧差距禁止回写" in text
    for value in ledger.OBSOLETE_RAW_GAP_VALUES:
        assert value not in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_bpfold_official_registry_contract():
    assert ledger.verify_bpfold_official_contract() == []


def test_archiveii_official_values_ci_and_full_coverage_boundary():
    text = ledger.DOC.read_text()
    assert "0.8156522070450173" in text
    assert "0.8999331354520079" in text
    assert "0.8204522394549935" in text
    assert "+0.0511175356573565, +0.0619467954524887" in text
    assert "−0.0012121956293115, +0.0078699689970171" in text
    assert "RiNALMo-ft 的对应 CI **跨 0**" in text
    assert "CI 只属于 macro-F1 差值" in text
    assert "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR" in text
    assert "19 个有部分预测的身份均为 2,544/3,966" in text
    assert "PARTIAL_COVERAGE_NOT_RANKED" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_archiveii_official_artifact_contract():
    assert ledger.verify_archiveii_official_contract() == []


def test_pdb_rebuild_overlap_and_rfam_temporal_boundaries():
    text = ledger.DOC.read_text()
    assert "105bb55d2e959e587c85d3de7607fdc3c14c5c73" in text
    assert "62 条重建、91 条 UNCHECKED" in text
    assert "58 true / 4 false" in text
    assert "4 条 DSSR scheme 修复" in text
    assert "旧 58/95 与 56/2 不再是当前值" in text
    assert "PDB62 仍不得排名" in text
    assert "ref_pdb_ts_hard` 28/28" in text
    assert "TS-hard 与 TS1/TS3 不序列不相交" in text
    assert "Rfam 14.9 exact=**43**" in text
    assert "TEMPORAL_FIRST_RELEASE_UNCHECKED" in text
    assert "COMPARATIVE_INFERRED" in text
    assert "AUDIT_ONLY_NO_MODEL_SELECTION" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_pdb_rfam_artifact_contract():
    assert ledger.verify_pdb_rfam_contract() == []


def test_incident_recovery_and_step10000_validation_status():
    text = ledger.DOC.read_text()
    assert "442636dd660f93f1d39156ca9bbdbd107ecdbf42" in text
    assert "OSError: [Errno 122] Disk quota exceeded" in text
    assert "step=9500" in text
    assert "step=10000 validation 已完成" in text
    assert "0.4183340173506429" in text
    assert "stop=false" in text
    assert "completed_arms=[]" in text
    assert "evaluation=null" in text
    assert "四臂未收敛，当前无 margin 科学结论" in text
    assert "/home/foldnar_incident_checkpoints" in text
    assert "不得宣称最终镜像完成" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_incident_recovery_artifact_contract():
    assert ledger.verify_incident_recovery_contract() == []


def test_second_scheduler_quota_incident_and_final_head():
    text = ledger.DOC.read_text()
    assert "aa9f1c4a92e92d70841e33cec455734f160075c4" in text
    assert "第二次独立的 scheduler 状态写盘事故" in text
    assert "4077848` 已退出，matched `4077864` 仍存活" in text
    assert "recovery_takeover_status.json" in text
    assert "旧 teams lock/PID/state/events 仅作事故证据" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_takeover_artifact_contract():
    assert ledger.verify_takeover_contract() == []


def test_takeover_lock_pid_adoption_and_local_storage():
    text = ledger.DOC.read_text()
    assert "scheduler PID **4111890**" in text
    assert "PID **4077864**" in text
    assert "process_start_ticks=1329819713" in text
    assert "recovery_scheduler.pid` 内容为 `4111890\\n`" in text
    assert "fd 3 持有 `/home/foldnar_incident_scheduler/recovery_scheduler.lock`" in text
    assert "adopt 原进程而非重复训练" in text
    assert "recovery_code_commit=a61a6ea8e52f72fa129387b74cb28b3af8bc64ff" in text
    assert "teams 写入仅限合法终态后的最终小证据镜像" in text
    assert "当前评估未开始、目录尚未创建" in text


def test_takeover_best15000_remains_nonterminal_and_no_conclusion():
    text = ledger.DOC.read_text()
    assert "train step=15500" in text
    assert "15500 validation pending" in text
    assert "step=15000、validation_loss=0.3896151208942566" in text
    assert "best_step=15000" in text
    assert "bad_evaluations=0" in text
    assert "save_best=true" in text
    assert "stop=false" in text
    assert "best.pt` 存在" in text
    assert "completed_arms=[]" in text
    assert "四臂未收敛，当前无 margin 科学结论" in text


def test_ts0_projected_values_and_official_coverage_boundary():
    text = ledger.DOC.read_text()
    assert "8a2008a4cda3c43b3000f25151b80408b0bdb110" in text
    assert "0.8032405714870655" in text
    assert "0.7568974968024850" in text
    assert "−0.0674523877056411, −0.0435507990639508" in text
    assert "a8b1089a67a84665c49f0c228064693a53548590" in text
    assert "0.6812363564199246" in text
    assert "现有 **7 个全覆盖身份**" in text
    assert "1302/1305" in text
    assert "旧零全覆盖" in text
    assert "homology-only=575" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_ts0_artifact_contract():
    assert ledger.verify_ts0_contract() == []


def test_testsetb_exposure_and_official_coverage_boundary():
    text = ledger.DOC.read_text()
    assert "0.8403202122116432" in text
    assert "0.8670430033503144" in text
    assert "+0.0169091444489763, +0.0452470510286115" in text
    assert "TR0 精确序列暴露 247/428" in text
    assert "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT" in text
    assert "不可独立排名" in text
    assert "7342f26303d3acc8d61f395c040534242d0f7895" in text
    assert "0.6468595682161368" in text
    assert "现有 **7 个全覆盖身份**" in text
    assert "429/430" in text
    assert "原 20 个 projected 身份指标经独立验证保持不变" in text
    assert "homology-only=140" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_testsetb_artifact_contract():
    assert ledger.verify_testsetb_contract() == []


def test_full_recovery_preserves_projected_history_and_exposure():
    text = ledger.DOC.read_text()
    assert "Full 覆盖恢复联合不变量" in text
    assert "projected 原值/CI/暴露不变" in text
    assert "旧零覆盖仅历史" in text
    assert "当前若保留 CI 字段也必须一致" in text
    assert "RNAstructure 在 full1305 仍为 1302、full430 仍为 429" in text
    assert text.count("official full coverage=0") == 2


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_projected_recovery_history_artifact_contract():
    assert ledger.verify_projected_recovery_history() == []


def test_rfam_family_release_counts_and_sequence_distinction():
    text = ledger.DOC.read_text()
    assert "105bb55d2e959e587c85d3de7607fdc3c14c5c73" in text
    assert "11,186 条 `UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED`" in text
    assert "family first-release **14.10=166、15.0=99**" in text
    assert "14.9_OR_EARLIER=10,921、UNCHECKED=40" in text
    assert "family release≠sequence release" in text
    assert "PROPOSED_ONLY_NOT_REGISTERED" in text


@pytest.mark.skipif(ledger.missing_sources(), reason="host-specific frozen artifacts are unavailable")
def test_rfam_release_artifact_contract():
    assert ledger.verify_rfam_release_contract() == []


def test_label_quality_soft_review_and_zero_deletion():
    text = ledger.DOC.read_text()
    assert 33421 + 9143 + 95181 + 10540 == 148285
    assert "TR1c `DATABASE` 与 CHANRG `COMPARATIVE_INFERRED` 均不能称为实验真值" in text
    assert "95,052 条 noncanonical" in text
    assert "10,250 条 short-span" in text
    assert "硬冲突记录=0" in text
    assert "实际删除=0" in text
    assert "不自动删除" in text


@pytest.mark.skipif(any(not repo.exists() for repo, _ in ledger.SOURCE_COMMITS.values()), reason="host-specific source repositories are unavailable")
def test_source_commit_identity():
    assert ledger.verify_source_commits() == []


def test_pretrain_exposure_conservation_and_sota_limits():
    text = ledger.DOC.read_text()
    assert 7978 + 539 + 626 == 9143
    assert 5381 + 7 == 5388
    assert "EXACT=7,978 / HOMOLOGY=539 / NOT_DETECTED_UNKNOWN=626" in text
    assert "bpRNA-new EXACT=5,381/5,388" in text
    assert "完整官方预训练语料仍 `UNCHECKED`" in text
    assert "dev 不能外推 clean-OOD" in text
    assert "剩余 7 条均为 HOMOLOGY" in text


def test_next_page_exposure_strata_and_margin_pending():
    text = ledger.DOC.read_text()
    assert "下一页必须暴露分层" in text
    assert "margin 双训练未完成" in text
    assert "/home/rna_jev_margin_evaluator" in text
    assert "536b79e450c1ad18b67e06cb39ecb3730546221b" in text
    assert "不得干扰 GPU 或改原 16 门" in text
    assert "margin 仍无结论" in text
    assert "本节只约束后续内容，不修改 PPT" in text


def test_obsolete_margin_evaluator_is_not_current():
    text = ledger.DOC.read_text()
    assert ledger.OBSOLETE_MARGIN_EVALUATOR_COMMIT not in text
    assert ledger.SOURCE_COMMITS["independent margin evaluator"][1] == "536b79e450c1ad18b67e06cb39ecb3730546221b"
    assert "current independent margin evaluator" in text


def test_model_dev_v3_registration_and_conservation():
    text = ledger.DOC.read_text()
    assert "302b70a228d77b0f1847489c93ead402e33510b5" in text
    assert "PASS_THREE_ROUNDS" in text
    assert 206588 + 19528 + 20897 == 247013
    assert "206,588 / 19,528 / 20,897" in text
    assert "NOT_COMPUTED_BY_REGISTRY_BUILD" in text


def test_v3_exposure_roles_and_original_16_gates():
    text = ledger.DOC.read_text()
    assert "EXACT sensitivity-only，HOMOLOGY/UNKNOWN diagnostic-only，原16门不变" in text
    assert "PREREGISTERED_SENSITIVITY_ONLY" in text
    assert "gate_use=FORBIDDEN" in text
    assert "original_16_gates_modified=false" in text
    assert "n_gates=16 / n_failed_gates=12" in text


@pytest.mark.skipif(any(not repo.exists() for repo, _ in ledger.SOURCE_COMMITS.values()), reason="host-specific source repositories are unavailable")
def test_evaluator_model_dev_v3_contract():
    assert ledger.verify_evaluator_v3_contract() == []


def test_linearpartition_recall_and_gec_boundary():
    assert 53 / 73 == pytest.approx(0.726027397260274)
    assert 8 / 73 == pytest.approx(0.1095890410958904)
    text = ledger.DOC.read_text()
    assert "长序列只有 baseline 回答，GEC unsupported" in text
    assert "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION" in text
    assert "candidate recall 不是最终结构准确率" in text
    assert "\n+- **" not in text
