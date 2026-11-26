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


def test_raw_baseline_absolute_gap_arithmetic():
    structrfm = 0.5437893868896873
    assert 0.7370074184282851 - structrfm == pytest.approx(0.19321803153859785)
    assert 0.6967639314022964 - structrfm == pytest.approx(0.15297454451260917)
    text = ledger.DOC.read_text()
    assert "这不是相对百分比，也不是我们的候选提升" in text
    assert "BPfold official-reference 标签结果和 CONTRAfold TR1c-clean 子集均不得混入" in text


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
    assert "8b1861f9f8c116250aba194799bf5ee7d9a4f39e" in text
    assert "不得干扰 GPU 或改原 16 门" in text
    assert "margin 仍无结论" in text
    assert "本节只约束后续内容，不修改 PPT" in text


def test_linearpartition_recall_and_gec_boundary():
    assert 53 / 73 == pytest.approx(0.726027397260274)
    assert 8 / 73 == pytest.approx(0.1095890410958904)
    text = ledger.DOC.read_text()
    assert "长序列只有 baseline 回答，GEC unsupported" in text
    assert "HOLDOUT_DESCRIPTIVE_ONLY_NOT_FOR_MODEL_SELECTION" in text
    assert "candidate recall 不是最终结构准确率" in text
    assert "\n+- **" not in text
