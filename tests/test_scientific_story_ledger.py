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
