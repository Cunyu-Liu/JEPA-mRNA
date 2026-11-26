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
