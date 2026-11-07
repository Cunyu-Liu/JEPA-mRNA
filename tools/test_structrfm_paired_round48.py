import pytest

from tools.compare_structrfm_paired_round48 import (holm_adjust, micro_f1,
                                                    paired_bootstrap, paired_rows)


def test_paired_bootstrap_zero_and_deterministic():
    left = [{"tp": 2, "fp": 1, "fn": 1}, {"tp": 0, "fp": 1, "fn": 2}]
    first = paired_bootstrap(left, left, 17, 1000)
    second = paired_bootstrap(left, left, 17, 1000)
    assert first == second
    assert first["delta_micro_f1"] == 0
    assert first["ci95"] == [0, 0]
    assert first["p_two_sided"] == 1
    assert micro_f1((2, 1, 1)) == pytest.approx(2 / 3)


def test_paired_bootstrap_direction_and_holm():
    left = [{"tp": 0, "fp": 0, "fn": 2}, {"tp": 0, "fp": 0, "fn": 1}]
    right = [{"tp": 2, "fp": 0, "fn": 0}, {"tp": 1, "fp": 0, "fn": 0}]
    result = paired_bootstrap(left, right, 23, 1000)
    assert result["delta_micro_f1"] == 1
    assert result["ci95"] == [1, 1]
    assert result["p_two_sided"] == pytest.approx(2 / 1001)
    rows = [{"p_two_sided": 0.01}, {"p_two_sided": 0.02}, {"p_two_sided": 0.03}]
    holm_adjust(rows)
    assert [row["p_holm_main8"] for row in rows] == pytest.approx([0.03, 0.04, 0.04])


def test_pairing_rejects_different_support_or_order():
    base = {"data_sha256": "abc", "raw_n": 2, "n": 2, "n_long_unsupported": 0,
            "per_sequence": [{"name": "a", "tp": 1, "fp": 0, "fn": 0},
                             {"name": "b", "tp": 1, "fp": 0, "fn": 0}]}
    assert paired_rows(base, base) == (base["per_sequence"], base["per_sequence"])
    changed = {**base, "n_long_unsupported": 1}
    with pytest.raises(ValueError, match="supported and unsupported"):
        paired_rows(base, changed)
    reordered = {**base, "per_sequence": list(reversed(base["per_sequence"]))}
    with pytest.raises(ValueError, match="sequence order"):
        paired_rows(base, reordered)
