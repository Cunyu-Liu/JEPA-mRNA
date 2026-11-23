import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capacity_freeze", ROOT / "tools/audit_repr_head_capacity_freeze.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _contract():
    return json.loads((ROOT / "configs/representation_head_mechanism_4arm_v1.json").read_text())


def _recheck(candidate):
    control = {
        "micro_auprc": 0.4, "micro_recall_at_precision_0.8": 0.2,
        "micro_pair_f1": 0.8, "macro_pair_f1": 0.75,
        "micro_pair_precision": 0.85, "macro_pair_precision": 0.8,
        "micro_pair_recall": 0.76, "macro_pair_recall": 0.7,
        "micro_stem_f1": 0.6, "macro_stem_f1": 0.55,
    }
    return {
        "scope": "INDEPENDENT_REPRESENTATION_HEAD_RECHECK_FROM_RAW_SCORES",
        "overall": "PASS", "n_sequences": 196,
        "recomputed_arms": {
            "control_matched": control,
            "head_deep": dict(control, **candidate),
            "repr_head": dict(control, **candidate),
        },
    }


def test_raw_recheck_deltas_and_gates_do_not_use_official_comparisons():
    result = MODULE.comparisons_from_raw_recheck(_recheck({
        "micro_auprc": 0.5, "micro_recall_at_precision_0.8": 0.3,
        "micro_pair_f1": 0.79, "macro_pair_f1": 0.74,
    }), _contract())
    assert result["head_deep"]["delta_vs_control_matched"]["micro_auprc"] == pytest.approx(0.1)
    assert result["head_deep"]["direction_passed"] is False
    assert "micro_pair_f1_noninferior" in result["head_deep"]["failed_gates"]
    assert "macro_pair_f1_noninferior" in result["head_deep"]["failed_gates"]


def test_raw_ranking_gain_cannot_offset_pair_and_stem_failure():
    result = MODULE.comparisons_from_raw_recheck(_recheck({
        "micro_auprc": 0.7, "micro_recall_at_precision_0.8": 0.6,
        "micro_pair_f1": 0.75, "macro_pair_f1": 0.65,
        "micro_stem_f1": 0.5, "macro_stem_f1": 0.45,
    }), _contract())
    assert result["repr_head"]["checks"]["micro_auprc_strictly_positive"] is True
    assert result["repr_head"]["checks"]["micro_recall_at_precision_0.8_strictly_positive"] is True
    assert result["repr_head"]["direction_passed"] is False


def test_raw_recheck_must_be_complete():
    recheck = _recheck({})
    recheck["overall"] = "FAIL"
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.comparisons_from_raw_recheck(recheck, _contract())
