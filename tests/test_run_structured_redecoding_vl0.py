import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "srd_runner", ROOT / "tools/run_structured_redecoding_vl0.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(micro_delta=0.01, macro_delta=0.01):
    base = {"micro_pair_f1": 0.8, "macro_pair_f1": 0.75,
            "micro_pair_precision": 0.85, "macro_pair_precision": 0.8,
            "micro_stem_f1": 0.6, "macro_stem_f1": 0.55}
    candidate = dict(base, micro_pair_f1=base["micro_pair_f1"] + micro_delta,
                     macro_pair_f1=base["macro_pair_f1"] + macro_delta,
                     nested_canonical_satisfaction=1.0, failed_sequences=0, fallback_sequences=0)
    return {"current_mea": base, "nested_srd_stacking": candidate}


def test_gate_requires_both_pair_f1_improvements():
    result = MODULE.promotion_decision(summary(macro_delta=-0.001))
    assert result["single_source_gate_passed"] is False
    assert result["external_evaluation_allowed"] is False
    assert "macro_pair_f1_strictly_positive" in result["failed_gates"]


def test_gate_rejects_fallback_even_with_metric_gain():
    value = summary()
    value["nested_srd_stacking"]["fallback_sequences"] = 1
    value["nested_srd_stacking"]["failed_sequences"] = 1
    result = MODULE.promotion_decision(value)
    assert result["single_source_gate_passed"] is False
    assert "no_failed_sequences" in result["failed_gates"]
    assert "no_fallback" in result["failed_gates"]


def test_gate_pass_does_not_open_external_or_sota():
    result = MODULE.promotion_decision(summary())
    assert result["single_source_gate_passed"] is True
    assert result["decision"] == "SINGLE_SOURCE_SRD_PASS_MULTIMODEL_STILL_REQUIRED"
    assert result["external_evaluation_allowed"] is False
    assert result["strict_ood_allowed"] is False
    assert result["sota_claim"] is False
