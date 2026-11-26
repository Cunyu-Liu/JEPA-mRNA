from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/gec_nll_v3_requalified_artifact_v1.py"
SPEC = importlib.util.spec_from_file_location("gec_nll_v3_requalified_artifact_v1", MODULE_PATH)
F = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = F
SPEC.loader.exec_module(F)


def reports():
    primary = {"arms": {}}
    second = {"primary_agreement": {}, "per_arm": {}, "qualification_pass": True,
              "primary_and_second_agree": True}
    for arm in F.ARMS:
        primary["arms"][arm] = {
            "qualification_pass": True,
            "validation": {"stop_step": 13000, "best_step": 10500,
                           "best_validation_loss": 0.25, "bad_evaluations": 5},
            "historical_code": {"pass": True}, "data_identity": {"pass": True},
        }
        second["primary_agreement"][arm] = True
        second["per_arm"][arm] = {
            "qualification_pass": True,
            "direct_tensors_and_run_meta": {"all_model_tensors_equal": True,
                                             "sole_defect_is_restored_bad_counter": True},
            "predictions": {"pass": True},
        }
    return primary, second


def test_final_summary_preserves_qualification_and_science_boundary():
    primary, second = reports()
    summary = F.summarize(primary, second)
    assert summary["status"] == "PASS_NLL_V3_REQUALIFIED"
    assert all(row["qualification_pass"] for row in summary["per_arm"].values())
    assert summary["reference_use"]["original_scientific_fail_unchanged"]
    assert not summary["reference_use"]["margin_superior_to_nll_claim_allowed"]
    assert not summary["reference_use"]["scientific_gate_recomputed_or_changed"]


def test_final_summary_rejects_second_implementation_disagreement():
    primary, second = reports()
    arm = "matched_control"
    second["primary_agreement"][arm] = False
    summary = F.summarize(primary, second)
    assert not summary["qualification_pass"]
    assert summary["per_arm"][arm]["qualification"] == "NOT_QUALIFIED"
