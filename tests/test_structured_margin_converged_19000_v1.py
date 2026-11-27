import pytest

from tools import audit_structured_margin_converged_19000_v1 as audit


def validation_chain():
    rows = []
    for step in range(500, 19001, 500):
        rows.append({"step": step, "eligible": True, "improved": False,
                     "save_best": False, "stop": False, "best_step": 16500,
                     "bad_evaluations": 0})
    best = rows[32]
    assert best["step"] == 16500
    best.update({"improved": True, "save_best": True})
    for bad, step in enumerate(range(17000, 19001, 500), 1):
        rows[step // 500 - 1].update({"bad_evaluations": bad, "stop": bad == 5})
    return rows


def test_stop_sequence_reconstructs_exact_five_bad_evaluations():
    audit.validate_stop_sequence(validation_chain(), 16500, 19000)


def test_stop_sequence_rejects_missing_or_early_stop():
    rows = validation_chain()
    rows[33]["stop"] = True
    with pytest.raises(ValueError, match="before 19000|patience record"):
        audit.validate_stop_sequence(rows, 16500, 19000)
    rows = validation_chain()
    del rows[20]
    with pytest.raises(ValueError, match="missing or duplicate"):
        audit.validate_stop_sequence(rows, 16500, 19000)


def test_shared_identity_ignores_only_arm():
    left = {"arm": "matched_control", "config": "same", "ids": [1, 2]}
    right = {"arm": "ensemble_context", "config": "same", "ids": [1, 2]}
    assert audit.shared_identity(left, right) is True
    right["config"] = "different"
    assert audit.shared_identity(left, right) is False
