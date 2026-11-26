import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools.verify_structured_margin_smoke_v1 import (
    ARMS, matched_reference_train_ids, require_two_finite_steps,
)


def row(step, loss=0.25):
    return {
        "step": step, "optimizer_step": True, "divergence_reason": "",
        "loss": loss, "gradient_norm": 1.0, "residual_gradient_norm": 0.1,
        "context_log_z": [2.0], "residual_changed_from_initial": True,
    }


def test_smoke_requires_exactly_two_finite_updating_steps():
    require_two_finite_steps([row(1), row(2)])
    with pytest.raises(AssertionError):
        require_two_finite_steps([row(1)])
    with pytest.raises(AssertionError):
        require_two_finite_steps([row(1), row(2, float("nan"))])
    bad = row(2)
    bad["residual_gradient_norm"] = 0.0
    with pytest.raises(AssertionError):
        require_two_finite_steps([row(1), bad])


def test_launcher_has_no_full_training_or_decoder_search():
    text = (REPO / "tools/run_structured_margin_smoke_v1.sh").read_text()
    assert "--steps 2" in text
    assert "structured_margin_smoke_v1" in text
    assert "--steps 0" not in text
    assert "temperature" not in text and "gamma" not in text and "threshold" not in text
    assert "seed 1" not in text and "seed 2" not in text


def test_all_arm_aggregation_preserves_reference_and_rejects_mismatch():
    results = {
        arm: {
            "batch_order": [{"step": 1}], "train_record_ids": ["train"],
            "validation_record_ids": ["validation"],
        }
        for arm in ARMS
    }
    assert matched_reference_train_ids(results) == ["train"]
    assert results[ARMS[0]]["batch_order"] == [{"step": 1}]
    results[ARMS[2]]["batch_order"] = [{"step": 2}]
    with pytest.raises(AssertionError, match="batch or train identity"):
        matched_reference_train_ids(results)
