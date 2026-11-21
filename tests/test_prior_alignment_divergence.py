import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from train_plan_a_clean_head_p1 import divergence_reason


def test_divergence_is_opt_in_and_preserves_default_behavior():
    assert divergence_reason(1.0, 1.0) == ""
    assert divergence_reason(11.0, 1.0, 10.0, 100.0).startswith("LOSS_THRESHOLD_EXCEEDED")
    assert divergence_reason(1.0, 101.0, 10.0, 100.0).startswith("GRADIENT_NORM_THRESHOLD_EXCEEDED")


def test_divergence_rejects_non_finite_values():
    assert divergence_reason(math.nan, 1.0) == "NON_FINITE_LOSS_OR_GRADIENT_NORM"
    assert divergence_reason(1.0, math.inf) == "NON_FINITE_LOSS_OR_GRADIENT_NORM"
