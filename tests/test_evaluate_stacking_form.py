import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_stacking_form", ROOT / "tools/evaluate_stacking_form.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_uses_control_prediction_and_true_stacking_neighbor():
    truth = {(0, 15), (1, 14), (6, 9)}
    control = {(0, 15), (6, 10)}
    assert MODULE.target_pairs(truth, control) == {(1, 14)}
    assert MODULE.target_pairs(truth, set()) == set()


def test_target_excludes_isolated_pairs_even_with_unrelated_true_positive():
    truth = {(0, 15), (6, 9)}
    control = {(0, 15)}
    assert MODULE.target_pairs(truth, control) == set()
