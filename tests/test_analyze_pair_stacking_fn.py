import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stacking_fn", ROOT / "tools/analyze_pair_stacking_fn.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pair_environment_distinguishes_interior_boundary_and_isolated():
    truth = {(0, 15), (1, 14), (2, 13), (6, 9)}
    assert MODULE.pair_environment((1, 14), truth) == "helix_interior"
    assert MODULE.pair_environment((0, 15), truth) == "helix_boundary"
    assert MODULE.pair_environment((2, 13), truth) == "helix_boundary"
    assert MODULE.pair_environment((6, 9), truth) == "isolated_pair"


def test_pair_environment_requires_true_neighbors_not_predicted_neighbors():
    truth = {(0, 15)}
    assert MODULE.pair_environment((1, 13), truth) == "isolated_pair"
