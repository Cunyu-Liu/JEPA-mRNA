import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from verify_prior_alignment_optimizer_matrix import aggregate


def build_root(tmp_path, deltas):
    metrics = []
    for scope in ("micro", "macro"):
        for level in ("pair", "stem"):
            for metric in ("precision", "recall", "f1"):
                metrics.append(f"{scope}_{level}_{metric}")
    for seed, delta in enumerate(deltas):
        stage = tmp_path / f"seed{seed}_backbone_lr_1e-6"
        stage.mkdir()
        arms = {}
        for arm in ("C0", "C1", "C2", "C3"):
            value = 0.8 + (delta if arm == "C2" else 0.0)
            arms[arm] = {metric: value for metric in metrics}
            arms[arm]["n_sequences"] = 196
        (stage / "optimizer_scientific_verification.json").write_text(json.dumps({
            "stability_pass": True,
            "arms": arms,
        }))
    return tmp_path


def test_optimizer_matrix_requires_all_original_margins(tmp_path):
    result = aggregate(build_root(tmp_path, [0.02, 0.02, 0.02]))
    assert result["candidate_selected"]
    assert result["external_evaluation_allowed"]
    assert result["C3_repeatedly_weaker_descriptive"]


def test_optimizer_matrix_rejects_inconsistent_seed_direction(tmp_path):
    result = aggregate(build_root(tmp_path, [0.02, 0.02, -0.1]))
    assert not result["candidate_selected"]
    assert not result["checks"]["three_seed_macro_direction"]
    assert not result["checks"]["C2_vs_C0_micro"]
