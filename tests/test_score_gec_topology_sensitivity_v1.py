from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import gec_seed0_scientific_evaluator as G
import score_gec_topology_sensitivity_v1 as S

TEST_MODULE_PATH = Path(__file__).resolve().parent / "test_gec_seed0_scientific_evaluator.py"
SPEC = importlib.util.spec_from_file_location("gec_scientific_fixture_source", TEST_MODULE_PATH)
FIXTURES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = FIXTURES
SPEC.loader.exec_module(FIXTURES)


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sensitivity_fixture(fixture, root):
    raw = sorted(fixture["ids"][:30])
    excluded = []
    clean = raw
    paths = {}
    for name, filename, values in (
        ("raw_dev_ids", "raw.txt", raw),
        ("own_clean_dev_ids", "own.txt", clean),
        ("matched_clean_dev_ids", "matched.txt", clean),
        ("excluded_dev_ids", "excluded.txt", excluded),
    ):
        path = root / filename
        path.write_text("".join(value + "\n" for value in values))
        paths[name] = identity(path)
    scope = {
        "n_sequences": len(clean), "n_pairs": 17 * len(clean),
        "leakage_state": {"exact_loop_helix_topology_signature": "NOT_DETECTED_UNKNOWN"},
    }
    manifest = root / "sensitivity.json"
    manifest.write_text(json.dumps({
        "schema_version": "foldnar.model_dev_topology_sensitivity.v1",
        "panel_id": "tr1c_cluster80_dev_v1", "primary_gates_changed": False,
        "input_identities": {"registry": identity(fixture["registry"])},
        "artifacts": paths,
        "scopes": {"raw": {"n_sequences": 30, "n_pairs": 510},
                   "own_clean": scope, "matched_clean": scope},
    }))
    return manifest


def test_strict_full_and_sensitivity_views_use_same_bound_predictions(tmp_path):
    fixture = FIXTURES._scientific_fixture(tmp_path)
    manifest = sensitivity_fixture(fixture, tmp_path)
    tracked = [fixture["registry"], fixture["config"], manifest] + list(fixture["prediction_paths"].values())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    result = S.evaluate_dual_view(
        fixture["registry"], manifest, fixture["prediction_paths"], fixture["metadata_paths"],
        fixture["config"], resamples=8,
    )
    strict = G.evaluate(
        fixture["registry"], "tr1c_cluster80_dev_v1", fixture["prediction_paths"],
        fixture["metadata_paths"], fixture["config"], resamples=8,
    )
    assert result["primary_full_cluster_dev"] == strict
    assert result["status"] == strict["status"]
    assert result["primary_full_cluster_dev"]["panel"]["n_sequences"] == 30
    for name in ("own_clean", "matched_clean"):
        view = result["sensitivity_views"][name]
        assert view["panel"]["n_sequences"] == 30
        assert view["panel"]["n_pairs"] == 510
        assert view["panel"]["exact_topology_overlap_sequences_removed"] == 0
        assert view["status"] == "SENSITIVITY_ONLY_NEVER_REPLACE_PRIMARY_GATES"
        assert "diagnostic_gates" in view and "gates" not in view
        assert view["panel"]["approximate_structural_similarity"] == "UNCHECKED"
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}


def test_complete_full_predictions_are_mandatory(tmp_path):
    fixture = FIXTURES._scientific_fixture(tmp_path)
    manifest = sensitivity_fixture(fixture, tmp_path)
    path = fixture["prediction_paths"]["matched_control"]
    original = path.read_text().splitlines()
    path.write_text("\n".join(original[:-1]) + "\n")
    with pytest.raises(ValueError, match="prediction count does not match panel"):
        S.evaluate_dual_view(
            fixture["registry"], manifest, fixture["prediction_paths"], fixture["metadata_paths"],
            fixture["config"], resamples=8,
        )


def test_cli_refuses_overwrite(tmp_path):
    fixture = FIXTURES._scientific_fixture(tmp_path)
    manifest = sensitivity_fixture(fixture, tmp_path)
    output = tmp_path / "report.json"
    output.write_text("keep\n")
    args = ["--registry", str(fixture["registry"]), "--sensitivity-manifest", str(manifest),
            "--gec-config", str(fixture["config"]), "--resamples", "8", "--output", str(output)]
    for arm in G.ARMS:
        args += ["--prediction", f"{arm}={fixture['prediction_paths'][arm]}",
                 "--run-metadata", f"{arm}={fixture['metadata_paths'][arm]}"]
    assert S.main(args) == 2
    assert output.read_text() == "keep\n"
