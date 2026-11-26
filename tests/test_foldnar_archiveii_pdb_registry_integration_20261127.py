import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "registry_integration_verifier",
    ROOT / "tools/verify_foldnar_archiveii_pdb_registry_integration_20261127.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def load(path):
    return json.loads((ROOT / path).read_text())


def test_three_round_independent_verification_passes():
    for round_number in (1, 2, 3):
        result = VERIFIER.verify(round_number)
        assert result["verification_passed"], result["failures"]
        assert result["round"] == round_number
        assert result["builder_imported"] is False


def test_partial_coverage_cannot_be_ranked():
    partial = {
        "status": "PARTIAL_COVERAGE_NOT_RANKED",
        "n_with_prediction": 2544,
        "n_panel_sequences": 3966,
        "coverage": 2544 / 3966,
        "ranking_eligible": False,
    }
    assert not VERIFIER.ranking_eligible(partial)
    partial["ranking_eligible"] = True
    assert not VERIFIER.ranking_eligible(partial)


def test_full_coverage_requires_complete_counts():
    panel = {
        "status": "PASS_FULL_COVERAGE",
        "n_with_prediction": 3966,
        "n_panel_sequences": 3966,
        "coverage": 1.0,
        "ranking_eligible": True,
    }
    assert VERIFIER.ranking_eligible(panel)
    panel["n_with_prediction"] = 3965
    assert not VERIFIER.ranking_eligible(panel)


def test_archive_legacy_assets_are_circular_audit_only():
    registry = load("manifests/foldnar_v1/prediction_registry.json")
    rows = [row for row in registry["rows"] if row.get("dataset_id") == "archiveii_embok_clean"]
    assert len(rows) == 32
    assert all(row["label_lineage"]["disposition"] == "circular-audit-only" for row in rows)
    assert all(row["independent_sota_eligible"] is False for row in rows)


def test_archive_full_panel_has_only_one_ranking_identity():
    overlay = load("manifests/foldnar_v1/archiveii_official_reference_prediction_registry.json")
    eligible = [
        row["prediction_id"] for row in overlay["rows"]
        if VERIFIER.ranking_eligible(row["panels"]["archiveii_official_reference_3966"])
    ]
    assert eligible == ["bpfold_official_release/archiveii"]
    assert overlay["baseline_summary"]["official_full_3966"]["strongest_single_predictor"] == "UNCHECKED_NO_FULL_COVERAGE_SINGLE_PREDICTOR"


def test_pdb_rebuild_conserves_153_records():
    data = load("manifests/foldnar_v1/data_registry.json")
    rebuild = data["pdb_raw_source_rebuild"]
    assert rebuild["n_source_records"] == 153
    assert rebuild["n_rebuilt"] == 58
    assert rebuild["n_unresolved"] == 95
    assert sum(rebuild["panel_rebuilt"].values()) == 58
    assert sum(rebuild["unresolved_status_counts"].values()) == 95
    assert all(status.endswith("UNCHECKED") for status in rebuild["unresolved_status_counts"])


def test_registry_verifier_rejects_partial_ranking_mutation(monkeypatch):
    overlay_path = ROOT / "manifests/foldnar_v1/archiveii_official_reference_prediction_registry.json"
    document = json.loads(overlay_path.read_text())
    row = next(row for row in document["rows"] if row["prediction_id"] == "bpfold/archiveii_embok_clean")
    row["panels"]["archiveii_official_reference_3966"]["ranking_eligible"] = True
    original_read_json = VERIFIER.read_json

    def mocked_read_json(path):
        if Path(path) == overlay_path:
            return document
        return original_read_json(path)

    monkeypatch.setattr(VERIFIER, "read_json", mocked_read_json)
    result = VERIFIER.verify(1)
    assert not result["verification_passed"]
    assert "partial_not_ranked:archiveii_official_reference_3966:bpfold/archiveii_embok_clean" in result["failures"]
