import hashlib
import json
from pathlib import Path

import pytest

from tools.freeze_xens2_turneroff_mea import EVIDENCE_FILES, freeze_candidate


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_evidence(root: Path):
    for name in EVIDENCE_FILES:
        write_json(root / name, {})
    write_json(root / "final_pair_report/final_pair_diagnostic_decision.json", {
        "passing_candidates": [{"score_variant": "xens2_off", "decoder": "mea", "vl0_delta_micro_f1": 0.0222775}],
        "turner_on_passing_candidates": [],
    })
    write_json(root / "decode_xens2_vl0/frozen_decode_config.json", {
        "selection_policy": "micro F1, then macro F1, then micro precision; no external selection",
        "config": {"xens2_off": {"mea": {"temperature": 2.0, "gamma": 4.0}}},
    })
    write_json(root / "decode_xens2_vl0/decode_run_manifest.json", {"official": True})
    write_json(root / "raw_scores_chunk64/raw_xens2_full_verification.json", {"status": "PASS", "n_sequences": 708})
    write_json(root / "decode_xens2_decision/independent_route_decision_verification.json", {"status": "PASS", "n_checks": 42})
    write_json(root / "decode_xens2_external/independent_external_raw_dp_verification.json", {"status": "PASS", "n_checks": 15})


def make_weights(tmp_path: Path):
    specs = {}
    for index, name in enumerate(("plana0", "plana1", "r2d")):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(f"weight-{index}".encode())
        specs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "weight": (0.35, 0.35, 0.30)[index]}
    return specs


def test_freeze_candidate_identity_and_archive(tmp_path):
    root = tmp_path / "artifacts"
    make_evidence(root)
    out = tmp_path / "frozen"
    manifest = freeze_candidate(root, out, make_weights(tmp_path))
    assert manifest["status"] == "PASS"
    candidate = json.loads((out / "candidate.json").read_text())
    assert candidate["display_name"] == "xens2-TurnerOff-MEA"
    assert candidate["distinct_from_base_identity"] is True
    assert candidate["production_repair"] is False
    assert candidate["sota_claim"] is False
    assert candidate["decoder"] == {
        "name": "inside_outside_posterior_mea", "temperature": 2.0, "gamma": 4.0,
        "legal_pair_definition": "AU/UA, GC/CG, GU/UG; j-i>3", "structure_class": "nested",
    }
    assert len(manifest["archived_evidence"]) == len(EVIDENCE_FILES)
    with pytest.raises(FileExistsError):
        freeze_candidate(root, out, make_weights(tmp_path))


def test_freeze_rejects_changed_parameters(tmp_path):
    root = tmp_path / "artifacts"
    make_evidence(root)
    config_path = root / "decode_xens2_vl0/frozen_decode_config.json"
    config = json.loads(config_path.read_text())
    config["config"]["xens2_off"]["mea"]["temperature"] = 1.0
    write_json(config_path, config)
    with pytest.raises(ValueError, match="unexpected frozen"):
        freeze_candidate(root, tmp_path / "frozen", make_weights(tmp_path))


def test_freeze_rejects_weight_mismatch(tmp_path):
    root = tmp_path / "artifacts"
    make_evidence(root)
    specs = make_weights(tmp_path)
    specs["plana0"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weight SHA256 mismatch"):
        freeze_candidate(root, tmp_path / "frozen", specs)
