from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import zipfile

import pytest
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ppt/publish_external_results_round72.py"
SPEC = importlib.util.spec_from_file_location("publish_external_results_round72", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
ROUND71_TEST_PATH = ROOT / "tests/test_phase0_structured_margin_result_round71.py"
R71_SPEC = importlib.util.spec_from_file_location("round71_fixture_for_publication", ROUND71_TEST_PATH)
R71 = importlib.util.module_from_spec(R71_SPEC)
assert R71_SPEC.loader is not None
R71_SPEC.loader.exec_module(R71)
DECK = ROOT / "ppt/RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"


def fixture_root(tmp_path: Path) -> Path:
    configured = os.environ.get("ROUND72_PUBLICATION_FIXTURE_ROOT")
    return Path(configured) if configured else tmp_path


def evaluation_fixture(root: Path, scientific_pass: bool) -> dict:
    label = "pass" if scientific_pass else "fail"
    case = root / label
    evaluation = R71.make_fixture(case / "evaluation", scientific_pass)
    cross_source = R71.make_cross_source_fixture(case / "cross_source", M.MODEL_DEV_REGISTRY)
    return {"case": case, "evaluation": evaluation, "cross_source": cross_source}


def publish_fixture(fixture: dict, suffix: str = "delivery", **overrides) -> dict:
    case = fixture["case"]
    options = {"delivery_deck": case / f"{suffix}.pptx", "work_dir": case / f"{suffix}_work"}
    options.update(overrides)
    return M.run_publication(delivery_deck=options.pop("delivery_deck"), work_dir=options.pop("work_dir"),
                             evaluation_root=fixture["evaluation"],
                             cross_source_root=fixture["cross_source"], **options)


def zip_parts(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}


def assert_success(report: dict, fixture: dict, scientific_pass: bool, source_hash: str) -> None:
    delivery = Path(report["delivery_path"])
    backup = Path(report["backup_path"])
    assert len(Presentation(delivery).slides) == 17
    assert len(Presentation(backup).slides) == 14
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == source_hash
    assert len(Presentation(DECK).slides) == 14
    assert report["scientific_overall_pass"] is scientific_pass
    assert report["automatic_git_commit"] is False
    assert len(report["steps"]) == 6
    expected_names = ["round71_readiness", "round71_build_page34",
                      "round71_independent_verification", "round72_readiness",
                      "round72_build_pages35_36", "round72_independent_verification"]
    assert [step["name"] for step in report["steps"]] == expected_names
    for step in report["steps"]:
        assert step["exit_code"] == 0
        assert Path(step["stdout_log"]).is_file() and Path(step["stderr_log"]).is_file()
        assert step["deck_sha256_before"] and step["deck_sha256_after"]
        assert step["source_deck_sha256"] == source_hash
        assert not any(Path(part).name == "git" for part in step["command"])
    transitions = [(step["deck_sha256_before"], step["deck_sha256_after"]) for step in report["steps"]]
    assert transitions[0][0] == transitions[0][1] == source_hash
    assert transitions[1][0] != transitions[1][1]
    assert transitions[2][0] == transitions[2][1]
    assert transitions[3][0] == transitions[3][1]
    assert transitions[4][0] != transitions[4][1]
    assert transitions[5][0] == transitions[5][1]
    M.verify_prior_xml(DECK, delivery)
    independent = json.loads((Path(report["work_dir"]) /
                              "transaction/round72_independent_build_verification.json").read_text())
    assert independent["exposure_embedded_matches"] is True
    assert independent["exposure_cluster_dev_counts"] == R71.M.EXPOSURE_COUNTS


def test_synthetic_scientific_pass_and_fail_publish_atomically_without_touching_source(tmp_path):
    source_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    root = fixture_root(tmp_path)
    for scientific_pass in (True, False):
        fixture = evaluation_fixture(root, scientific_pass)
        report = publish_fixture(fixture)
        assert_success(report, fixture, scientific_pass, source_hash)
        assert report["scientific_status"] == ("PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
                                               if scientific_pass else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL")
    assert hashlib.sha256(DECK.read_bytes()).hexdigest() == source_hash


def test_independent_pass_replay_has_identical_zip_parts(tmp_path):
    source_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    fixture = evaluation_fixture(tmp_path / "replay", True)
    first = publish_fixture(fixture, "first")
    second = publish_fixture(fixture, "second")
    assert_success(first, fixture, True, source_hash)
    assert_success(second, fixture, True, source_hash)
    assert zip_parts(Path(first["delivery_path"])) == zip_parts(Path(second["delivery_path"]))


def test_in_progress_missing_exposure_hash_error_and_mid_pipeline_failure_leave_no_delivery(tmp_path):
    source_hash = hashlib.sha256(DECK.read_bytes()).hexdigest()
    for mode in ("IN_PROGRESS", "MISSING_EXPOSURE", "HASH_ERROR", "MID_PIPELINE"):
        fixture = evaluation_fixture(tmp_path / mode, True)
        evaluation_path = fixture["evaluation"] / "independent_evaluation.json"
        if mode == "IN_PROGRESS":
            evaluation = json.loads(evaluation_path.read_text())
            evaluation.update(status="IN_PROGRESS", final_verdict_allowed=False)
            evaluation_path.write_text(json.dumps(evaluation))
        elif mode == "MISSING_EXPOSURE":
            (fixture["evaluation"] / "exposure_sensitivity.json").unlink()
        elif mode == "HASH_ERROR":
            prediction = fixture["evaluation"] / "predictions/ensemble_context/dev_predictions.jsonl"
            prediction.write_text(prediction.read_text() + "changed\n")
        options = {}
        if mode == "MID_PIPELINE":
            options["panel_registry"] = tmp_path / "missing_official_reference_registry.json"
        with pytest.raises(M.PublicationError):
            publish_fixture(fixture, **options)
        case = fixture["case"]
        assert not (case / "delivery.pptx").exists()
        assert not (case / "delivery.source14_backup.pptx").exists()
        assert hashlib.sha256(DECK.read_bytes()).hexdigest() == source_hash
        assert len(Presentation(DECK).slides) == 14
        failure = json.loads((case / "delivery_work/publication_failure.json").read_text())
        assert failure["status"] == "FAILED_NO_PARTIAL_DELIVERY"
        assert failure["delivery_exists"] is False
        if mode in ("IN_PROGRESS", "MISSING_EXPOSURE"):
            assert failure["completed_steps"] == []
        elif mode == "HASH_ERROR":
            assert failure["completed_steps"] == []
            assert failure["step_records"][0]["exit_code"] != 0
        else:
            assert failure["completed_steps"] == ["round71_readiness", "round71_build_page34",
                                                   "round71_independent_verification"]
            assert len(failure["step_records"]) == 4


def test_wait_receives_late_final_pass_marker(tmp_path):
    fixture = evaluation_fixture(tmp_path / "wait", True)
    path = fixture["evaluation"] / "independent_evaluation.json"
    final = path.read_text()
    evaluation = json.loads(final)
    evaluation.update(status="IN_PROGRESS", final_verdict_allowed=False)
    path.write_text(json.dumps(evaluation))
    timer = threading.Timer(0.15, lambda: path.write_text(final))
    timer.start()
    try:
        marker = M.wait_for_final_marker(fixture["evaluation"], wait=True,
                                         timeout_seconds=3, poll_interval=0.03)
    finally:
        timer.join()
    assert marker["ready"] and marker["overall_pass"] is True
    assert marker["wait_attempts"] >= 1
    assert set(marker["files"]) == set(M.FINAL_FILES)
