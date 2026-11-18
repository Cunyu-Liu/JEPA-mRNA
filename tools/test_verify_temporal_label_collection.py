import hashlib
import pytest

from verify_temporal_label_collection import validate_collection_result, validate_mapping_failures, resolve_error_result


def test_cached_result_uses_label_file_validation(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{}")
    validate_collection_result("cached", {"chains": [1], "rejected_chains": []}, path)
    result = {"accepted_chains": 1, "rejected_chains": 0,
              "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_collection_result(result, {"chains": [1], "rejected_chains": []}, path)


def test_mapping_failures_must_match_rejection_and_not_be_accepted():
    label = {"mapping_failures": {"A": "unsupported modified RNA residue: X"},
             "rejected_chains": [{"auth_chain": "A", "reason": "unsupported modified RNA residue: X"}], "chains": []}
    validate_mapping_failures(label)
    label["chains"] = [{"auth_chain": "A"}]
    with pytest.raises(AssertionError):
        validate_mapping_failures(label)


def test_recovered_error_requires_archive_and_label(tmp_path):
    (tmp_path / "error.shared_cwd_failure.txt").write_text("old failure")
    (tmp_path / "labels.json").write_text("{}")
    state, detail = resolve_error_result({"error": "old failure"}, tmp_path)
    assert state == "recovered"
    assert detail["recovery_logs"] == ["error.shared_cwd_failure.txt"]
    (tmp_path / "error.txt").write_text("current failure")
    assert resolve_error_result({"error": "current failure"}, tmp_path)[0] == "error"
