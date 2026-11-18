import hashlib

from verify_temporal_label_collection import validate_collection_result


def test_cached_result_uses_label_file_validation(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{}")
    validate_collection_result("cached", {"chains": [1], "rejected_chains": []}, path)
    result = {"accepted_chains": 1, "rejected_chains": 0,
              "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_collection_result(result, {"chains": [1], "rejected_chains": []}, path)
