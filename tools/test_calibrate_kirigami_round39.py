import hashlib
import json

import numpy as np

from benchmark_eternafold_round38 import canonical
from calibrate_kirigami_round39 import pair_confidence, prepare, sha256, validate_result


def test_prepare_and_validate_native_probabilities(tmp_path):
    rows = [{"name": "example", "seq": "AUGCAUGC", "pairs": [[0, 3], [1, 2]]}]
    data = tmp_path / "clean.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    weights = tmp_path / "weights.ckpt"
    weights.write_bytes(b"test-only-weights")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "PASS", "clean_n": 1, "clean_sha256": sha256(data)}))
    matrix = np.triu(np.full((8, 8), 0.25, dtype=np.float32), k=1)
    probabilities = tmp_path / "source.npz"
    saved_matrices = np.empty(1, dtype=object)
    saved_matrices[0] = matrix
    np.savez_compressed(probabilities, names=np.array(["example"], dtype=object),
                        probabilities=saved_matrices)
    pairs = [[0, 7], [1, 6]]
    tp, fp, fn = 0, 2, 2
    f1 = 2 * tp / (2 * tp + fp + fn)
    identity = [["example", pairs]]
    repeat = {"prediction_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
              "probability_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
              "metrics": {"tp": tp, "fp": fp, "fn": fn, "micro_f1": f1},
              "elapsed_seconds": 1.0}
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "PASS", "data": str(data), "data_sha256": sha256(data),
                                  "clean_audit": str(audit), "weights": str(weights),
                                  "weights_sha256": sha256(weights), "n_sequences": 1,
                                  "device": "cuda", "probabilities": str(probabilities),
                                  "probabilities_sha256": sha256(probabilities),
                                  "repeats": [repeat, repeat, repeat],
                                  "per_sequence": [{"name": "example", "pred_pairs": pairs, "f1": f1}]}))
    out = tmp_path / "calibration.json"
    prepare(source, out)
    result = json.loads(out.read_text())
    assert result["per_sequence"][0]["mean_selected_pair_probability"] == 0.25
    assert result["per_sequence"][0]["structure_f1"] == 0.0
    assert result["f1_runs"] == [0.0, 0.0, 0.0]
    assert validate_result(out)["n_sequences"] == 1


def test_pair_confidence_empty_and_pseudoknot():
    probabilities = np.zeros((8, 8), dtype=np.float32)
    probabilities[0, 7] = 0.4
    probabilities[1, 5] = 0.6
    assert pair_confidence([], probabilities) == 0.0
    assert pair_confidence({(0, 7), (1, 5)}, probabilities) == 0.5
