import json

from calibrate_mxfold2_round39 import structure_counts, truth_for
from calibrate_vienna_round39 import RNA, SYSTEMS, native_prediction, prepare, sha256, validate_result


def test_native_probability_roundtrip(tmp_path):
    rows = [{"name": "a", "seq": "AUGCAUGCAUGC", "pairs": [[0, 3]]},
            {"name": "b", "seq": "GGGAAACCC", "pairs": [[0, 8], [1, 7], [2, 6]]}]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = {}
    for system in SYSTEMS:
        per_sequence = []
        tp = fp = fn = 0
        for row in rows:
            predictions, _ = native_prediction(row["seq"])
            pairs = predictions[system]
            a, b, c, f1 = structure_counts(set(pairs), truth_for(row))
            tp, fp, fn = tp + a, fp + b, fn + c
            per_sequence.append({"name": row["name"], "pred_pairs": [list(pair) for pair in pairs], "f1": f1})
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        repeats = [{"metrics": {"tp": tp, "fp": fp, "fn": fn, "micro_f1": f1}} for _ in range(3)]
        results[system] = {"per_sequence": per_sequence, "repeats": repeats}
    results["nussinov_turner"] = {"per_sequence": [], "repeats": []}
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "PASS", "data": str(data), "data_sha256": sha256(data),
                                  "n_sequences": len(rows), "vienna_version": RNA.__version__,
                                  "leakage": "not_sequence_fitted", "results": results}))
    out = tmp_path / "calibration"
    prepare(source, out)
    shared = None
    for system in SYSTEMS:
        result = json.loads((out / f"{system}.json").read_text())
        assert result["n_sequences"] == 2 and len(result["repeats"]) == 3
        assert validate_result(out / f"{system}.json")["n_sequences"] == 2
        if shared is None:
            shared = result["pair_arrays_sha256"]
        assert result["pair_arrays_sha256"] == shared
