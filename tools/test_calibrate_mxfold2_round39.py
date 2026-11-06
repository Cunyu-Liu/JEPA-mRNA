import json
from types import SimpleNamespace

import numpy as np

from calibrate_mxfold2_round39 import prepare, sha256


def test_native_bpp_above_one_is_terminal_failure_without_ece(tmp_path, monkeypatch):
    import torch
    from mxfold2.predict import Predict

    row = {"name": "x", "seq": "AUGCAUGC", "pairs": []}
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(row) + "\n")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"test")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "PASS", "clean_n": 1, "clean_sha256": sha256(data)}))
    protocol = {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                "n_length_filtered": 0, "sample_scope": "full_source_file", "data": str(data),
                "data_sha256": sha256(data), "clean_audit": str(audit), "n_sequences": 1,
                "model_report": {"weights": str(weights), "weights_sha256": sha256(weights),
                                 "config": {}, "n_parameters": 0}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "PASS", "protocol": protocol, "f1_runs": [1.0] * 3,
                                  "predictions": [{"name": "x", "pairs": []}]}))

    class FakeModel:
        def parameters(self):
            return iter([SimpleNamespace(is_cuda=True, numel=lambda: 0)])

        def load_state_dict(self, state, strict):
            assert strict

        def cuda(self):
            return self

        def eval(self):
            return self

        def __call__(self, seqs, return_partfunc):
            length = len(seqs[0])
            return [0.0], ["." * length], [[]], [0.0], [np.full((length + 1, length + 1), 1.1)]

    model = FakeModel()
    monkeypatch.setattr(Predict, "build_model", lambda self, args: (model, {}))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    out = tmp_path / "calibration" / "split.json"
    prepare(source, out)
    result = json.loads(out.read_text())
    assert result["status"] == "FAIL_NATIVE_BPP_RANGE_NOT_PROBABILITY_ELIGIBLE"
    assert result["calibration"] == {} and result["pair_arrays"] is None
    assert result["source_structure_f1_runs"] == [1.0] * 3
    assert result["failure"]["above_one_in_scored_mask"] > 0
    assert not out.with_suffix(".npz").exists()
