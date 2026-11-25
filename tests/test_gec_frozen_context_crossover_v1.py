from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import gec_frozen_context_crossover_v1 as C


def test_decoder_and_four_arm_contract_cannot_drift():
    config = {"arms": list(C.ARMS), "training": {"seed": 0},
              "model": {"d_z": 128, "residual_hidden": 64, "decoder": copy.deepcopy(C.DECODER)}}
    C.validate_config(config)
    for key, value in (("gamma", 2.0), ("temperature", 0.5), ("search_allowed", True)):
        changed = copy.deepcopy(config)
        changed["model"]["decoder"][key] = value
        with pytest.raises(ValueError, match="decoder identity"):
            C.validate_config(changed)


def test_hand_checked_pair_stem_isolated_and_span_metrics():
    records = [{"id": "x", "sequence": "A" * 20, "pairs": [(0, 11), (1, 10), (4, 15)]}]
    families = {"x": "RF1"}
    accumulator = C.MetricAccumulator(records, families, ["RF1"])
    accumulator.update(records[0], [(0, 11), (2, 9), (4, 15)])
    result = accumulator.export()
    assert result["pair_confusion_tp_fp_fn"] == [2, 1, 1]
    assert result["stem_confusion_tp_fp_fn"] == [1, 2, 1]
    assert result["micro_pair_f1"] == pytest.approx(2 / 3)
    assert result["micro_stem_f1"] == pytest.approx(0.4)
    assert result["isolated_pair:isolated"]["confusion_tp_fp_fn"] == [1, 2, 0]
    assert result["pair_span:8_15"]["confusion_tp_fp_fn"] == [2, 0, 1]


def test_macro_uses_only_registered_eligible_family_means():
    records = [{"id": f"s{i}", "sequence": "A" * 12,
                "pairs": [(0, 11)]} for i in range(31)]
    families = {f"s{i}": ("RF1" if i < 30 else None) for i in range(31)}
    accumulator = C.MetricAccumulator(records, families, ["RF1"])
    for index, record in enumerate(records):
        accumulator.update(record, [(0, 11)] if index % 2 == 0 else [])
    result = accumulator.export()
    assert result["macro_pair_recall"] == pytest.approx(0.5)
    assert result["micro_pair_recall"] == pytest.approx(16 / 31)


def test_smoke_metrics_explicitly_mark_macro_not_applicable():
    records = [{"id": "x", "sequence": "A" * 12, "pairs": [(0, 11)]}]
    accumulator = C.MetricAccumulator(records, {"x": "RF1"}, ["RF1"], macro_enabled=False)
    accumulator.update(records[0], [(0, 11)])
    result = accumulator.export()
    assert result["micro_pair_f1"] == 1.0
    assert result["macro_pair_f1"] is None


def test_native_pair_order_is_exact_after_normalization():
    assert C.native_pairs({"predicted_pairs": [[4, 8], [0, 7]]}, 10) == ((0, 7), (4, 8))
    with pytest.raises(ValueError, match="duplicate pair or endpoint"):
        C.native_pairs({"predicted_pairs": [[0, 7], [7, 9]]}, 10)


def test_crossover_computes_base_once_and_shares_ensemble_numerics(monkeypatch):
    record = {"id": "x", "sequence": "ACGUACGU", "sequence_sha256": "abc", "pairs": [(0, 7)]}
    first = torch.zeros(8, 8)
    representation = torch.zeros(8, 8, 128)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[0, 7] = mask[7, 0] = True
    calls = []
    monkeypatch.setattr(C, "encode_base", lambda *args: (first, representation, mask))
    monkeypatch.setattr(C, "gec_geometry_v1", lambda *args: torch.zeros(8, 8, 8))
    monkeypatch.setattr(C, "decode_scores", lambda scores, legal: ((0, 7),))

    class Numerics:
        marginals = torch.ones(8, 8, dtype=torch.float64)
        log_z = torch.tensor(2.0, dtype=torch.float64)
        context = torch.zeros(8, 8, 6, dtype=torch.float64)

    def context(*args, **kwargs):
        calls.append((args, kwargs))
        return Numerics()

    class Residual:
        def __call__(self, pair, numerics, geometry, legal):
            assert pair is representation
            return torch.zeros(8, 8)

    monkeypatch.setattr(C, "compute_gec_context_v1", context)
    head = type("Head", (), {"min_loop": 3})()
    rows = C.crossover_record(record, object(), head, Residual(), "cpu")
    assert set(rows) == set(C.ARMS)
    assert len(calls) == 5
    assert "marginals" not in calls[0][1]
    for args, kwargs in calls[1:]:
        assert kwargs["marginals"] is Numerics.marginals
        assert kwargs["log_z"] is Numerics.log_z
        assert kwargs["shuffle_key"].startswith("gec_destroy_v1|0|0|0|")


def test_decode_is_float64_exact_io_with_fixed_mea(monkeypatch):
    observed = {}

    def inside(scores, mask):
        observed["dtype"] = scores.dtype
        return 0.0, np.ones((5, 5))

    def decode(posterior, mask, gamma):
        observed["gamma"] = gamma
        return {(0, 4)}

    monkeypatch.setattr(C.harness, "inside_outside", inside)
    monkeypatch.setattr(C, "decode_mea", decode)
    result = C.decode_scores(torch.ones(5, 5), torch.ones(5, 5, dtype=torch.bool))
    assert result == ((0, 4),)
    assert observed == {"dtype": np.dtype("float64"), "gamma": 1.0}


def test_frozen_modules_require_eval_cuda_and_no_grad():
    module = torch.nn.Linear(2, 1).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="not entirely on CUDA"):
        C.require_frozen((("head", module),))
    module.train()
    with pytest.raises(ValueError, match="not in eval mode"):
        C.require_frozen((("head", module),))


def test_atomic_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "rows.jsonl.gz"
    C.atomic_jsonl_gz([{"x": 1}], path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        C.atomic_jsonl_gz([{"x": 2}], path)
    assert path.read_bytes() == before
    report = tmp_path / "summary.json"
    C.atomic_json({"ok": True}, report)
    with pytest.raises(FileExistsError):
        C.atomic_json({"ok": False}, report)
    assert json.loads(report.read_text()) == {"ok": True}
