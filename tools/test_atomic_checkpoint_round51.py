import os

import pytest
import torch

from tools.train_plan_a import atomic_torch_save


def test_atomic_save_replaces_complete_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1, "weight": torch.tensor([2.0])}, path)
    result = torch.load(path, weights_only=False)
    assert result["step"] == 1
    assert result["weight"].item() == 2.0
    assert not list(tmp_path.glob("checkpoint.pt.tmp.*"))


def test_atomic_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous-checkpoint")

    def fail(*args):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="injected"):
        atomic_torch_save({"step": 2}, path)
    assert path.read_bytes() == b"previous-checkpoint"
    assert not list(tmp_path.glob("checkpoint.pt.tmp.*"))
