import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.stacking_closure import MODES
import xens2_eval


class DummyEncoder:
    def load_state_dict(self, state):
        return None

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.mark.parametrize("mode", MODES)
def test_load_plana_restores_checkpoint_scorer(tmp_path, monkeypatch, mode):
    head = FlatDecisionHead(d_model=1280, d_z=128, scorer=mode, chunk_size=0)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {"encoder": {}, "head": head.state_dict()},
                "meta": {"scorer": mode, "d_z": 128, "resnet_blocks": 1, "resnet_hidden": 128}}, checkpoint)
    monkeypatch.setattr(xens2_eval, "load_encoder", lambda device, dtype: (DummyEncoder(), {}))
    _, loaded = xens2_eval.load_plana(str(checkpoint), "cpu")
    assert loaded.scorer == mode
    assert loaded.resnet2d.mode == mode
