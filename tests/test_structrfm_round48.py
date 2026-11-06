import json

import pytest
import torch

from tools.structrfm_round48 import file_sha256, install_structrfm_training


class DummyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"A": 3, "C": 4, "G": 5, "U": 6, "N": 7}[token]


def test_sha256(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_training_vocab_uses_official_special_tokens(monkeypatch):
    import tools.train_plan_a as base
    import tools.xens2_eval as xens2_eval
    import tools.structrfm_round48 as adapter

    monkeypatch.setattr(adapter, "load_structrfm_encoder",
                        lambda path: (None, DummyTokenizer(), {"strict_identity": True}))
    install_structrfm_training(base, xens2_eval, "unused")
    assert base.VOCAB["<cls>"] == 1
    assert base.VOCAB["<eos>"] == 2
    assert base.VOCAB["T"] == base.VOCAB["U"]
    assert base.VOCAB["A"] == 3
