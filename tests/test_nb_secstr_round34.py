import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(_ROOT / "tools"))

import nb_secstr_dual_round34 as pipeline  # noqa: E402


def test_cache_fingerprint_matches_generator_contract_for_four_splits(tmp_path):
    pretrained = tmp_path / "pretrained.pt"
    extractor = tmp_path / "extractor.py"
    pretrained.write_bytes(b"synthetic-weights")
    extractor.write_bytes(b"synthetic-generator")
    for index, role in enumerate(("train", "validation", "ts0", "archiveii600")):
        path = tmp_path / f"{role}.jsonl"
        rows = [{"seq": f"acgu{index}"}, {"seq": f"ACGT{index}"},
                {"seq": f"UUAA{index}"}, {"seq": f"GGCC{index}"}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        unique = list(dict.fromkeys(row["seq"].upper().replace("T", "U") for row in rows))[:2]
        unique.sort(key=lambda seq: (len(seq), seq))
        expected = {
            "jsonl": str(path), "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sequence_sha256": hashlib.sha256("".join(seq + "\n" for seq in unique).encode()).hexdigest(),
            "n_input_records": 4, "n_unique_sequences": 2, "max_length": 600,
            "pretrained": str(pretrained),
            "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(extractor.read_bytes()).hexdigest(), "limit": 2,
        }
        actual = pipeline.cache_fingerprint(path, rows, 2, pretrained=pretrained, extractor=extractor)
        assert actual == expected
        assert len(actual) == 10


@pytest.mark.parametrize("batch", [1, 2])
def test_loss_train_uses_upper_triangle_per_batch(batch, monkeypatch):
    created = []

    def fake_forward(head_name, head, embeddings, seq_ids, lengths):
        logits = torch.linspace(-0.5, 0.5, embeddings.shape[0] * pipeline.MAX_CORE ** 2).reshape(
            embeddings.shape[0], pipeline.MAX_CORE, pipeline.MAX_CORE).requires_grad_()
        created.append(logits)
        return logits

    monkeypatch.setattr(pipeline, "forward", fake_forward)
    records = [
        {"seq": "A", "embedding": np.zeros((1, 1024), dtype=np.float16), "pairs": []},
        {"seq": "AU", "embedding": np.zeros((2, 1024), dtype=np.float16), "pairs": [(0, 1)]},
    ][:batch]
    loss = pipeline.loss_train("official", None, records, "cpu")
    _, targets, _, _ = pipeline.collate(records, "cpu")
    logits = created[0]
    upper = torch.triu(torch.ones(pipeline.MAX_CORE, pipeline.MAX_CORE, dtype=torch.bool), diagonal=1)
    expected = torch.stack([
        F.binary_cross_entropy_with_logits(logits[row][upper], targets[row][upper])
        for row in range(batch)
    ]).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_resume_identity_and_early_stop_restoration():
    args = SimpleNamespace(head="official", seed=0, limit=32)
    identity = {"code_sha256": "code", "batch_size": 1}
    train_evidence = {"jsonl_sha256": "train"}
    valid_evidence = {"jsonl_sha256": "validation"}
    saved = {"head": "official", "seed": 0, "limit": 32, "early_stopped": True,
             "run_identity": identity, "train_evidence": train_evidence,
             "valid_evidence": valid_evidence}
    assert pipeline.resume_early_stopped(saved, args, identity, train_evidence, valid_evidence) is True
    saved["early_stopped"] = False
    assert pipeline.resume_early_stopped(saved, args, identity, train_evidence, valid_evidence) is False
    for field, message in (("seed", "resume identity mismatch"),
                           ("run_identity", "resume code/data/hyperparameter identity mismatch"),
                           ("train_evidence", "resume train/validation cache evidence mismatch")):
        broken = dict(saved)
        broken[field] = 1 if field == "seed" else {}
        with pytest.raises(RuntimeError, match=message):
            pipeline.resume_early_stopped(broken, args, identity, train_evidence, valid_evidence)
