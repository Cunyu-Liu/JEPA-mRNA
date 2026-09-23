"""The frozen-embedding path: lookup by sequence, and no mixing of two sources.

Two things here would fail silently in production and are therefore tested
directly:

* ``EmbeddingStore`` must key on the **sequence**, not the position.  The extractor
  excludes sequences longer than the backbone's positional limit, so a positional
  index would shift every embedding after the first exclusion -- and a shifted
  embedding still has a valid shape, so nothing downstream would complain.
* A batch must not mix cached embeddings with on-the-fly encoder activations.
  ``objective_terms`` rejects that combination rather than averaging two
  incomparable representations.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import rnajepa.train_decision as TD  # noqa: E402
from rnajepa.decision_head import HeadOnlyModel  # noqa: E402
from rnajepa.rlcd import ObjectiveWeights  # noqa: E402

D = 8
SEQS = ["GGGAAACCCUUU", "ACGUACGUACGU", "UUUAAAGGGCCC"]


def _write_embedding_dir(tmp_path, *, excluded: int = 0):
    """A minimal shard in the layout ``tools/extract_rinalmo_embeddings.py`` writes."""
    h = np.concatenate([np.full((len(s), D), i + 1, dtype=np.float16)
                        for i, s in enumerate(SEQS)], axis=0)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in SEQS])]).astype(np.int64)
    path = tmp_path / "tiny.shard0of1.npz"
    np.savez_compressed(path, h=h, offsets=offsets, lengths=np.array([len(s) for s in SEQS]),
                        seqs=np.array(SEQS, dtype=object))
    manifest = {
        "model": "test", "dtype": "float16",
        "entries": [{"split": "tiny", "shard": 0, "shards": 1, "path": str(path),
                     "n_sequences": len(SEQS), "n_residues": int(h.shape[0]),
                     "d_model": D, "dtype": "float16", "skipped": False,
                     "range": [0, len(SEQS)],
                     "n_excluded_over_max_len": excluded, "excluded_lengths": [2048] * excluded}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _write_jsonl(tmp_path):
    path = tmp_path / "tiny.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for seq in SEQS:
            fh.write(json.dumps({"seq": seq, "structure": "." * len(seq)}) + "\n")
    return path


def test_embedding_store_looks_up_by_sequence(tmp_path):
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path)))
    assert store.d_model == D
    assert store.n_sequences == len(SEQS)
    # each sequence returns its own block, identified by the constant we wrote
    for i, seq in enumerate(SEQS):
        block = store.get(seq)
        assert block.shape == (len(seq), D)
        assert np.allclose(block, i + 1)


def test_embedding_store_refuses_to_fall_back_for_a_missing_sequence(tmp_path):
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path)))
    with pytest.raises(TD.ConfigError, match="no cached embedding"):
        store.get("ACGUACGUACGUACGU")


def test_embedding_store_records_excluded_sequences(tmp_path):
    """Exclusions must be visible, because a missing embedding is a real absence."""
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path, excluded=2)))
    assert "excluded_over_max_len=2" in store.source


def test_build_decision_model_with_embeddings_returns_a_head_only_model(tmp_path):
    cfg = TD.TrainConfig(tiny=True, embedding_dir=str(tmp_path), embedding_d_model=D)
    model = TD.build_decision_model(cfg)
    assert isinstance(model, HeadOnlyModel)
    assert model.head_only is True
    # the head must be sized from the cached embeddings, not from config.d_model
    assert model.head.pair_repr.proj.in_features == 3 * D


def test_embedding_dir_without_d_model_is_rejected(tmp_path):
    cfg = TD.TrainConfig(tiny=True, embedding_dir=str(tmp_path))
    with pytest.raises(TD.ConfigError, match="requires --embedding-d-model"):
        TD.build_decision_model(cfg)


def test_collate_and_device_transfer_carry_embeddings(tmp_path):
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path)))
    ds = TD.load_dataset_from_jsonl(str(_write_jsonl(tmp_path)),
                                    TD.TeacherLabelStore.from_mock(),
                                    need_teacher_probs=False,
                                    embedding_store=store)
    assert ds.describe()["has_embeddings"] is True
    batch = TD.collate(list(ds.examples))
    assert batch["embeddings"][0].shape == (len(SEQS[0]), D)


def test_objective_rejects_mixing_cached_and_encoder_activations(tmp_path):
    """Half frozen / half trainable representations must be an error, not a mean."""
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path)))
    ds = TD.load_dataset_from_jsonl(str(_write_jsonl(tmp_path)),
                                    TD.TeacherLabelStore.from_mock(),
                                    need_teacher_probs=False,
                                    embedding_store=store)
    batch = TD.collate(list(ds.examples))
    weights = ObjectiveWeights(lambda_nll=1.0, lambda_distill=0.0,
                              lambda_rlcd=0.0, lambda_cal=0.0)

    head_only = TD.build_decision_model(
        TD.TrainConfig(tiny=True, embedding_dir=str(tmp_path), embedding_d_model=D))
    head_only(torch.as_tensor(batch["embeddings"][0], dtype=torch.float32).unsqueeze(0),
              batch["seq_ids"][:1])  # sanity: the head accepts (B, L, D)

    # a head-only model given a batch without embeddings
    bare = dict(batch)
    bare["embeddings"] = [None] * len(batch["embeddings"])
    with pytest.raises(TD.ConfigError, match="disagree about where representations"):
        TD.objective_terms(head_only, bare, weights)

    # an encoder model given a batch with embeddings
    encoder_model = TD.build_decision_model(TD.TrainConfig(tiny=True))
    assert not getattr(encoder_model, "head_only", False)
    with pytest.raises(TD.ConfigError, match="disagree about where representations"):
        TD.objective_terms(encoder_model, batch, weights)


def test_head_only_run_trains_end_to_end_on_cpu(tmp_path):
    """The whole driver path: cached embeddings -> head -> loss -> checkpoint."""
    store = TD.EmbeddingStore.from_dir(str(_write_embedding_dir(tmp_path)))
    ds = TD.load_dataset_from_jsonl(str(_write_jsonl(tmp_path)),
                                    TD.TeacherLabelStore.from_mock(),
                                    need_teacher_probs=False,
                                    embedding_store=store)
    cfg = TD.TrainConfig(steps=3, batch_size=2, tiny=True, seed=0,
                         out_dir=str(tmp_path / "run"), device="cpu", allow_cpu=True,
                         lambda_distill=0.0, lambda_rlcd=0.0, lambda_cal=0.0,
                         log_every=1, embedding_dir=str(tmp_path), embedding_d_model=D)
    result = TD.run_training(cfg, ds, TD.TeacherLabelStore.from_mock())
    assert result["steps_completed"] == 3
    assert os.path.isfile(os.path.join(cfg.out_dir, "resume.pt"))
    meta = json.loads((tmp_path / "run" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["embedding_dir"] == str(tmp_path)
    assert meta["data"]["has_embeddings"] is True
    # no encoder was built, so the parameter count must be head-sized
    assert meta["model"]["n_params"] < 5_000_000
