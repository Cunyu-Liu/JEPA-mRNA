import json

import numpy as np
import pytest
import torch

from rnajepa.train_decision import (EmbeddingStore, TrainConfig, TeacherLabelStore,
                                    load_dataset_from_jsonl, make_synthetic_dataset,
                                    run_training, write_mock_teacher_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pairjepa_auxiliary_training_updates_ema(tmp_path):
    sequences = make_synthetic_dataset(4, 8, seed=0)
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"seq": example.seq, "pairs": example.gt_pairs}) + "\n"
                            for example in sequences))
    write_mock_teacher_dir([example.seq for example in sequences], str(tmp_path / "teacher"))
    total = sum(len(example.seq) for example in sequences)
    rng = np.random.default_rng(7)
    h = rng.normal(size=(total, 8)).astype(np.float16)
    lengths = np.asarray([len(example.seq) for example in sequences], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    seqs = np.asarray([example.seq for example in sequences], dtype=object)
    shard = tmp_path / "embeddings/train.shard0of1.npz"
    shard.parent.mkdir()
    np.savez(shard, h=h, offsets=offsets, lengths=lengths, seqs=seqs)
    entry = {"split": "train", "path": str(shard), "n_sequences": len(sequences),
             "n_residues": total, "d_model": 8, "dtype": "float16"}
    (tmp_path / "embeddings/manifest.json").write_text(json.dumps({
        "model": "synthetic-engineering", "entries": [entry, dict(entry, split="bprna_vl0")]}))
    config = TrainConfig(out_dir=str(tmp_path / "run"), embedding_dir=str(tmp_path / "embeddings"),
                         embedding_d_model=8, scorer="pairjepa", lambda_jepa=0.1,
                         steps=2, batch_size=2, d_z=16, hidden=16, head_chunk_size=0,
                         device="cuda", log_every=1, save_every=0)
    teacher = TeacherLabelStore.from_dir(str(tmp_path / "teacher"))
    store = EmbeddingStore.from_dir(str(tmp_path / "embeddings"), split="train")
    dataset = load_dataset_from_jsonl(str(data), teacher, embedding_store=store)
    result = run_training(config, dataset, teacher)
    assert "jepa" in result["run_meta"]["objective_terms_active"]
    rows = [json.loads(line) for line in (tmp_path / "run/train_log.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all("jepa" in row["terms"] for row in rows)
    checkpoint = torch.load(tmp_path / "run/resume.pt", map_location="cpu")
    assert checkpoint["step"] == 2
    model_state = checkpoint["model"]
    for name, tensor in model_state.items():
        if name.startswith("head.pair_jepa.scorer."):
            teacher_tensor = model_state[name.replace("scorer", "teacher", 1)]
            assert not torch.equal(tensor, teacher_tensor), name
