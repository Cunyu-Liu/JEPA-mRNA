import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import rnajepa.train_decision as TD  # noqa: E402


def test_embedding_required_subset_and_hash_are_enforced(tmp_path):
    sequences = np.array(["ACGU", "GGCA", "UUAA"], dtype=object)
    lengths = [4, 4, 4]
    offsets = np.array([0, 4, 8, 12], dtype=np.int64)
    h = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    shard = tmp_path / "toy.npz"
    np.savez(shard, h=h, offsets=offsets, seqs=sequences)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [{
        "split": "toy", "path": str(shard), "sha256": TD.sha256_file(shard),
    }]}), encoding="utf-8")

    store = TD.EmbeddingStore.from_dir(
        str(tmp_path), split="toy", required_sequences={"GGCA"})
    assert store.n_sequences == 1
    assert store.get("GGCA").shape == (4, 2)
    assert store.get("GGCA").dtype == np.float16
    np.testing.assert_array_equal(store.get("GGCA"), h[4:8])
    assert "required_sequences=1" in store.source

    with pytest.raises(TD.ConfigError, match="missing 1 required sequences"):
        TD.EmbeddingStore.from_dir(
            str(tmp_path), split="toy", required_sequences={"ACGU", "AAAA"})

    manifest.write_text(json.dumps({"entries": [{
        "split": "toy", "path": str(shard), "sha256": "0" * 64,
    }]}), encoding="utf-8")
    with pytest.raises(TD.ConfigError, match="refusing to use changed weights"):
        TD.EmbeddingStore.from_dir(str(tmp_path), split="toy")


def test_main_reuses_one_teacher_store(tmp_path, monkeypatch, capsys):
    rows = [{"seq": "ACGUACGU", "structure": "........", "pairs": []},
            {"seq": "GGCAUUAA", "structure": "........", "pairs": []}]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    matrices = np.empty(2, dtype=object)
    matrices[:] = [np.zeros((8, 8)), np.zeros((8, 8))]
    shard = teacher / "shard_00000.npz"
    np.savez(shard, sequences=np.array([row["seq"] for row in rows], dtype=object),
             probs=matrices)
    (teacher / "manifest.json").write_text(json.dumps({
        "teacher": "test", "n_sequences": 2,
        "shards": [{"index": 0, "file": shard.name, "n": 2}],
    }), encoding="utf-8")

    original = TD.TeacherLabelStore.from_dir
    calls = []

    def counted(cls, directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(TD.TeacherLabelStore, "from_dir", classmethod(counted))
    code = TD.main(["--data", str(data), "--teacher-dir", str(teacher),
                    "--lambda-distill", "0", "--steps", "1", "--dry-run"])
    assert code == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
