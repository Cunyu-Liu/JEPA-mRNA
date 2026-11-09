import json

import pytest

from tools import parallel_teacher_labels


class FailedProcess:
    def __init__(self, command, stdout, stderr):
        self.command = command

    def wait(self):
        return 1


def test_worker_receives_max_length(tmp_path, monkeypatch):
    commands = []

    def popen(command, stdout, stderr, env):
        commands.append(command)
        return FailedProcess(command, stdout, stderr)

    monkeypatch.setattr(parallel_teacher_labels.subprocess, "Popen", popen)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"seq": "A" * 601}) + "\n")
    with pytest.raises(SystemExit):
        parallel_teacher_labels.run(str(data), str(tmp_path / "teacher"), workers=1,
                                    shard_size=1, max_length=1024)
    assert "--max-length" in commands[0]
    assert commands[0][commands[0].index("--max-length") + 1] == "1024"


def test_merge_rejects_missing_teacher_sequences(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"seq": seq}) + "\n" for seq in ["AAAA", "CCCC"]))
    out = tmp_path / "teacher"
    part = out / "part_00"
    part.mkdir(parents=True)
    (part / "shard_00000.npz").write_bytes(b"placeholder")
    (part / "manifest.json").write_text(json.dumps({
        "teacher": "thermo:viennarna", "teacher_version_lock": "ViennaRNA 2.7.2",
        "shards": [{"file": "shard_00000.npz", "n": 1, "sha256": "x", "seq_sha256": "y"}]}))
    with pytest.raises(RuntimeError, match="teacher sequence conservation"):
        parallel_teacher_labels.run(str(data), str(out), workers=1, shard_size=1,
                                    max_length=0, merge_only=True)
