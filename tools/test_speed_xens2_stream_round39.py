import json

import pytest

import speed_xens2_stream_round39 as speed


def test_interference_is_recorded_and_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(speed, "compute_pids", lambda: {speed.os.getpid(), 12345})
    with pytest.raises(RuntimeError, match="EXCLUSIVE_SPEED_INTERFERENCE:12345"):
        speed.assert_exclusive()
    out = tmp_path / "speed.json"
    with pytest.raises(SystemExit) as error:
        speed.abort_interference(RuntimeError("EXCLUSIVE_SPEED_INTERFERENCE:12345"), out)
    assert error.value.code == 2
    files = list(tmp_path.glob("speed.interference_*.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert result["status"] == "ABORTED_EXCLUSIVE_SPEED_INTERFERENCE"
    assert not out.exists()


def test_own_process_is_not_interference(monkeypatch):
    monkeypatch.setattr(speed, "compute_pids", lambda: {speed.os.getpid()})
    speed.assert_exclusive()
