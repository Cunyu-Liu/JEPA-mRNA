import json

import pytest

from tools.adjudicate_pairjepa_round48 import diagnostics


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(loss=0.1, rank=2.0):
    return {"masked_pairs": 3, "loss": loss, "target_variance": 0.2,
            "prediction_variance": 0.1, "target_effective_rank": rank,
            "prediction_effective_rank": rank}


def test_jepa_diagnostics_require_finite_nonempty_values(tmp_path):
    write_rows(tmp_path / "jepa_diagnostics.jsonl", [row(), row(loss=0.2, rank=1.5)])
    result = diagnostics(tmp_path, 0)
    assert result["n_records"] == 2 and result["minimum_prediction_effective_rank"] == 1.5
    write_rows(tmp_path / "jepa_diagnostics.jsonl", [row(rank=-1)])
    with pytest.raises(AssertionError):
        diagnostics(tmp_path, 0)
    write_rows(tmp_path / "jepa_diagnostics.jsonl", [])
    with pytest.raises(AssertionError):
        diagnostics(tmp_path, 0)
