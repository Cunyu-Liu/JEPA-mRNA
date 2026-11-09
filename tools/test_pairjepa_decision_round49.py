import json
import subprocess
from pathlib import Path

import pytest

from tools.adjudicate_pairjepa_round48 import diagnostics, sha256, validate_independent_qa


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


def test_independent_qa_is_required_and_bound_to_summary(tmp_path):
    for suffix in ("", "_greedy", "_turner0"):
        name = "arm" + suffix
        summary = tmp_path / f"eval_decision/{name}/summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"status": "PASS_TEST"}))
        qa = tmp_path / f"audits/round49/{name}_independent_qa.json"
        qa.parent.mkdir(parents=True, exist_ok=True)
        qa.write_text(json.dumps({"status": "PASS_PAIRJEPA_INDEPENDENT_QA", "complete": True,
                                  "experiment": name, "n_cells": 39,
                                  "file_sha256": {f"eval_decision/{name}/summary.json": sha256(summary)}}))
    sources = {}
    validate_independent_qa(tmp_path, "arm", sources)
    assert len(sources) == 3
    (tmp_path / "eval_decision/arm_greedy/summary.json").write_text("tampered")
    with pytest.raises(AssertionError):
        validate_independent_qa(tmp_path, "arm", {})


def test_seed_two_paths_use_new_local_seed_in_all_queue_functions(tmp_path):
    script = Path("tools/run_pairjepa_round48.sh").read_text()
    functions = "train_seed() {" + script.split("train_seed() {", 1)[1].split("readarray -t ARMS", 1)[0]
    (tmp_path / "logs/round48").mkdir(parents=True)
    command = f"""
set -e
seed=1
ROOT={tmp_path}
REPO=/home/rna_jev
PY=/bin/echo
ARM=pairjepa_ema
SCORER=pairjepa
TARGET=ema
RIPE_MODE=r5
RIPE_ENCODER=0
PAIR_LATENT_DIM=490
ABLATION=greedy
SPLITS=(bprna_ts0)
{functions}
train_seed 2
eval_seed 2
ablation_seed 2
"""
    subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    assert (tmp_path / "runs/pairjepa_ema_s2").is_dir()
    train = (tmp_path / "logs/round48/pairjepa_ema_s2.log").read_text()
    assert "--seed 2" in train and "/runs/pairjepa_ema_s2" in train
    for log in ["eval_pairjepa_ema_s2_bprna_ts0.log",
                "eval_pairjepa_ema_greedy_s2_bprna_ts0.log"]:
        assert "/runs/pairjepa_ema_s2/best.pt" in (tmp_path / "logs/round48" / log).read_text()
