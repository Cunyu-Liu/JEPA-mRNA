import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools.probe_structured_margin_memory_v1 import (
    BATCH_SIZE, GIB, LENGTH, batch_mask_metadata, probe_gate,
    release_thresholds, select_worst_case_batch,
)


def record(index, length=LENGTH, pairs=10):
    sequence = "A" * length
    return {
        "id": f"record-{index}", "sequence": sequence,
        "sequence_sha256": f"hash-{index}",
        "pairs": [(i, i + 4) for i in range(pairs)],
    }


def test_release_thresholds_use_gpu_reserve_and_conservative_dual_scaling():
    peak = 8 * GIB
    total = 32 * GIB
    result = release_thresholds(peak, total)
    reserve = math.ceil(max(4 * GIB, 0.20 * total))
    assert result["reserve_bytes"] == reserve
    assert result["single_arm"]["required_free_bytes"] == math.ceil(1.5 * peak + reserve)
    assert result["dual_arm"]["required_free_bytes"] == math.ceil(1.5 * 2 * peak + reserve)
    assert result["dual_arm"]["required_free_bytes"] > result["single_arm"]["required_free_bytes"]
    with pytest.raises(ValueError):
        release_thresholds(0, total)


def test_gate_queues_instead_of_preempting_when_memory_is_insufficient():
    allowed = probe_gate(24 * 1024, 24)
    queued = probe_gate(24 * 1024 - 1, 24)
    assert allowed["status"] == "RUN_ALLOWED" and allowed["sufficient"]
    assert queued["status"] == "QUEUED_INSUFFICIENT_FREE_MEMORY" and not queued["sufficient"]
    assert "never kill or preempt" in queued["policy"]


def test_worst_case_batch_is_real_length_500_and_deterministic():
    records = [record(index, length=499 if index == 0 else LENGTH, pairs=index) for index in range(6)]
    selected = select_worst_case_batch(records)
    assert [row["id"] for row in selected] == ["record-5", "record-4", "record-3", "record-2"]
    assert len(selected) == BATCH_SIZE
    assert all(len(row["sequence"]) == LENGTH for row in selected)
    with pytest.raises(ValueError, match="need four real"):
        select_worst_case_batch(records[:2])


def test_batch_metadata_binds_real_canonical_masks_and_targets():
    row = record(1, length=12, pairs=1)
    row["sequence"] = "AACCUCCUCCCC"
    row["pairs"] = [(0, 4), (1, 7)]
    metadata = batch_mask_metadata([row])[0]
    assert metadata["record_id"] == row["id"]
    assert metadata["target_pair_count"] == 2
    assert metadata["canonical_pair_count"] == 6
    assert len(metadata["canonical_mask_sha256"]) == 64
    assert len(metadata["target_pairs_sha256"]) == 64
    bad = dict(row, pairs=[(0, 5)])
    with pytest.raises(ValueError, match="non-canonical or illegal"):
        batch_mask_metadata([bad])


def test_launcher_is_bounded_and_cannot_launch_training_or_decoder_search():
    launcher = (REPO / "tools/run_structured_margin_memory_probe_v1.sh").read_text()
    probe = (REPO / "tools/probe_structured_margin_memory_v1.py").read_text()
    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "--device cuda:0" in launcher
    assert "--min-free-gib 24" in launcher
    assert "train_plan_a_gec_convergence_v1.py" not in launcher
    assert "optimizer.step" not in probe
    assert "restore_unfrozen_blocks(blocks, TERMINAL_UNFROZEN_BLOCKS)" in probe
    assert "pair_hamming_structured_margin" in probe
    assert "compute_gec_context_v1" in probe
    assert "harness.valid_pair_mask" in probe
    assert "loss.backward()" in probe
    assert '"index": device.index' in probe
    assert "properties.index" not in probe
    assert "torch.float64" in probe
    assert "seed 1" not in launcher and "seed 2" not in launcher
    assert "temperature" not in launcher and "gamma" not in launcher and "threshold" not in launcher
