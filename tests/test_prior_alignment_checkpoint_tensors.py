import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from verify_prior_alignment_checkpoint_tensors import verify_checkpoint


def _make_checkpoint(tmp_path, step, unfrozen, tamper=False):
    stage = tmp_path / "full_seed0"
    run = stage / "train_off"
    snapshots = stage / "snapshots"
    run.mkdir(parents=True)
    snapshots.mkdir()
    run_meta = {"seed": 0, "status": "completed" if step == 20000 else "running",
                "early_stopped": False, "schedule": {"steps": 20000}}
    (run / "run_meta.json").write_text(json.dumps(run_meta))
    (run / "train_log.jsonl").write_text(json.dumps({"step": step, "unfrozen": unfrozen}) + "\n")
    encoder = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    backbone_hash = tensor_state_sha256(encoder)
    payload = {"step": step, "model": {"encoder": encoder},
               "meta": {"seed": 0, "prior_alignment": {"contract_sha256": "contract-test"}},
               "prior_alignment_checkpoint": {"step": step, "actual_unfrozen_blocks": unfrozen,
                                               "backbone_update_sha256": backbone_hash}}
    checkpoint = snapshots / f"train_off_step{step}.pt"
    torch.save(payload, checkpoint)
    receipt = {"step": step, "actual_unfrozen_blocks": unfrozen,
               "backbone_update_sha256": "0" * 64 if tamper else backbone_hash,
               "checkpoint_sha256": sha256_file(checkpoint)}
    Path(str(checkpoint) + ".json").write_text(json.dumps(receipt))
    return stage.parent


@pytest.mark.parametrize("step,unfrozen", [(3200, 4), (20000, 33)])
def test_checkpoint_tensor_verification_reloads_and_recomputes(tmp_path, step, unfrozen):
    root = _make_checkpoint(tmp_path, step, unfrozen)
    result = verify_checkpoint(root, 0, "train_off", step, unfrozen, "contract-test")
    assert result["status"] == "PASS"
    assert result["checks"]["embedded_backbone_sha256"] is True
    if step == 20000:
        assert result["stopping_reason"] == "FIXED_20000_STEP_BUDGET_REACHED_NO_EARLY_STOP"


def test_checkpoint_tensor_verification_rejects_tampered_receipt(tmp_path):
    root = _make_checkpoint(tmp_path, 3200, 4, tamper=True)
    with pytest.raises(ValueError, match="receipt_backbone_sha256"):
        verify_checkpoint(root, 0, "train_off", 3200, 4, "contract-test")
