import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analyze_prior_alignment_unfreeze_trajectory import summarize_rows
from rnajepa.prior_structure_metrics import pair_and_stem_counts
from verify_prior_alignment_optimizer_pilot import verify_stage


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def build_stage(root, status="completed"):
    stage = root / "stage"
    arms = {}
    for arm in ("on", "off"):
        run = stage / f"train_{arm}"
        run.mkdir(parents=True)
        write_json(run / "run_meta.json", {
            "status": status,
            "stop_reason": "step_cap",
            "steps_completed": 9600,
            "schedule": {"backbone_lr": 1e-6, "head_lr": 1e-4,
                         "abort_loss_threshold": 10, "abort_gnorm_threshold": 100},
            "prior_alignment": {"shared_init_sha256": "same"},
        })
        (run / "train_log.jsonl").write_text(json.dumps({"loss": 0.1, "gnorm": 1.0}) + "\n")
        (stage / "snapshots").mkdir(exist_ok=True)
        (stage / f"snapshots/train_{arm}_step9600.pt").touch()
    truth = {(0, 5)}
    rows = []
    for index in range(196):
        for arm in ("C0", "C1", "C2", "C3"):
            row = {"sequence_sha256": str(index), "name": str(index), "arm": arm}
            row.update(pair_and_stem_counts(truth, truth))
            rows.append(row)
    arms = summarize_rows(rows)
    decode = stage / "decode_vl0"
    decode.mkdir()
    write_json(decode / "vl0_selected_summary.json", {"arms": arms})
    (decode / "vl0_selected_per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    write_json(root / "baseline/full_seed0/trajectory/step9600/vl0_selected_summary.json", {"arms": arms})
    return stage


def test_optimizer_verifier_accepts_stable_stage(tmp_path):
    stage = build_stage(tmp_path)
    result = verify_stage(stage, tmp_path / "baseline")
    assert result["stability_pass"]
    assert not result["external_evaluation_allowed"]


def test_optimizer_verifier_rejects_divergence(tmp_path):
    stage = build_stage(tmp_path, status="diverged")
    result = verify_stage(stage, tmp_path / "baseline")
    assert not result["stability_pass"]
    assert not result["checks"]["on_completed_9600"]
