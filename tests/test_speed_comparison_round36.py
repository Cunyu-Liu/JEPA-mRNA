import json
from pathlib import Path

from tools.build_round36_speed_comparison import main


def result(model, batch, throughput, status="PASS", agreement=None, delta=None):
    equivalence = None if agreement is None else {
        "pair_agreement": agreement,
        "abs_delta_f1": delta,
    }
    return {
        "status": status,
        "model": model,
        "protocol": {
            "batch_statistics": {
                "nominal_batch_size": batch,
                "mean_actual_batch_size": min(batch, 2.0),
            },
            "r2d_encoder_autocast_bf16": False,
        },
        "throughput_median_seq_per_s": throughput,
        "runs": [{"strict_micro_f1": 0.8, "peak_memory_mib": 1000.0}],
        "f1_runs": [0.8],
        "cross_config_equivalence": equivalence,
    }


def test_only_eligible_candidate_can_win(tmp_path):
    profile = tmp_path / "profile"
    optimized = tmp_path / "optimized"
    baselines = tmp_path / "baselines"
    for directory in [profile, optimized, baselines]:
        directory.mkdir()
    (profile / "r2d_b1_profile.json").write_text(json.dumps(result("r2d", 1, 20.0)))
    (optimized / "r2d_b4_profile.json").write_text(json.dumps(result("r2d", 4, 30.0, agreement=0.995, delta=0.0005)))
    (optimized / "r2d_b16_profile.json").write_text(json.dumps(result("r2d", 16, 99.0, agreement=0.98, delta=0.0005)))
    out = tmp_path / "summary.json"
    import sys
    sys.argv = ["build", "--profile-dir", str(profile), "--optimized-dir", str(optimized),
                "--baseline-dir", str(baselines), "--out", str(out)]
    main()
    payload = json.loads(out.read_text())
    assert payload["best_eligible_by_model"]["r2d"]["throughput_median_seq_per_s"] == 30.0
    assert payload["speed_sota_claim"] == "not_assessed_pending_complete_same_protocol_baselines"
    assert len(payload["optimization_candidates"]) == 2
