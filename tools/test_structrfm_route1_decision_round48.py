import hashlib
import json

from tools.adjudicate_structrfm_route1_round48 import ARMS, build_decision


def test_route1_decision_requires_complete_sources_and_uses_vl0_only(tmp_path):
    for arm_index, (arm, (experiment, prefix)) in enumerate(ARMS.items()):
        base = tmp_path / "eval_decision" / experiment
        base.mkdir(parents=True)
        summary = {"status": "PASS_TEST", "splits": {
            "archiveii": {"n_long_unsupported": 3},
            "rfam_fam": {"n_long_unsupported": 5}}}
        (base / "summary.json").write_text(json.dumps(summary))
        sota = {"status": "PASS_STRUCTRFM_INDEPENDENT_COMPARISON",
                "result_sha256": hashlib.sha256((base / "summary.json").read_bytes()).hexdigest(),
                "main_sota_any_wins": 0, "main_sota_external_single_wins": 0,
                "rfam": {"ours_f1_mean": 0.5, "sota": False},
                "chanrg_measured_wins": 0}
        (base / "sota_comparison.json").write_text(json.dumps(sota))
        confidence = {"splits": {str(index): {} for index in range(13)}}
        (base / "confidence_summary.json").write_text(json.dumps(confidence))
        for seed in range(3):
            run = tmp_path / f"runs/{prefix}{seed}"
            run.mkdir(parents=True)
            meta = {"status": "completed", "converged": True, "early_stopped": True,
                    "best_validation_loss": arm_index + seed + 0.5}
            (run / "run_meta.json").write_text(json.dumps(meta))
        for ablation in ("greedy", "turner0"):
            path = tmp_path / f"eval_decision/{experiment}_{ablation}"
            path.mkdir()
            (path / "summary.json").write_text(json.dumps({"status": "PASS_TEST"}))
    paired = {"status": "PASS_STRUCTRFM_S1_S2_S3_PAIRED_MICRO_F1_BOOTSTRAP",
              "comparisons": {"S2_minus_S1": [], "S3_minus_S1": [], "S3_minus_S2": []}}
    (tmp_path / "eval_decision/structrfm_paired_bootstrap.json").write_text(json.dumps(paired))
    decision = build_decision(tmp_path)
    assert decision["complete"] and decision["allow_ripe"]
    assert decision["status"] == "PASS_ROUTE1_ADJUDICATION_WITH_NEGATIVE_SOTA"
    assert decision["vl0_reference"] == {"arm": "S1", "seed": 0,
                                         "best_validation_loss": 0.5,
                                         "selection": "minimum best VL0 loss only; not test-selected"}
    assert not decision["long_sequence_sota_claim"]
    assert not decision["best_confidence_claim"]
    assert not decision["exclusive_speed_sota_claim"]
    assert decision["arms"]["S3"]["unsupported"] == {"archiveii": 3, "rfam_fam": 5}
