import json
import sys

import numpy as np

from tools.analyze_prism_confidence_round42 import risk_curve
from tools.analyze_structrfm_confidence_round48 import COVERAGE, evaluate, main


def test_risk_curve_matches_registered_protocol():
    rows = [{"structure_conf": value, "f1": f1}
            for value, f1 in zip([0.1, 0.4, 0.5, 0.7, 0.9], [0.0, 0.2, 0.5, 0.8, 1.0])]
    thresholds = np.quantile([row["structure_conf"] for row in rows], 1 - COVERAGE)
    actual = evaluate(rows, thresholds)
    expected_risk, expected_aurc = risk_curve(
        np.array([row["structure_conf"] for row in rows]),
        np.array([row["f1"] for row in rows]), thresholds)
    np.testing.assert_allclose(actual["risk"], expected_risk, rtol=0, atol=0)
    assert actual["aurc"] == expected_aurc
    np.testing.assert_allclose(actual["spearman"], 1.0, rtol=0, atol=1e-12)


def test_reference_seed_and_thresholds_come_only_from_vl0(tmp_path):
    experiment = "structrfm_frozen"
    output = tmp_path / f"eval_decision/{experiment}"
    output.mkdir(parents=True)
    for seed in range(3):
        run = tmp_path / f"runs/structrfm_frozen_jev_s{seed}"
        run.mkdir(parents=True)
        (run / "run_meta.json").write_text(json.dumps({"best_validation_loss": [3.0, 2.0, 4.0][seed]}))
        rows = [{"structure_conf": (seed + 1) * value, "f1": value}
                for value in [0.1, 0.2, 0.3, 0.4, 0.5]]
        (output / f"vl0_s{seed}.json").write_text(json.dumps({"repeats": 3, "per_sequence": rows}))
    sys.argv = ["confidence", "--root", str(tmp_path), "--vl0-only"]
    main()
    summary = json.loads((output / "confidence_summary.json").read_text())
    thresholds = json.loads((output / "confidence_vl0_thresholds.json").read_text())
    assert thresholds["reference_seed"] == 1
    expected = np.quantile([0.2, 0.4, 0.6, 0.8, 1.0], 1 - COVERAGE)
    np.testing.assert_allclose(thresholds["thresholds"], expected, rtol=0, atol=0)
    assert summary["splits"] == {}
    assert set(summary["vl0"]) == {"s0", "s1", "s2"}
