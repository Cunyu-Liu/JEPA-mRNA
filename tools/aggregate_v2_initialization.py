import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
report = {"status": "REPORT_INITIALIZATION_CONFOUNDING_ONLY", "scientific_scope": "model-dev only; no external accuracy claim", "arms": {}}
for arm in ("H0", "H2"):
    warm = json.loads((ROOT / f"artifacts/p2/pilot_v1/{arm}/model_dev_eval.json").read_text())
    native = json.loads((ROOT / f"artifacts/v2/initialization/native_{arm}/model_dev_eval.json").read_text())
    assert warm["data_role"] == native["data_role"] == "model-dev"
    report["arms"][arm] = {
        "warm_plan_a_macro_f1": warm["pair_level"]["macro"]["f1"],
        "native_rinalmo_macro_f1": native["pair_level"]["macro"]["f1"],
        "native_minus_warm": native["pair_level"]["macro"]["f1"] - warm["pair_level"]["macro"]["f1"],
        "warm_checkpoint_sha256": warm["checkpoint_sha256"],
        "native_checkpoint_sha256": native["checkpoint_sha256"],
        "native_step": native["checkpoint_step"],
    }
report["native_h2_minus_native_h0"] = report["arms"]["H2"]["native_rinalmo_macro_f1"] - report["arms"]["H0"]["native_rinalmo_macro_f1"]
report["optimizer_continuation"] = "BLOCKED_ORIGINAL_RESUME_CORRUPT; only fresh optimizer states are compared"
report["interpretation"] = "Initialization is a confound, not a mechanism result. Native undertraining and warm-start forgetting must not be used as dual-view evidence."
out = ROOT / "manifests/v2/initialization_report.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
