import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
p0 = json.loads((ROOT / "artifacts/p2/pilot_v1/H0/model_dev_eval.json").read_text())
results = {"P0_no_motif": p0["pair_level"]["macro"]["f1"]}
for arm in ("inner_outer", "shuffled", "turner0", "energy_only"):
    result = json.loads((ROOT / f"artifacts/v2/bpfold/{arm}/model_dev_eval.json").read_text())
    assert result["data_role"] == "model-dev"
    assert not result["structure_labels_used_for_inference"]
    results[arm] = result["pair_level"]["macro"]["f1"]
passed = results["inner_outer"] > max(results["P0_no_motif"], results["shuffled"])
report = {
    "status": "PASS_BPFOLD_DEV_ONLY" if passed else "STOP_BPFOLD_DEV_NEGATIVE",
    "p1_passed": passed,
    "external_evaluation_started": False,
    "scientific_scope": "single-seed model-dev only; no external accuracy or BRIQ-generalization claim",
    "results_macro_f1": results,
    "p1_minus_p0": results["inner_outer"] - results["P0_no_motif"],
    "p1_minus_shuffled": results["inner_outer"] - results["shuffled"],
    "next_rule": "Only P1 exceeding both P0 and shuffled can proceed to frozen external evaluation.",
}
out = ROOT / "manifests/v2/bpfold_decision.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
