import json
from pathlib import Path


def test_prior_alignment_contract_is_two_trainings_and_four_identities():
    contract = json.loads((Path(__file__).parents[1] / "configs/prior_alignment_2x2_v1.json").read_text())
    assert contract["status"] == "FROZEN_BEFORE_FULL_MATRIX"
    assert set(contract["training_trajectories"]) == {"train_on", "train_off"}
    arms = {row["arm"]: row for row in contract["evaluation_identities"]}
    assert set(arms) == {"C0", "C1", "C2", "C3"}
    assert arms["C0"]["training_trajectory"] == arms["C1"]["training_trajectory"] == "train_on"
    assert arms["C2"]["training_trajectory"] == arms["C3"]["training_trajectory"] == "train_off"
    assert arms["C0"]["decoder"] == "MAP"
    assert all(arms[name]["decoder"] == "MEA" for name in ("C1", "C2", "C3"))
    assert arms["C0"]["inference_turner"] == arms["C3"]["inference_turner"] == "on"
    assert arms["C1"]["inference_turner"] == arms["C2"]["inference_turner"] == "off"


def test_prior_alignment_contract_locks_objective_budget_and_external_policy():
    root = Path(__file__).parents[1]
    contract = json.loads((root / "configs/prior_alignment_2x2_v1.json").read_text())
    objective = contract["training"]["objective"]
    assert objective["lambda_nll"] == 1.0
    assert objective["nll_normalization"] == "length"
    assert all(objective[name] == 0.0 for name in (
        "lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_pair_recall", "lambda_jepa"))
    assert contract["training"]["backbone_lr"] == 1e-5
    assert contract["training"]["early_stopping"] is False
    assert contract["pilot"]["steps"] == 3200
    assert contract["pilot"]["old_external_panels"] is False
    assert contract["full_matrix_gate"]["n_training_runs"] == 6
    assert contract["full_matrix_gate"]["steps_per_training_trajectory"] == 20000
    assert contract["decode_selection"]["external_panels_may_select_parameters"] is False
    assert contract["decode_selection"]["old_external_panels_may_select_routes"] is False
    assert "no historical Plan-A-trained checkpoint" in contract["shared_initialization"]["source"]


def test_full_matrix_scientific_endpoints_and_checkpoint_policy_are_frozen():
    contract = json.loads((Path(__file__).parents[1] / "configs/prior_alignment_2x2_v1.json").read_text())
    gate = contract["full_matrix_gate"]
    primary = gate["primary_endpoint"]
    secondary = gate["secondary_endpoint"]
    assert gate["additional_pilot_required"] is False
    assert gate["scientific_criteria_frozen_before_launch"] is True
    assert primary["comparison"] == "C2_minus_C1"
    assert primary["macro_pair_f1_mean_delta_min"] == 0.01
    assert primary["micro_pair_f1_mean_delta_min"] == 0.0
    assert primary["micro_pair_precision_mean_delta_min"] == -0.005
    assert primary["macro_pair_precision_mean_delta_min"] == -0.005
    assert "all three" in primary["seed_direction_rule"]
    assert secondary["micro_stem_f1_mean_delta_min"] == -0.005
    assert secondary["macro_stem_f1_mean_delta_min"] == -0.005
    assert gate["checkpoint_trajectory"]["required_retained_steps"] == [3200, 9600, 16800, 20000]
    assert gate["checkpoint_trajectory"]["expected_unfrozen_blocks"] == {
        "3200": 4, "9600": 20, "16800": 33, "20000": 33}
    assert set(gate["required_comparisons"]) >= {"C2_vs_C1", "C2_vs_C0", "C2_vs_C3"}
    assert "only if C2 meets" in contract["external_policy"]["future_use"]
