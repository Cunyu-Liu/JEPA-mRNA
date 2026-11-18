import json
from pathlib import Path


def test_prior_alignment_contract_is_two_trainings_and_four_identities():
    contract = json.loads((Path(__file__).parents[1] / "configs/prior_alignment_2x2_v1.json").read_text())
    assert contract["status"] == "FROZEN_BEFORE_TRAINING"
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
