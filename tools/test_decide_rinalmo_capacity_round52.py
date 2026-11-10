from decide_rinalmo_capacity_round52 import adjudicate


def reports(underfit):
    training = {"status": "PASS_L_O_TRAINING_THREE_CHECK_QA", "allow_capacity_diagnostic": True,
                "runs": {}}
    errors = {}
    representations = {}
    for arm in ("length", "olr"):
        for seed in (42, 2026, 3407):
            key = f"{arm}_s{seed}"
            training["runs"][key] = {"best_vl0": -0.8 if arm == "length" else 22.0,
                                     "max_vl0": 100.0 if arm == "length" else 200.0}
            train_nll = 2.0 if underfit else 0.5
            errors[key] = {"train_mean_losses": {"nll": train_nll, "total": train_nll + 1},
                           "validation_mean_losses": {"nll": 1.0, "total": 1.5}}
            representations[key] = {"train": {"weighted_nll": train_nll},
                                    "validation": {"weighted_nll": 1.0}}
    diagnostics = {"status": "PASS_ALL_L_O_ERROR_AND_REPRESENTATION_PROBES"}
    replication = {"status": "PASS_RINALMO_CAPACITY_INDEPENDENT_CUDA_REPLICATION",
                   "equivalent": True, "arms": ["length_s42", "olr_s42"]}
    return training, diagnostics, replication, errors, representations


result = adjudicate(*reports(False))
assert not result["capacity_insufficiency_supported"]
assert not result["allow_c_wide"] and not result["allow_c_local"]
assert result["allow_j_gate"] and not result["allow_pairjepa_training"]
assert result["next_base"] == "olr"
result = adjudicate(*reports(True))
assert result["capacity_insufficiency_supported"]
assert result["allow_c_wide"] and result["allow_c_local"]
assert not result["allow_j_gate"] and not result["allow_pairjepa_training"]
print("PASS capacity adjudication, no-expansion default, and VL0-only base rule")
