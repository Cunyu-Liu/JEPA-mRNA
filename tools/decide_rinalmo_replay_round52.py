import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--qa", action="append", required=True)
    parser.add_argument("--training-qa", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 2026))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.result) != len(args.seeds) or len(args.qa) != len(args.seeds):
        raise ValueError("results and independent QA must cover every requested seed")
    results = [json.loads(Path(path).read_text()) for path in args.result]
    qas = [json.loads(Path(path).read_text()) for path in args.qa]
    assert {raw["seed"] for raw in results} == set(args.seeds)
    assert {qa["seed"] for qa in qas} == set(args.seeds)
    assert all(raw["backbone"] == "rinalmo" for raw in results)
    assert len({raw["code_commit"] for raw in results}) == 1
    assert len({raw["code_sha256"] for raw in results}) == 1
    for raw, qa, result_path, qa_path in zip(results, qas, args.result, args.qa):
        assert qa["status"] == "PASS_INDEPENDENT_R_REPLAY_QA"
        assert qa["result_sha256"] == sha256_file(result_path)
        assert qa["seed"] == raw["seed"] and qa["code_sha256"] == raw["code_sha256"]
        assert qa["backbone"] == "rinalmo"
    training_path = Path(args.training_qa)
    training = json.loads(training_path.read_text())
    assert training["status"] == "PASS_R0_TRAINING_THREE_CHECK_QA" and training["allow_r_replay"]
    assert set(training["runs"]) == {"42", "2026", "3407"}
    for raw in results:
        completed = training["runs"][str(raw["seed"])]
        assert completed["n_unfrozen"] == raw["n_unfrozen"]
        assert completed["best_sha256"] == raw["best_sha256"]
        assert completed["resume_sha256"] == raw["resume_sha256"]
    ratios = {"head": [], "backbone": []}
    shadow_lr_ratios = []
    negative_cosines = []
    for raw in results:
        for batch in raw["batches"]:
            for role in ratios:
                original = next(group for group in batch["shadow_adam"]["sum_original_lr"]
                                if group["role"] == role)
                length = next(group for group in batch["shadow_adam"]["length_original_lr"]
                              if group["role"] == role)
                ratios[role].append(length["update_rms"] / original["update_rms"])
                shadow = next(group for group in batch["shadow_adam"]["sum_shadow_backbone_1e-6"]
                              if group["role"] == role)
                if role == "backbone":
                    shadow_lr_ratios.append(shadow["update_rms"] / original["update_rms"])
            checks = batch["gradient_checks"]
            negative_cosines.append({
                "seed": raw["seed"], "queue_offset": batch["queue_offset"],
                "nll_sum_aux": checks["nll_sum_aux_cosine"],
                "nll_length_aux": checks["nll_length_aux_cosine"],
            })
    decision = {
        "status": "PASS_R_REPLAY_THREE_CHECK_QA",
        "experiment": "R-replay",
        "backbone": "rinalmo",
        "evidence_scope": "best_weights_with_terminal_optimizer_scheduler_and_stream_no_parameter_update",
        "training_qa_sha256": sha256_file(training_path),
        "result_sha256": {str(raw["seed"]): sha256_file(path)
                          for raw, path in zip(results, args.result)},
        "qa_sha256": {str(qa["seed"]): sha256_file(path) for qa, path in zip(qas, args.qa)},
        "code_commit": results[0]["code_commit"],
        "code_sha256": results[0]["code_sha256"],
        "n_unfrozen": {str(raw["seed"]): raw["n_unfrozen"] for raw in results},
        "length_over_sum_update_rms": {
            role: {"min": min(values), "median": statistics.median(values),
                   "max": max(values), "n": len(values)} for role, values in ratios.items()},
        "shadow_backbone_1e-6_over_original_update_rms": {
            "min": min(shadow_lr_ratios), "median": statistics.median(shadow_lr_ratios),
            "max": max(shadow_lr_ratios), "n": len(shadow_lr_ratios)},
        "negative_nll_aux_cosine_batches": sum(
            item["nll_sum_aux"] < 0 or item["nll_length_aux"] < 0 for item in negative_cosines),
        "total_batches": len(negative_cosines),
        "cosines": negative_cosines,
        "interpretation": [
            "Length normalization requires an independent L-length training arm; shadow updates alone do not establish stabilization or structure gains.",
            "Backbone LR 1e-6 only diagnoses the corresponding shadow update scale, not validation or F1 improvements.",
            "Negative NLL-auxiliary cosine is batch-dependent and does not justify jointly changing auxiliary weights, loss, or LR.",
            "Replay is limited to each run's actual terminal unfreezing state and cannot by itself isolate an unfreezing-ramp effect.",
        ],
        "allow_l_length_single_factor": True,
        "allow_o_lr_single_factor": True,
        "allow_loss_optimizer_interaction": False,
        "allow_u_ramp": False,
        "allow_capacity_stage": False,
        "scientific_effect_conclusion": False,
        "test_data_used": False,
    }
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
