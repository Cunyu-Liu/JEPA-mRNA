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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.result) != 2 or len(args.qa) != 2:
        raise ValueError("seed2 primary and seed0 confirmation with QA are required")
    results = [json.loads(Path(path).read_text()) for path in args.result]
    qas = [json.loads(Path(path).read_text()) for path in args.qa]
    assert {raw["seed"] for raw in results} == {0, 2}
    assert len({raw["code_commit"] for raw in results}) == 1
    assert len({raw["code_sha256"] for raw in results}) == 1
    for raw, qa in zip(results, qas):
        assert qa["status"] == "PASS_INDEPENDENT_R_REPLAY_QA"
        assert qa["result_sha256"] == sha256_file(args.result[results.index(raw)])
        assert qa["seed"] == raw["seed"]
        assert qa["code_sha256"] == raw["code_sha256"]
    ratios = {"head": [], "backbone": []}
    shadow_lr_ratios = []
    negative_cosines = []
    for raw in results:
        for batch in raw["batches"]:
            for role in ratios:
                original = next(group for group in batch["shadow_adam"]["sum_original_lr"] if group["role"] == role)
                length = next(group for group in batch["shadow_adam"]["length_original_lr"] if group["role"] == role)
                ratios[role].append(length["update_rms"] / original["update_rms"])
                shadow = next(group for group in batch["shadow_adam"]["sum_shadow_backbone_1e-6"] if group["role"] == role)
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
        "evidence_scope": "best_weights_with_terminal_optimizer_scheduler_and_stream_no_parameter_update",
        "result_sha256": {str(raw["seed"]): sha256_file(args.result[results.index(raw)]) for raw in results},
        "qa_sha256": {str(qa["seed"]): sha256_file(args.qa[qas.index(qa)]) for qa in qas},
        "code_commit": results[0]["code_commit"],
        "code_sha256": results[0]["code_sha256"],
        "length_over_sum_update_rms": {
            role: {"min": min(values), "median": statistics.median(values), "max": max(values), "n": len(values)}
            for role, values in ratios.items()
        },
        "shadow_backbone_1e-6_over_original_update_rms": {
            "min": min(shadow_lr_ratios), "median": statistics.median(shadow_lr_ratios),
            "max": max(shadow_lr_ratios), "n": len(shadow_lr_ratios),
        },
        "negative_nll_aux_cosine_batches": sum(
            item["nll_sum_aux"] < 0 or item["nll_length_aux"] < 0 for item in negative_cosines),
        "total_batches": len(negative_cosines),
        "cosines": negative_cosines,
        "interpretation": [
            "Length normalization does not show uniformly smaller shadow updates at the restored terminal optimizer state; stabilization requires the independent L-length training arm.",
            "Backbone LR 1e-6 reduces the corresponding shadow update magnitude by 10x; this is an optimizer diagnostic, not evidence of better validation or structure metrics.",
            "Negative NLL-auxiliary cosine is batch-dependent and does not justify changing auxiliary weights together with loss or LR.",
            "Full-unfreeze replay alone cannot isolate an unfreezing-ramp effect.",
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
