import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_objects(path):
    decoder = json.JSONDecoder()
    text = Path(path).read_text()
    values = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        values.append(value)
    return values


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    run = root / "runs/round52_rinalmo_r0_smoke_s42"
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["arm"] == "round52_rinalmo_r0_smoke_s42" and meta["seed"] == 42
    assert meta["status"] == "completed" and meta["steps_completed"] == 2
    assert meta["stop_reason"] == "step_cap" and not meta["converged"]
    assert not meta["early_stopped"] and meta["schedule"]["steps"] == 2
    assert meta["schedule"]["batch_size"] == 2
    assert meta["schedule"]["head_lr"] == 1e-4 and meta["schedule"]["backbone_lr"] == 1e-5
    assert meta["validation"]["every"] == 2 and meta["validation"]["patience"] == 6
    assert meta["objective"]["nll_normalization"] == "sum"
    assert all(meta["objective"][key] == 1.0 for key in
               ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    assert meta["scorer"] == "pairmlp" and meta["pair_latent_dim"] == 490
    assert meta["ripe"] == {"mode": "r6", "encoder": False, "encoder_mode": None}
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    report = meta["backbone_report"]
    assert report["weights"] == "/root/rna_jepa_data/weights/rinalmo-giga"
    assert report["d_model"] == 1280 and report["n_layers"] == 33
    assert report["loaded_fraction_of_encoder"] == 0.996
    assert sorted(report["expected_absent"]) == ["pooler.dense.bias", "pooler.dense.weight"]
    train = read_jsonl(run / "train_log.jsonl")
    validation = read_jsonl(run / "validation_log.jsonl")
    assert train and all(math.isfinite(row["loss"]) and math.isfinite(row["gnorm"])
                         for row in train)
    assert all(all(math.isfinite(value) for value in row["terms"].values()) for row in train)
    assert len(validation) == 1 and validation[0]["step"] == 2
    assert validation[0]["validation_loss"] == meta["best_validation_loss"]
    assert math.isfinite(meta["best_validation_loss"])
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    resume = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
    assert best["step"] == resume["step"] == 2
    assert best["validation_loss"] == meta["best_validation_loss"]
    for part in ("encoder", "head"):
        assert list(best["model"][part]) == list(resume[part])
        assert all(torch.equal(best["model"][part][key], resume[part][key])
                   for key in best["model"][part])
    assert resume["n_unfrozen"] == 0
    assert resume["scheduler"]["base_lrs"] == [1e-4, 1e-5]
    assert resume["scheduler"]["last_epoch"] == 2
    assert resume["scheduler"]["_last_lr"] == [6e-5, 6e-6]
    assert [group["lr"] for group in resume["optimizer"]["param_groups"]] == [6e-5, 6e-6]
    assert resume["optimizer"]["state"]
    assert resume["stream_state"]
    preflight = json_objects(root / "audits/round52/rinalmo_preflight.json")
    assert len(preflight) == 2
    card = preflight[1]
    assert card["model_card_check"] == "PARTIAL_TOP1_MISMATCH"
    assert not card["matches_model_card_top1"] and not card["unexpected"]
    assert card["model_card_expected_top1"] == "G" and card["top5"][0][0] == "A"
    assert {item[0] for item in card["top5"]} == {"A", "C", "G", "U", "X"}
    weights = Path("/root/rna_jepa_data/weights/rinalmo-giga")
    output = {
        "status": "PASS_R0_SMOKE_THREE_CHECK_QA",
        "scientific_conclusion": False,
        "run": meta["arm"],
        "best_step": meta["best_step"],
        "best_vl0": meta["best_validation_loss"],
        "train": train,
        "validation": validation,
        "resume_n_unfrozen": resume["n_unfrozen"],
        "scheduler_last_lr": resume["scheduler"]["_last_lr"],
        "best_resume_weights_equal": True,
        "model_card_top1": preflight[1]["top5"][0],
        "backbone_report": report,
        "hashes": {
            "best": sha256_file(run / "best.pt"),
            "resume": sha256_file(run / "resume.pt"),
            "run_meta": sha256_file(run / "run_meta.json"),
            "train_log": sha256_file(run / "train_log.jsonl"),
            "validation_log": sha256_file(run / "validation_log.jsonl"),
            "train_code": sha256_file(Path(args.repo) / "tools/train_plan_a.py"),
            "weights": sha256_file(weights / "model.safetensors"),
            "weights_config": sha256_file(weights / "config.json"),
            "train_data": sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"),
            "validation_data": sha256_file(root / "data/validation/vl0_196.jsonl"),
            "teacher_manifest": sha256_file(
                "/root/rna_jepa_data/ss_data/teacher/bprna_tr1c/manifest.json"),
            "validation_teacher_manifest": sha256_file(root / "data/teacher_vl0/manifest.json"),
        },
        "limitations": [
            "Two steps validate engineering only, not convergence, capacity, architecture efficacy, or SOTA.",
            "R0 formal training remains steps=0 with 500-step VL0 and patience=6.",
            "No structRFM checkpoint, optimizer, result, or test-set selection is used.",
            "The official MLM head loads with no unexpected keys, but model-card top1 is A rather than G; retain PARTIAL_TOP1_MISMATCH and do not claim exact model-card reproduction.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "best_vl0": output["best_vl0"],
                      "hashes": output["hashes"]}, indent=2))


if __name__ == "__main__":
    main()
