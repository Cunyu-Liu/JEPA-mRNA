#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from structrfm_official_recovery_20261126 import ASSET_NAME, DEFAULT_DATASET, DEFAULT_ROOT, DEFAULT_SOURCE, inspect_checkpoint, load_dataset, normalize_sequence, sha256, utc_now, validate_source, write_json
from run_structrfm_bprna_new_gpu_20261126 import configure_gpu, prepare_models


def load_interface(path):
    spec = importlib.util.spec_from_file_location("interface", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["interface"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Export real official-model DP parameters from 18 GPU-forward readiness sequences; no DP inference or prediction output")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    checkpoint = inspect_checkpoint(args.root / ASSET_NAME)
    source = validate_source(args.source)
    interface_path = args.root / "runtime/cp310/interface.cpython-310-x86_64-linux-gnu.so"
    if sha256(interface_path) != "616ec59fd9c853bebbcb811eeff8af8ec2ce2b7380971dda9de63b9cc33cb6d0":
        raise ValueError("cp310 interface identity mismatch")
    load_interface(interface_path)
    torch, device, architecture = configure_gpu(args.gpu)
    model, tokenizer, pretrained_model = prepare_models(args.source, checkpoint["path"], device, args.root)
    rows = load_dataset(args.dataset)
    selected = sorted(rows, key=lambda row: (len(row["seq"]), row["name"]))[:17]
    longest = max(rows, key=lambda row: len(row["seq"]))
    if longest["name"] not in {row["name"] for row in selected}:
        selected.append(longest)
    output_dir = args.root / "readiness_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    cases = []
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sequence = normalize_sequence(row["seq"])
            input_ids = tokenizer("[CLS]" + sequence)["input_ids"][:512]
            input_ids = [0 if value is None else value for value in input_ids]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            embeddings = pretrained_model(input_tensor)
            parameters = model.make_param((sequence,), embeddings)[0]
            arrays = {}
            for branch in ("turner", "positional"):
                for name, value in parameters[branch].items():
                    arrays[f"{branch}__{name}"] = value.detach().cpu().numpy().copy()
            path = output_dir / f"case_{index:02d}.npz"
            np.savez(path, **arrays)
            cases.append({"case_id": f"case_{index:02d}", "sequence": sequence, "seed": 1000 + index, "constraint": None, "reference": None, "dataset_name": row["name"], "parameter_file": str(path), "parameter_file_sha256": sha256(path)})
            print(f"exported {index + 1}/18 length={len(sequence)}", flush=True)
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_18_REAL_MODEL_GPU_FORWARD_PARAMETER_EXPORTS_NO_DP_PREDICTIONS",
        "checkpoint": checkpoint,
        "source": source,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_architecture": architecture,
        "model_forward_device": str(device),
        "n_cases": len(cases),
        "cases": cases,
        "prediction_artifact": None,
        "metrics": None,
        "full_dataset_inference_performed": False,
    }
    write_json(output_dir / "cases.json", manifest)
    print(json.dumps({"status": manifest["status"], "n_cases": len(cases), "gpu_architecture": architecture, "predictions": None}, indent=2))


if __name__ == "__main__":
    main()
