#!/usr/bin/env python3
import json
import time
from pathlib import Path

from structrfm_official_recovery_20261126 import DEFAULT_ROOT, normalize_sequence, sha256, utc_now, validate_prediction_rows, write_json
from run_structrfm_bprna_new_gpu_20261126 import configure_gpu, preflight, prepare_models


def main():
    root = DEFAULT_ROOT
    checked = preflight(require_runtime=True)
    torch, device, architecture = configure_gpu("0")
    model, tokenizer, pretrained_model = prepare_models(checked["source"]["path"], checked["checkpoint"]["path"], device, root)
    row = max(checked["dataset_rows"], key=lambda value: len(value["seq"]))
    sequence = normalize_sequence(row["seq"])
    input_ids = tokenizer("[CLS]" + sequence)["input_ids"][:512]
    input_ids = [0 if value is None else value for value in input_ids]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        embeddings = pretrained_model(input_tensor)
        parameters = model.make_param((sequence,), embeddings)[0]
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        param_on_cpu = {
            "turner": {key: value.to("cpu") for key, value in parameters["turner"].items()},
            "positional": {key: value.to("cpu") for key, value in parameters["positional"].items()},
        }
        param_on_cpu = {key: model.clear_count(value) for key, value in param_on_cpu.items()}
        dp_started = time.perf_counter()
        import interface
        _, _, connects = interface.predict_mxfold(
            sequence,
            param_on_cpu,
            max_internal_length=30,
            max_helix_length=model.max_helix_length,
            constraint=None,
            reference=None,
            loss_pos_paired=0.0,
            loss_neg_paired=0.0,
            loss_pos_unpaired=0.0,
            loss_neg_unpaired=0.0,
        )
        dp_seconds = time.perf_counter() - dp_started
    validate_prediction_rows([row], [{"name": row["name"], "seq": sequence, "connects": connects[1:]}])
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_LONGEST_489_B1_CUDA_FORWARD_CPU_DP_PROBE",
        "sequence_name": row["name"],
        "sequence_length": len(sequence),
        "batch_size": 1,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_architecture": architecture,
        "model_forward_device": str(device),
        "dp_device": "cpu",
        "cuda_forward_seconds": forward_seconds,
        "cpu_dp_seconds": dp_seconds,
        "total_probe_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "peak_memory_allocated_mib": peak_allocated / (1 << 20),
        "peak_memory_reserved_mib": peak_reserved / (1 << 20),
        "gate_peak_mib": max(peak_allocated, peak_reserved) / (1 << 20),
        "checkpoint_sha256": checked["checkpoint"]["sha256"],
        "input_dataset_sha256": sha256(checked["dataset_path"]),
        "existing_usage_record": str(root / "reports/preprobe_existing_gpu_usage_20261126.txt"),
        "prediction_artifact": None,
        "full_dataset_inference_performed": False,
    }
    write_json(root / "reports/longest_489_b1_probe_20261126.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
