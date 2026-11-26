#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from structrfm_official_recovery_20261126 import (  # noqa: E402
    ASSET_NAME,
    DEFAULT_DATASET,
    DEFAULT_ROOT,
    DEFAULT_SOURCE,
    INFERENCE_COMMIT,
    MODEL_CONFIG,
    inspect_checkpoint,
    load_dataset,
    normalize_sequence,
    sha256,
    utc_now,
    validate_prediction_rows,
    validate_source,
    write_json,
)


class PreflightError(RuntimeError):
    pass


def inside_root(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise PreflightError(f"Asset or output must remain under {root}: {path}") from error


def runtime_preflight(source=DEFAULT_SOURCE):
    if sys.version_info[:2] != (3, 8) or platform.machine() != "x86_64":
        raise PreflightError(
            "Official DP runtime requires CPython 3.8 x86_64 for "
            "interface.cpython-38-x86_64-linux-gnu.so; current interpreter is "
            f"{platform.python_version()} {platform.machine()}"
        )
    source = Path(source)
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "tasks/seqcls_ssp"))
    import transformers.data.data_collator as collator
    if not hasattr(collator, "pad_without_fast_tokenizer_warning"):
        def pad_without_fast_tokenizer_warning(features, tokenizer, **kwargs):
            from transformers.data.data_collator import DataCollatorWithPadding
            keep = {key: value for key, value in kwargs.items() if key in {"padding", "max_length", "pad_to_multiple_of", "return_tensors"}}
            return DataCollatorWithPadding(tokenizer, **keep)(features)
        collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning
    try:
        modules = {name: importlib.import_module(name) for name in ("torch", "numpy", "transformers", "structRFM.model", "structRFM.data", "ss_pred", "param_turner2004")}
    except (ImportError, AttributeError) as error:
        raise PreflightError(f"Official runtime import failed: {type(error).__name__}: {error}") from error
    return {
        "python": platform.python_version(),
        "platform": platform.machine(),
        "torch": modules["torch"].__version__,
        "numpy": modules["numpy"].__version__,
        "transformers": modules["transformers"].__version__,
        "official_dp": "interface.predict_mxfold",
        "source_commit": INFERENCE_COMMIT,
    }


def preflight(root=DEFAULT_ROOT, dataset_path=DEFAULT_DATASET, source=DEFAULT_SOURCE, require_runtime=True):
    root = Path(root)
    rows = load_dataset(dataset_path)
    if max(len(row["seq"]) for row in rows) > 511:
        raise PreflightError("Rows over 511 nt require a separately verified official long-sequence protocol")
    try:
        checkpoint = inspect_checkpoint(root / ASSET_NAME)
        source_identity = validate_source(source)
        runtime = runtime_preflight(source) if require_runtime else None
    except ValueError as error:
        raise PreflightError(str(error)) from error
    return {"root": root, "dataset_path": Path(dataset_path), "dataset_rows": rows, "checkpoint": checkpoint, "source": source_identity, "runtime": runtime}


def prepare_models(source, checkpoint_path, device):
    source = Path(source)
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "tasks/seqcls_ssp"))
    import torch
    import transformers.data.data_collator as collator
    if not hasattr(collator, "pad_without_fast_tokenizer_warning"):
        def pad_without_fast_tokenizer_warning(features, tokenizer, **kwargs):
            from transformers.data.data_collator import DataCollatorWithPadding
            keep = {key: value for key, value in kwargs.items() if key in {"padding", "max_length", "pad_to_multiple_of", "return_tensors"}}
            return DataCollatorWithPadding(tokenizer, **keep)(features)
        collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning
    from structRFM.model import get_structRFM, get_model_scale
    from structRFM.data import get_mlm_tokenizer
    from ss_pred import structRFMForSsp, MixedFold
    import param_turner2004

    tokenizer = get_mlm_tokenizer(max_length=514)
    encoder = get_structRFM(from_pretrained=None, output_hidden_states=True, tokenizer=tokenizer, **get_model_scale("base"))
    pretrained_model = structRFMForSsp(encoder)
    model = MixedFold(init_param=param_turner2004, **MODEL_CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    pretrained_model.load_state_dict(checkpoint["pretrained_model"], strict=True)
    model.to(device).eval()
    pretrained_model.to(device).eval()
    return model, tokenizer, pretrained_model


def execute(checked, output_path, gpu="0"):
    if checked.get("runtime") is None:
        raise PreflightError("Runtime preflight is mandatory before execution")
    output_path = Path(output_path)
    inside_root(output_path, checked["root"])
    if output_path.exists():
        raise PreflightError(f"Refusing to overwrite predictions: {output_path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTHONHASHSEED"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is required for --execute; no CPU inference fallback is permitted")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model, tokenizer, pretrained_model = prepare_models(checked["source"]["path"], checked["checkpoint"]["path"], device)
    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(checked["dataset_rows"]):
            sequence = normalize_sequence(row["seq"])
            input_ids = tokenizer("[CLS]" + sequence)["input_ids"][:512]
            input_ids = [0 if value is None else value for value in input_ids]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            embeddings = pretrained_model(input_tensor)
            _, _, structures = model((sequence,), embeddings)
            predictions.append({"name": row["name"], "seq": sequence, "connects": [int(value) for value in structures[0][1:]]})
            if (index + 1) % 200 == 0:
                print(f"predicted {index + 1}/{len(checked['dataset_rows'])}", flush=True)
    validated = validate_prediction_rows(checked["dataset_rows"], predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in validated:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output_path)
    reloaded = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    validate_prediction_rows(checked["dataset_rows"], reloaded)
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_COMPLETE_OFFICIAL_5388_PREDICTIONS",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "checkpoint": checked["checkpoint"],
        "source": checked["source"],
        "runtime": checked["runtime"],
        "cuda_visible_devices": str(gpu),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {"path": str(checked["dataset_path"]), "sha256": sha256(checked["dataset_path"]), "n_sequences": 5388, "labels_excluded_from_model_input": True},
        "model_form": "single_checkpoint_not_ensemble",
        "predictions": {"path": str(output_path), "sha256": sha256(output_path), "n_sequences": len(reloaded), "coverage": 1.0},
        "external_test_used_for_training_or_tuning": False,
    }
    write_json(output_path.with_suffix(".run_manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Official structRFM v0.0.8 bpRNA-new 5388 GPU launcher; default is CPU-only preflight")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true", help="Run GPU inference only after every preflight passes")
    args = parser.parse_args()
    try:
        checked = preflight(args.root, args.dataset, args.source, require_runtime=True)
        if not args.execute:
            print(json.dumps({"status": "PREFLIGHT_PASS_GPU_NOT_EXECUTED", "n_sequences": 5388, "checkpoint_sha256": checked["checkpoint"]["sha256"], "runtime": checked["runtime"]}, indent=2))
            return 0
        result = execute(checked, args.output, args.gpu)
        print(json.dumps(result, indent=2))
        return 0
    except (PreflightError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_PREFLIGHT", "gpu_executed": False, "reason": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
