#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata
import importlib.util
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


INTERFACE_FILENAME = "interface.cpython-310-x86_64-linux-gnu.so"
INTERFACE_SHA256 = "616ec59fd9c853bebbcb811eeff8af8ec2ce2b7380971dda9de63b9cc33cb6d0"
OFFICIAL_INTERFACE_SHA256 = "7d332a297216ecc0d5ccbeec356c42876b543d089cba8846f478ebbaec4ea007"


def load_verified_interface(root):
    root = Path(root)
    expected_name = f"interface.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
    if sys.implementation.name != "cpython" or platform.machine() != "x86_64" or INTERFACE_FILENAME != expected_name:
        raise PreflightError(f"No verified interface for {sys.implementation.name} {platform.python_version()} {platform.machine()}")
    path = root / "runtime/cp310" / INTERFACE_FILENAME
    inside_root(path, root)
    if not path.is_file() or sha256(path) != INTERFACE_SHA256:
        raise PreflightError(f"Verified cp310 interface missing or SHA256 mismatch: {path}")
    report_path = root / "reports/differential/differential_report.json"
    if not report_path.is_file():
        raise PreflightError(f"Missing official cp38 differential report: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS_OFFICIAL_CP38_VS_REBUILT_CP310" or report.get("n_cases") != 18:
        raise PreflightError("Official cp38 differential report is incomplete")
    if report.get("reference_interface_sha256") != OFFICIAL_INTERFACE_SHA256 or report.get("candidate_interface_sha256") != INTERFACE_SHA256:
        raise PreflightError("Differential report interface identity mismatch")
    spec = importlib.util.spec_from_file_location("interface", path)
    if spec is None or spec.loader is None:
        raise PreflightError(f"Cannot load verified interface: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interface"] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve() or not hasattr(module, "predict_mxfold"):
        raise PreflightError("Loaded interface identity mismatch")
    return {"path": str(path), "sha256": INTERFACE_SHA256, "differential_report": str(report_path), "differential_cases": 18}


def runtime_preflight(source=DEFAULT_SOURCE, root=DEFAULT_ROOT):
    interface = load_verified_interface(root)
    source = Path(source)
    sys.path.insert(0, str(Path(root) / "pydeps"))
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
        modules = {name: importlib.import_module(name) for name in ("torch", "numpy", "transformers", "BPfold", "structRFM.model", "structRFM.data", "ss_pred", "param_turner2004")}
    except (ImportError, AttributeError) as error:
        raise PreflightError(f"Official runtime import failed: {type(error).__name__}: {error}") from error
    return {
        "python": platform.python_version(),
        "platform": platform.machine(),
        "torch": modules["torch"].__version__,
        "torch_cuda": modules["torch"].version.cuda,
        "torch_arch_list": modules["torch"].cuda.get_arch_list(),
        "numpy": modules["numpy"].__version__,
        "transformers": modules["transformers"].__version__,
        "bpfold": importlib.metadata.version("BPfold"),
        "official_dp": "interface.predict_mxfold",
        "interface": interface,
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
        runtime = runtime_preflight(source, root) if require_runtime else None
    except ValueError as error:
        raise PreflightError(str(error)) from error
    return {"root": root, "dataset_path": Path(dataset_path), "dataset_rows": rows, "checkpoint": checkpoint, "source": source_identity, "runtime": runtime}


def prepare_models(source, checkpoint_path, device, root=DEFAULT_ROOT):
    source = Path(source)
    sys.path.insert(0, str(Path(root) / "pydeps"))
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


def configure_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTHONHASHSEED"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is required; no whole-model CPU inference fallback is permitted")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise PreflightError(f"Current torch does not compile for {architecture}: {torch.cuda.get_arch_list()}")
    torch.manual_seed(0)
    return torch, torch.device("cuda:0"), architecture


def execute(checked, output_path, gpu="0"):
    if checked.get("runtime") is None:
        raise PreflightError("Runtime preflight is mandatory before execution")
    output_path = Path(output_path)
    inside_root(output_path, checked["root"])
    if output_path.exists():
        raise PreflightError(f"Refusing to overwrite predictions: {output_path}")
    torch, device, _ = configure_gpu(gpu)
    model, tokenizer, pretrained_model = prepare_models(checked["source"]["path"], checked["checkpoint"]["path"], device, checked["root"])
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


def readiness_one(checked, gpu="0"):
    if checked.get("runtime") is None:
        raise PreflightError("Runtime preflight is mandatory before GPU readiness")
    torch, device, architecture = configure_gpu(gpu)
    model, tokenizer, pretrained_model = prepare_models(checked["source"]["path"], checked["checkpoint"]["path"], device, checked["root"])
    row = min(checked["dataset_rows"], key=lambda value: len(value["seq"]))
    sequence = normalize_sequence(row["seq"])
    input_ids = tokenizer("[CLS]" + sequence)["input_ids"][:512]
    input_ids = [0 if value is None else value for value in input_ids]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        embeddings = pretrained_model(input_tensor)
        _, _, structures = model((sequence,), embeddings)
    prediction = {"name": row["name"], "seq": sequence, "connects": [int(value) for value in structures[0][1:]]}
    validate_prediction_rows([row], [prediction])
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_SINGLE_SEQUENCE_GPU_FORWARD_AND_CPU_DP_READINESS",
        "n_sequences_processed": 1,
        "sequence_name": row["name"],
        "sequence_length": len(sequence),
        "checkpoint_sha256": checked["checkpoint"]["sha256"],
        "runtime": checked["runtime"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_architecture": architecture,
        "model_forward_device": str(device),
        "dp_device": "cpu",
        "prediction_artifact": None,
        "metrics": None,
        "full_dataset_inference_performed": False,
        "summary_only_prediction_substitution": False,
    }
    write_json(checked["root"] / "reports/gpu_readiness_one_20261126.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Official structRFM v0.0.8 bpRNA-new 5388 GPU launcher; default is CPU-only preflight")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "predictions/structrfm_official_v0.0.8_bprna_new_5388.jsonl")
    parser.add_argument("--gpu", default="0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run full GPU inference only after every preflight passes")
    mode.add_argument("--readiness-one", action="store_true", help="Run one shortest-sequence GPU forward + CPU DP readiness check, never a prediction artifact")
    args = parser.parse_args()
    try:
        checked = preflight(args.root, args.dataset, args.source, require_runtime=True)
        if not args.execute and not args.readiness_one:
            print(json.dumps({"status": "PREFLIGHT_PASS_GPU_NOT_EXECUTED", "n_sequences": 5388, "checkpoint_sha256": checked["checkpoint"]["sha256"], "runtime": checked["runtime"]}, indent=2))
            return 0
        result = readiness_one(checked, args.gpu) if args.readiness_one else execute(checked, args.output, args.gpu)
        print(json.dumps(result, indent=2))
        return 0
    except (PreflightError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_PREFLIGHT", "gpu_executed": False, "reason": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
