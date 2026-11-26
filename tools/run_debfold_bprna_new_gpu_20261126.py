#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from debfold_official_recovery_20261126 import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_ROOT,
    OFFICIAL_COMMIT,
    load_dataset,
    sha256,
    utc_now,
    validate_prediction_rows,
    write_json,
)


class PreflightError(RuntimeError):
    pass


def inside_root(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise PreflightError(f"Official assets must remain under {root}: {path}") from error


def check_asset(path, expected_sha, root, label):
    path = Path(path)
    inside_root(path, root)
    if not path.is_file():
        raise PreflightError(f"Missing official {label}: {path}")
    actual = sha256(path)
    if actual != expected_sha:
        raise PreflightError(f"Official {label} SHA256 mismatch: {path}: {actual}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def preflight(adapter_path, root=DEFAULT_ROOT, dataset_path=DEFAULT_DATASET):
    root = Path(root)
    dataset_rows = load_dataset(dataset_path)
    adapter_path = Path(adapter_path)
    if not adapter_path.is_file():
        raise PreflightError(
            "DEBFold is not GPU-ready: official local implementation, weights and batch CLI were not published; "
            f"no verified adapter manifest at {adapter_path}"
        )
    adapter = json.loads(adapter_path.read_text())
    if adapter.get("status") != "READY_OFFICIAL_ADAPTER_VERIFIED":
        raise PreflightError("Adapter is not marked READY_OFFICIAL_ADAPTER_VERIFIED")
    if adapter.get("official_repository") != "https://github.com/cobisLab/DEBFold" or adapter.get("official_commit") != OFFICIAL_COMMIT:
        raise PreflightError("Adapter does not match the frozen official DEBFold commit")
    if any(len(row["seq"]) > 512 for row in dataset_rows):
        long_protocol = adapter.get("over_512_protocol", {})
        if long_protocol.get("status") != "VERIFIED_OFFICIAL_PROTOCOL" or not long_protocol.get("url"):
            raise PreflightError("Project rows over 512 nt require a verified official DEBFold long-sequence protocol")
    implementation = check_asset(adapter["implementation"]["entrypoint"], adapter["implementation"]["sha256"], root, "implementation entrypoint")
    weights = [check_asset(row["path"], row["sha256"], root, "weights") for row in adapter["weights"]]
    if not weights:
        raise PreflightError("At least one official weights file is required")
    command = adapter.get("command")
    placeholders = {"input_jsonl", "output_jsonl", "implementation", "weights"}
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise PreflightError("Official command must be a nonempty argument list; shell strings are forbidden")
    used = set()
    for part in command:
        for field in placeholders:
            marker = "{" + field + "}"
            if marker in part:
                used.add(field)
    if used != placeholders:
        raise PreflightError(f"Official command placeholders must be exactly {sorted(placeholders)}; got {sorted(used)}")
    return {
        "adapter": adapter,
        "dataset_rows": dataset_rows,
        "dataset_path": Path(dataset_path),
        "implementation": implementation,
        "weights": weights,
        "command": command,
    }


def execute(preflight_result, output_path, root=DEFAULT_ROOT, gpu="0"):
    root = Path(root)
    output_path = Path(output_path)
    inside_root(output_path, root)
    if output_path.exists():
        raise PreflightError(f"Refusing to overwrite predictions: {output_path}")
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    input_path = runtime / "bprna_new_5388_input.jsonl"
    temporary_input = input_path.with_suffix(".jsonl.tmp")
    with temporary_input.open("w") as handle:
        for row in preflight_result["dataset_rows"]:
            handle.write(json.dumps({"name": row["name"], "seq": row["seq"]}, ensure_ascii=False) + "\n")
    os.replace(temporary_input, input_path)
    values = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "implementation": preflight_result["implementation"]["path"],
        "weights": os.pathsep.join(row["path"] for row in preflight_result["weights"]),
    }
    command = [part.format(**values) for part in preflight_result["command"]]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONHASHSEED"] = "0"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=environment)
    prediction_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    validate_prediction_rows(preflight_result["dataset_rows"], prediction_rows)
    run_manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_COMPLETE_PER_SEQUENCE_PREDICTIONS",
        "official_repository": preflight_result["adapter"]["official_repository"],
        "official_commit": preflight_result["adapter"]["official_commit"],
        "implementation": preflight_result["implementation"],
        "weights": preflight_result["weights"],
        "command": command,
        "cuda_visible_devices": gpu,
        "dataset": {"path": str(preflight_result["dataset_path"]), "sha256": sha256(preflight_result["dataset_path"]), "n_sequences": 5388},
        "input": {"path": str(input_path), "sha256": sha256(input_path), "contains_reference_pairs": False},
        "predictions": {"path": str(output_path), "sha256": sha256(output_path), "n_sequences": len(prediction_rows), "coverage": 1.0},
        "external_test_used_for_tuning": False,
    }
    write_json(output_path.with_suffix(".run_manifest.json"), run_manifest)
    return run_manifest


def main():
    parser = argparse.ArgumentParser(description="Fail-closed official DEBFold bpRNA-new 5388 GPU launcher; default performs CPU preflight only")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ROOT / "official_adapter.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "predictions/debfold_official_bprna_new_5388.jsonl")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true", help="Actually invoke the verified official GPU command")
    args = parser.parse_args()
    try:
        checked = preflight(args.adapter, args.root, args.dataset)
        if not args.execute:
            print(json.dumps({"status": "PREFLIGHT_PASS_GPU_NOT_EXECUTED", "n_sequences": 5388, "dataset_sha256": sha256(args.dataset)}, indent=2))
            return 0
        result = execute(checked, args.output, args.root, args.gpu)
        print(json.dumps(result, indent=2))
        return 0
    except (PreflightError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_OFFICIAL_IMPLEMENTATION_AND_WEIGHTS", "gpu_executed": False, "reason": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
