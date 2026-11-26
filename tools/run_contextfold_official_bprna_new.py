#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from contextfold_official_recovery_20261126 import (
    ARTIFACT_ROOT,
    DATASET,
    DEFAULT_MODEL,
    ContractError,
    load_dataset,
    native_input,
    parse_native_output,
    sha256,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests/foldnar_v2/contextfold_official_recovery_20261126.json"


class PreflightError(RuntimeError):
    pass


def inside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise PreflightError(f"Asset or output must be under {root}") from error
    return resolved


def preflight(asset_root: Path, manifest_path: Path, dataset_path: Path, java: str | None = None) -> dict:
    document = json.loads(manifest_path.read_text())
    implementation = document.get("implementation", {})
    if implementation.get("status") != "READY_OFFICIAL_DISTRIBUTION_VERIFIED":
        raise PreflightError("Official ContextFold implementation and trained parameters are not verified; inference is fail-closed")
    archive = implementation.get("archive", {})
    archive_path = inside_root(Path(archive.get("path", "")), asset_root)
    expected_sha = archive.get("sha256")
    if not expected_sha or not archive_path.is_file() or sha256(archive_path) != expected_sha:
        raise PreflightError("Official ContextFold_1_00.zip identity/SHA256 is unavailable or mismatched")
    package_root = inside_root(asset_root / "source/ContextFold", asset_root)
    predictor = package_root / "bin/contextFold/app/Predict.class"
    model = inside_root(package_root / DEFAULT_MODEL, package_root)
    if not predictor.is_file() or not model.is_file():
        raise PreflightError("Official Predict.class or trained/StHighCoHigh.model is missing")
    java_executable = java or shutil.which("java")
    if not java_executable:
        raise PreflightError("Java 6+ is required by the official ContextFold 1.00 readme")
    rows = load_dataset(dataset_path)
    return {
        "document": document,
        "package_root": package_root,
        "model": model,
        "java": java_executable,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict:
    asset_root = args.asset_root.resolve()
    output = inside_root(args.output, asset_root)
    state = preflight(asset_root, args.manifest, args.dataset, args.java)
    rows = state["rows"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="contextfold_cpu_", dir=asset_root) as directory:
        work = Path(directory)
        native_in = work / "bprna_new.native.txt"
        native_out = work / "bprna_new.native.pred.txt"
        native_in.write_text(native_input(rows))
        command = [
            state["java"], "-cp", "bin", "contextFold.app.Predict",
            f"in:{native_in}", f"model:{state['model']}", f"out:{native_out}",
        ]
        completed = subprocess.run(
            command,
            cwd=state["package_root"],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Official ContextFold exited {completed.returncode}: {completed.stderr[-4000:]}")
        predictions = parse_native_output(native_out.read_text(), rows)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "COMPLETE_OFFICIAL_CPU_PREDICTION",
        "n_sequences": len(predictions),
        "output": str(output),
        "sha256": sha256(output),
        "model": DEFAULT_MODEL,
        "device": "CPU_JAVA",
        "reference_labels_used_for_inference": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay official ContextFold 1.00 on the frozen bpRNA-new 5388 contract")
    parser.add_argument("--asset-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--java")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    args = parser.parse_args()
    try:
        report = run(args)
    except (ContractError, PreflightError) as error:
        parser.exit(2, f"PREFLIGHT_BLOCKED: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
