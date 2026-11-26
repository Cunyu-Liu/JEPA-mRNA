#!/usr/bin/env python3
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

DEFAULT_OFFICIAL = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/ernierna_official_recovery_20261126/ERNIE-RNA")
REQUIRED = ("torch", "numpy", "fairseq", "hydra", "omegaconf", "regex", "sacrebleu", "tensorboardX",
            "portalocker", "bitarray", "antlr4", "yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "static checks must mask all GPU devices"
    sources = sorted(args.official_root.rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    modules = {}
    for name in REQUIRED:
        spec = importlib.util.find_spec(name)
        try:
            version = importlib.metadata.version(name) if spec is not None else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        modules[name] = {"present": spec is not None, "distribution_version": version,
                         "origin": None if spec is None else spec.origin}
    missing = [name for name, row in modules.items() if not row["present"]]
    torch_version = modules["torch"]["distribution_version"]
    official_torch = "1.10.0"
    result = {
        "status": "PASS_RUNTIME_STATIC_COMPATIBLE" if not missing and torch_version == official_torch else "BLOCKED_RUNTIME_ENVIRONMENT",
        "scope": "AST parsing and package metadata only; no checkpoint load, tensor operation, model construction or inference",
        "python": sys.version, "executable": sys.executable,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "official_root": str(args.official_root), "n_python_sources_ast_parsed": len(sources),
        "modules": modules, "missing_modules": missing,
        "official_environment": {"python": "3.9.16", "torch": official_torch, "fairseq": "0.12.2",
                                 "source": "environment.yml"},
        "torch_load_note": "Official README requires torch 1.10.0 or an explicit weights_only=False compatibility patch on newer torch.",
        "runtime_static_compatible": not missing and torch_version == official_torch,
        "gpu_inference_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
