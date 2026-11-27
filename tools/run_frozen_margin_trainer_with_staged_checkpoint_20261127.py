#!/usr/bin/env python3
"""Run the frozen trainer with verified local staging for large checkpoints."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
LOCAL_CHECKPOINT_ROOT = Path("/home/foldnar_incident_checkpoints")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(SOURCE_REPO / "src"))

from tools import train_plan_a_gec_convergence_v1 as trainer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_atomic_torch_save(value, path, checkpoint_root=LOCAL_CHECKPOINT_ROOT):
    path = Path(path)
    checkpoint_root = Path(checkpoint_root)
    stage = checkpoint_root / path.parent.name / path.name
    stage.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer._atomic_torch_save_without_staging(value, stage)
    stage_sha256 = sha256_file(stage)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        shutil.copyfile(stage, temporary)
        if sha256_file(temporary) != stage_sha256:
            raise OSError(f"staged checkpoint copy SHA256 mismatch: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[checkpoint-staging] path={path} stage={stage} sha256={stage_sha256}", flush=True)


def validate_prerequisites_with_recovery_root(args, config):
    original_root = config["full_training"]["artifact_root"]
    config["full_training"]["artifact_root"] = str(Path(args.out).resolve().parent)
    try:
        return trainer._validate_prerequisites_without_recovery_root(args, config)
    finally:
        config["full_training"]["artifact_root"] = original_root


def main() -> None:
    trainer._atomic_torch_save_without_staging = trainer.atomic_torch_save
    trainer._validate_prerequisites_without_recovery_root = trainer.validate_full_training_prerequisites
    trainer.atomic_torch_save = staged_atomic_torch_save
    trainer.validate_full_training_prerequisites = validate_prerequisites_with_recovery_root
    sys.argv[0] = str(SOURCE_REPO / "tools/train_plan_a_gec_convergence_v1.py")
    trainer.main()


if __name__ == "__main__":
    main()
