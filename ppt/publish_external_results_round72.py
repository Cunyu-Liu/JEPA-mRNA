#!/usr/bin/env python3
"""Wait for final margin evaluation and atomically publish a verified 17-slide delivery deck."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from pptx import Presentation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DECK_NAME = "RNA-JEV_FoldNar_Phase0验证记录_20261124.pptx"
SOURCE_DECK = OUT / DECK_NAME
DELIVERY_DECK = OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round72_delivery.pptx"
WORK_DIR = ROOT / ".round72_work/final_publication"
EVALUATION_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_independent_eval_v1/generation_01")
CROSS_SOURCE_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/cross_source_leakage_v1")
MODEL_DEV_REGISTRY = ROOT / "manifests/foldnar_v2/model_dev_registry.json"
STRUCT_RESULTS = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structrfm_official_v0.0.8/reports/structrfm_official_bprna_new_metrics_20261126.json")
STRUCT_REGISTRY = ROOT / "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json"
RAW_REGISTRY = ROOT / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
BASELINE_REGISTRY = ROOT / "manifests/foldnar_v1/baseline_registry.json"
LONGSEQ_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/longseq_linearpartition_contract_20261126")
FINAL_STATUSES = {"PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL", "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"}
FINAL_FILES = ("independent_evaluation.json", "gate_summary.json", "input_manifest.json",
               "exposure_sensitivity.json")


class PublicationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise PublicationError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                             dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        if sha256(temporary_path) != sha256(source):
            raise PublicationError(f"atomic copy hash mismatch: {source}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    if not isinstance(value, dict):
        raise PublicationError(f"JSON must be an object: {path}")
    return value


def read_final_marker(evaluation_root: Path) -> dict:
    evaluation_path = evaluation_root / "independent_evaluation.json"
    evaluation = load_json(evaluation_path)
    status = evaluation.get("status")
    if status == "IN_PROGRESS" or evaluation.get("final_verdict_allowed") is False:
        return {"ready": False, "status": status, "reason": "final_verdict_allowed is not true"}
    if status not in FINAL_STATUSES or evaluation.get("final_verdict_allowed") is not True:
        raise PublicationError(f"engineering/non-final evaluation status refused: {status}")
    overall = evaluation.get("overall_pass")
    expected = "PASS_STRUCTURED_MARGIN_INDEPENDENT_EVAL" if overall else "FAIL_STRUCTURED_MARGIN_INDEPENDENT_EVAL"
    if type(overall) is not bool or status != expected:
        raise PublicationError("final scientific status and overall_pass disagree")
    if not isinstance(evaluation.get("preregistered_exposure_sensitivity"), dict):
        raise PublicationError("final evaluation lacks embedded preregistered exposure sensitivity")
    for name in FINAL_FILES[1:]:
        path = evaluation_root / name
        if not path.is_file():
            raise PublicationError(f"final evaluation is missing required exposure/final evidence: {path}")
    identities = {name: identity(evaluation_root / name) for name in FINAL_FILES}
    return {"ready": True, "status": status, "overall_pass": overall,
            "evaluation_root": str(evaluation_root.resolve()), "files": identities,
            "observed_utc": utc_now()}


def wait_for_final_marker(evaluation_root: Path, wait: bool = False,
                          timeout_seconds: float = 0, poll_interval: float = 30) -> dict:
    if wait and (timeout_seconds <= 0 or poll_interval <= 0):
        raise PublicationError("--wait requires positive --timeout-seconds and --poll-interval")
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_reason = "not checked"
    while True:
        attempts += 1
        try:
            marker = read_final_marker(evaluation_root)
            if marker["ready"]:
                marker["wait_attempts"] = attempts
                return marker
            last_reason = marker["reason"]
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError) as error:
            if not wait:
                raise PublicationError(f"final marker unavailable: {error}") from error
            last_reason = str(error)
        if not wait:
            raise PublicationError(f"final margin evaluation is not complete: {last_reason}")
        if time.monotonic() >= deadline:
            raise PublicationError(f"timed out waiting for final margin evaluation: {last_reason}")
        time.sleep(min(poll_interval, max(0.01, deadline - time.monotonic())))


def require_source_unchanged(source: Path, expected_sha256: str) -> None:
    if sha256(source) != expected_sha256:
        raise PublicationError("source deck changed during publication; refusing delivery")


def verify_prior_xml(source: Path, delivery: Path, count: int = 14) -> None:
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(delivery) as after:
        for index in range(1, count + 1):
            part = f"ppt/slides/slide{index}.xml"
            if before.read(part) != after.read(part):
                raise PublicationError(f"source slide XML changed in delivery: {part}")


def run_publication(source_deck: Path = SOURCE_DECK, delivery_deck: Path = DELIVERY_DECK,
                    work_dir: Path = WORK_DIR, evaluation_root: Path = EVALUATION_ROOT,
                    cross_source_root: Path = CROSS_SOURCE_ROOT,
                    model_dev_registry: Path = MODEL_DEV_REGISTRY,
                    struct_results: Path = STRUCT_RESULTS, struct_registry: Path = STRUCT_REGISTRY,
                    raw_registry: Path = RAW_REGISTRY, baseline_registry: Path = BASELINE_REGISTRY,
                    longseq_root: Path = LONGSEQ_ROOT, wait: bool = False,
                    timeout_seconds: float = 0, poll_interval: float = 30,
                    backup_deck: Path | None = None) -> dict:
    source_deck = source_deck.resolve()
    delivery_deck = delivery_deck.resolve()
    work_dir = work_dir.resolve()
    evaluation_root = evaluation_root.resolve()
    cross_source_root = cross_source_root.resolve()
    backup_deck = (backup_deck.resolve() if backup_deck else
                   delivery_deck.with_name(delivery_deck.stem + ".source14_backup.pptx"))
    if work_dir.exists() and any(work_dir.iterdir()):
        raise PublicationError("--work-dir must be new or empty")
    if delivery_deck.exists() or backup_deck.exists():
        raise PublicationError("delivery and source14 backup must not already exist")
    work_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256(source_deck)
    if len(Presentation(source_deck).slides) != 14:
        raise PublicationError("source deck must remain the frozen 14-slide Phase0 deck")
    steps = []
    published = False
    try:
        marker = wait_for_final_marker(evaluation_root, wait, timeout_seconds, poll_interval)
        transaction = work_dir / "transaction"
        transaction.mkdir()
        deck = transaction / DECK_NAME
        shutil.copy2(source_deck, deck)
        round71_backup = transaction / "round71_source14.pptx"
        round71_report = transaction / "round71_build_verification.json"
        round71_independent = transaction / "round71_independent_build_verification.json"
        round72_backup = transaction / "round72_margin15.pptx"
        round72_report = transaction / "round72_build_verification.json"
        round72_independent = transaction / "round72_independent_build_verification.json"
        python = sys.executable
        external_args = ["--struct-results", str(struct_results.resolve()),
                         "--struct-registry", str(struct_registry.resolve()),
                         "--raw-registry", str(raw_registry.resolve()),
                         "--baseline-registry", str(baseline_registry.resolve()),
                         "--longseq-root", str(longseq_root.resolve())]
        commands = [
            ("round71_readiness", [python, str(OUT / "check_round71_readiness.py"),
             "--deck", str(deck), "--evaluation-root", str(evaluation_root),
             "--cross-source-root", str(cross_source_root),
             "--model-dev-registry", str(model_dev_registry.resolve())]),
            ("round71_build_page34", [python, str(OUT / "add_phase0_structured_margin_result_round71.py"),
             "--deck", str(deck), "--evaluation-root", str(evaluation_root),
             "--cross-source-root", str(cross_source_root), "--backup", str(round71_backup),
             "--verification", str(round71_report)]),
            ("round71_independent_verification", [python, str(OUT / "independent_build_verify_round71.py"),
             "--source-deck", str(round71_backup), "--primary-deck", str(deck),
             "--evaluation-root", str(evaluation_root), "--cross-source-root", str(cross_source_root),
             "--work-dir", str(transaction / "round71_independent"),
             "--verification", str(round71_independent)]),
            ("round72_readiness", [python, str(OUT / "check_round72_readiness.py"),
             "--deck", str(deck), "--margin-verification", str(round71_report),
             "--margin-independent-verification", str(round71_independent), *external_args]),
            ("round72_build_pages35_36", [python, str(OUT / "add_external_results_round72.py"),
             "--deck", str(deck), "--margin-verification", str(round71_report),
             "--margin-independent-verification", str(round71_independent),
             "--backup", str(round72_backup), "--verification", str(round72_report), *external_args]),
            ("round72_independent_verification", [python, str(OUT / "independent_build_verify_round72.py"),
             "--source-deck", str(round72_backup), "--margin-verification", str(round71_report),
             "--margin-independent-verification", str(round71_independent), "--primary-deck", str(deck),
             "--work-dir", str(transaction / "round72_independent"),
             "--verification", str(round72_independent)]),
        ]
        for index, (name, command) in enumerate(commands, start=1):
            require_source_unchanged(source_deck, source_sha256)
            before = sha256(deck)
            started = utc_now()
            started_monotonic = time.monotonic()
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = "0"
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=600, env=environment)
            record = {"step": index, "name": name, "command": command, "cwd": str(ROOT),
                      "started_utc": started, "finished_utc": utc_now(),
                      "duration_seconds": time.monotonic() - started_monotonic,
                      "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                      "deck_sha256_before": before, "deck_sha256_after": sha256(deck),
                      "source_deck_sha256": source_sha256}
            log_base = work_dir / "logs" / f"{index:02d}_{name}"
            log_base.parent.mkdir(parents=True, exist_ok=True)
            log_base.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
            log_base.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
            record["stdout_log"] = str(log_base.with_suffix(".stdout.log"))
            record["stderr_log"] = str(log_base.with_suffix(".stderr.log"))
            atomic_json(log_base.with_suffix(".json"), record)
            steps.append(record)
            require_source_unchanged(source_deck, source_sha256)
            if result.returncode:
                raise PublicationError(f"step {index} {name} failed with exit {result.returncode}; "
                                       f"see {record['stderr_log']}")
        if len(Presentation(deck).slides) != 17:
            raise PublicationError("verified transaction deck is not 17 slides")
        verify_prior_xml(source_deck, deck)
        independent72 = load_json(round72_independent)
        if (independent72.get("status") != "PASS_INDEPENDENT_CLEAN_BUILD_ALL_ZIP_PARTS_IDENTICAL"
                or independent72.get("prior_slide_xml_preserved") != 15
                or independent72.get("different_parts") != []):
            raise PublicationError("round72 independent verification did not pass")
        report = {
            "status": "READY_FOR_ATOMIC_PUBLICATION", "scientific_status": marker["status"],
            "scientific_overall_pass": marker["overall_pass"], "source_deck": identity(source_deck),
            "delivery_deck": identity(deck), "backup_deck": identity(source_deck),
            "delivery_path": str(delivery_deck), "backup_path": str(backup_deck),
            "work_dir": str(work_dir), "final_marker": marker, "steps": steps,
            "source_slide_xml_preserved": 14, "prior_slide_xml_preserved": 15,
            "slide_count": 17, "automatic_git_commit": False,
        }
        atomic_json(work_dir / "publication_success.json", report)
        require_source_unchanged(source_deck, source_sha256)
        atomic_copy(source_deck, backup_deck)
        atomic_copy(deck, delivery_deck)
        published = True
        require_source_unchanged(source_deck, source_sha256)
        return report
    except Exception as error:
        require_source_unchanged(source_deck, source_sha256)
        if published and delivery_deck.exists():
            delivery_deck.unlink()
        failure = {"status": "FAILED_NO_PARTIAL_DELIVERY", "error": str(error),
                   "failed_utc": utc_now(), "source_deck_sha256": source_sha256,
                   "delivery_deck": str(delivery_deck), "delivery_exists": delivery_deck.exists(),
                   "backup_deck": str(backup_deck), "backup_exists": backup_deck.exists(),
                   "completed_steps": [step["name"] for step in steps if step["exit_code"] == 0],
                   "step_records": steps, "automatic_git_commit": False}
        atomic_json(work_dir / "publication_failure.json", failure)
        if isinstance(error, PublicationError):
            raise
        raise PublicationError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-deck", type=Path, default=SOURCE_DECK)
    parser.add_argument("--delivery-deck", type=Path, default=DELIVERY_DECK)
    parser.add_argument("--backup-deck", type=Path)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--cross-source-root", type=Path, default=CROSS_SOURCE_ROOT)
    parser.add_argument("--model-dev-registry", type=Path, default=MODEL_DEV_REGISTRY)
    parser.add_argument("--struct-results", type=Path, default=STRUCT_RESULTS)
    parser.add_argument("--struct-registry", type=Path, default=STRUCT_REGISTRY)
    parser.add_argument("--raw-registry", type=Path, default=RAW_REGISTRY)
    parser.add_argument("--baseline-registry", type=Path, default=BASELINE_REGISTRY)
    parser.add_argument("--longseq-root", type=Path, default=LONGSEQ_ROOT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=0)
    parser.add_argument("--poll-interval", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        report = run_publication(args.source_deck, args.delivery_deck, args.work_dir,
                                 args.evaluation_root, args.cross_source_root, args.model_dev_registry,
                                 args.struct_results, args.struct_registry, args.raw_registry,
                                 args.baseline_registry, args.longseq_root, args.wait,
                                 args.timeout_seconds, args.poll_interval, args.backup_deck)
    except (PublicationError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"round72 publication refused without partial delivery: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
