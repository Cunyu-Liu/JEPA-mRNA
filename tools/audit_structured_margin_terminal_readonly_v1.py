#!/root/miniconda3/envs/lucaone_cu128/bin/python
"""Wait for and audit the first terminal structured-margin validation without GPU work."""

import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ARMS = ("matched_control", "ensemble_context")
STEP = 10000
TERMINAL_UNFROZEN = 20
EXPECTED_DEV = 9143
EXPECTED_LAUNCH_COMMIT = "a846d90519e575a79b7db76f63692afb755e8ac6"
EXPECTED_INTEGRATION_HEAD = "98abf143b604e1d1d1273777df3cf16dbb726dca"
MARGIN_METRIC = "length_normalized_pair_hamming_structured_margin"
DEFAULT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_full_v1/generation_01")
DEFAULT_OUTPUT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structured_margin_full_v1/step10000_readonly_audit.json")
DEFAULT_NLL = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/gec_seed0_v3_requalified_v1/qualification_summary.json")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class MetadataOnly:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return MetadataOnly

    def persistent_load(self, pid):
        return MetadataOnly(pid)


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        payload = archive.read(name)
    metadata = MetadataUnpickler(io.BytesIO(payload)).load()
    return metadata, {"data_pickle_bytes": len(payload), "data_pickle_sha256": sha256_bytes(payload)}


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validation_ready(root, arm):
    try:
        rows = read_jsonl(root / arm / "validation_log.jsonl")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return any(row.get("step") == STEP for row in rows)


def wait_for_terminal(root, poll_seconds, timeout_seconds):
    started = time.monotonic()
    pending_message = None
    while True:
        ready = {arm: validation_ready(root, arm) for arm in ARMS}
        if all(ready.values()):
            return
        errors = [arm for arm in ARMS if (root / arm / "validation_error.jsonl").exists()]
        if errors:
            raise RuntimeError(f"validation error ledger present: {','.join(errors)}")
        for arm in ARMS:
            if not ready[arm]:
                launch = read_json(root / arm / "launch_identity_v1.json")
                if not pid_alive(int(launch["pid"])):
                    raise RuntimeError(f"arm exited before step{STEP} validation: {arm}")
        message = ",".join(arm for arm in ARMS if not ready[arm])
        if message != pending_message:
            print(f"waiting for step{STEP} validation: {message}", file=sys.stderr, flush=True)
            pending_message = message
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for step{STEP} validation: {message}")
        time.sleep(poll_seconds)


def git_identity(code):
    def git(*args, check=True):
        result = subprocess.run(["git", "-C", str(code), *args], text=True, capture_output=True, check=False)
        if check and result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result

    return {
        "launch_commit": EXPECTED_LAUNCH_COMMIT,
        "integration_head": EXPECTED_INTEGRATION_HEAD,
        "audit_code_head": git("rev-parse", "HEAD").stdout.strip(),
        "launch_commit_is_ancestor_of_integration_head": git(
            "merge-base", "--is-ancestor", EXPECTED_LAUNCH_COMMIT, EXPECTED_INTEGRATION_HEAD, check=False,
        ).returncode == 0,
        "working_tree_clean": not git("status", "--porcelain").stdout.strip(),
    }


def audit_arm(root, arm, code):
    directory = root / arm
    meta = read_json(directory / "run_meta.json")
    launch = read_json(directory / "launch_identity_v1.json")
    config = read_json(Path(meta["gec_config"]))
    resume, resume_digest = checkpoint_metadata(directory / "resume.pt")
    best_path = directory / "best.pt"
    best, best_digest = checkpoint_metadata(best_path) if best_path.exists() else (None, {})
    validation_rows = read_jsonl(directory / "validation_log.jsonl")
    validation = next((row for row in validation_rows if row.get("step") == STEP), None)
    validation_errors = read_jsonl(directory / "validation_error.jsonl")
    batch_rows = [row for row in read_jsonl(directory / "batch_order.jsonl") if row.get("step", 0) <= STEP]
    compute_rows = [row for row in read_jsonl(directory / "compute_ledger.jsonl") if row.get("step", 0) <= STEP]
    train_rows = [row for row in read_jsonl(directory / "train_log.jsonl") if row.get("step", 0) <= STEP]
    dev_path = Path(config["assets"]["validation_ids"])
    dev_bytes = dev_path.read_bytes()
    dev_ids = dev_path.read_text().splitlines()
    registered_ids = meta["validation_record_ids"]
    trainer_path = code / "tools/train_plan_a_gec_convergence_v1.py"
    context_path = code / "src/rnajepa/gec_context.py"
    trainer_bytes = trainer_path.read_bytes()
    context_bytes = context_path.read_bytes()
    trainer = trainer_bytes.decode()
    context = context_bytes.decode()
    resume_convergence = resume["convergence"]
    finite_train = all(
        math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
        and math.isfinite(row["residual_gradient_norm"])
        and not row["divergence_reason"] and row["optimizer_step"]
        for row in train_rows
    )
    source_guards = {
        "explicit_cuda_entrypoint": "device = require_explicit_cuda(args.device)" in trainer,
        "cuda_probe_and_no_fallback": all(token in context for token in (
            'resolved.type != "cuda"', "torch.cuda.synchronize(resolved)", "CPU fallback is forbidden",
        )),
        "objective_device_guard": "CUDA structured objective left the requested device" in trainer,
        "validation_iterates_registered_records": "for row in validation_records:" in trainer,
        "per_record_and_aggregate_finite_guards": all(token in trainer for token in (
            "require_finite_validation_loss(loss.detach(), step, row, validation_metric(config))",
            "require_finite_validation_loss(aggregate, step, metric=validation_metric(config))",
        )),
        "nonfinite_gradient_guard": "not torch.isfinite(parameter.grad).all()" in trainer,
        "terminal_eligibility_guard": all(token in context for token in (
            "if n_unfrozen < self.terminal_unfrozen_blocks or step < self.terminal_step:",
            '"eligible": False, "improved": False, "save_best": False, "stop": False',
            '"save_best": improved', '"stop": self.bad_evaluations >= self.patience',
        )),
    }
    checks = {
        "launch_identity": launch["code_commit"] == EXPECTED_LAUNCH_COMMIT and launch["pid"] in (3977100, 3977304),
        "full_dev_unbounded": meta["schedule"]["validation_record_limit"] == 0 and len(dev_ids) == EXPECTED_DEV,
        "frozen_dev_hash": sha256_bytes(dev_bytes) == meta["validation_ids_sha256"],
        "dev_ids_unique_and_exact_membership": len(set(dev_ids)) == EXPECTED_DEV and len(set(registered_ids)) == EXPECTED_DEV and set(registered_ids) == set(dev_ids),
        "validation_length_id_sort_contract": 'records.sort(key=lambda row: (len(row["sequence"]), row["id"]))' in trainer,
        "validation_step10000_written_once": validation is not None and sum(row.get("step") == STEP for row in validation_rows) == 1,
        "validation_margin_finite": validation is not None and math.isfinite(validation["validation_loss"]),
        "validation_metric_correct": validation is not None and validation.get("validation_metric") == meta["validation_metric"] == MARGIN_METRIC,
        "no_validation_error": not validation_errors,
        "preterminal_validation_best_closed": all(
            not row["eligible"] and not row["save_best"] and not row["stop"] and row["best_step"] is None
            for row in validation_rows if row.get("step", 0) < STEP
        ),
        "terminal_validation_decision": validation is not None and all((
            validation["n_unfrozen"] == TERMINAL_UNFROZEN,
            validation["eligible"] is True,
            validation["improved"] is True,
            validation["save_best"] is True,
            validation["stop"] is False,
            validation["best_step"] == STEP,
            validation["best_validation_loss"] == validation["validation_loss"],
            validation["bad_evaluations"] == 0,
        )),
        "resume_step10000": resume["step"] == STEP and resume["optimizer_steps"] == STEP,
        "resume_running_latest": resume["checkpoint_phase"] == "latest" and resume["terminal_status"] == "running",
        "resume_terminal_unfrozen": resume["n_unfrozen"] == TERMINAL_UNFROZEN,
        "resume_identity": resume["run_identity"] == meta["run_identity"],
        "resume_prevalidation_best_closed": (
            not resume_convergence["eligible"] and resume_convergence["best_step"] is None
            and resume_convergence["best_validation_loss"] is None
        ),
        "best_checkpoint_present": best is not None,
        "best_checkpoint_step_and_phase": best is not None and all((
            best["step"] == STEP, best["optimizer_steps"] == STEP,
            best["checkpoint_phase"] == "best_eligible", best["terminal_status"] == "running",
            best["n_unfrozen"] == TERMINAL_UNFROZEN,
        )),
        "best_checkpoint_identity": best is not None and best["run_identity"] == meta["run_identity"],
        "best_resume_model_tensor_identity": best is not None and best["model_tensor_sha256"] == resume["model_tensor_sha256"],
        "best_checkpoint_convergence": best is not None and all((
            best["convergence"]["eligible"] is True,
            best["convergence"]["best_step"] == STEP,
            best["convergence"]["bad_evaluations"] == 0,
        )),
        "best_validation_loss_identity": best is not None and validation is not None and all((
            math.isfinite(best["validation_loss"]),
            best["validation_loss"] == validation["validation_loss"],
            best["convergence"]["best_validation_loss"] == validation["validation_loss"],
        )),
        "all_10000_train_steps_finite": len(train_rows) == STEP and [row["step"] for row in train_rows] == list(range(1, STEP + 1)) and finite_train,
        "batch_and_compute_steps_complete": len(batch_rows) == len(compute_rows) == STEP,
        "batch_compute_ids_match": all(
            batch["step"] == compute["step"] and batch["record_ids"] == compute["record_ids"] and len(batch["record_ids"]) == 4
            for batch, compute in zip(batch_rows, compute_rows)
        ),
        "source_hashes_match_run_identity": all(
            sha256_bytes(path.read_bytes()) == meta["run_identity"]["code_manifest"][name]["sha256"]
            for name, path in (
                ("tools/train_plan_a_gec_convergence_v1.py", trainer_path),
                ("src/rnajepa/gec_context.py", context_path),
            )
        ),
        **source_guards,
    }
    return {
        "checks": {name: bool(value) for name, value in checks.items()},
        "pid": launch["pid"],
        "code_commit": launch["code_commit"],
        "validation": validation,
        "validation_execution_evidence": {
            "registered_dev_count": len(registered_ids),
            "frozen_dev_count": len(dev_ids),
            "frozen_dev_ids_sha256": sha256_bytes(dev_bytes),
            "validation_record_limit": meta["schedule"]["validation_record_limit"],
            "completed_aggregate_ledger_requires_full_loop": source_guards["validation_iterates_registered_records"],
            "per_record_validation_id_ledger_present": False,
            "validation_error_records": len(validation_errors),
        },
        "resume_checkpoint": {
            "format": resume["format"], "step": resume["step"],
            "optimizer_steps": resume["optimizer_steps"], "phase": resume["checkpoint_phase"],
            "terminal_status": resume["terminal_status"], "n_unfrozen": resume["n_unfrozen"],
            "convergence": resume_convergence, **resume_digest,
        },
        "best_checkpoint": None if best is None else {
            "format": best["format"], "step": best["step"],
            "optimizer_steps": best["optimizer_steps"], "phase": best["checkpoint_phase"],
            "terminal_status": best["terminal_status"], "n_unfrozen": best["n_unfrozen"],
            "validation_loss": best["validation_loss"], "convergence": best["convergence"], **best_digest,
        },
        "train_step10000": train_rows[-1] if len(train_rows) >= STEP else None,
        "batch_order_rows_through_10000": batch_rows,
        "run_meta_status": {"status": meta["status"], "steps_completed_snapshot": meta["steps_completed"], "converged": meta["converged"]},
        "environment": meta["environment"],
        "decoder": meta["decoder"],
        "source_sha256": {"trainer": sha256_bytes(trainer_bytes), "convergence_and_cuda": sha256_bytes(context_bytes)},
        "source_guards": source_guards,
    }, batch_rows


def build_report(root, code, nll_summary):
    arms = {}
    orders = {}
    for arm in ARMS:
        arms[arm], orders[arm] = audit_arm(root, arm, code)
    registered_orders = [read_json(root / arm / "run_meta.json")["validation_record_ids"] for arm in ARMS]
    cross_checks = {
        "training_batch_and_sequence_order_identical_across_arms": orders[ARMS[0]] == orders[ARMS[1]],
        "validation_id_order_identical_across_arms": registered_orders[0] == registered_orders[1],
    }
    nll = read_json(nll_summary)
    provenance = git_identity(code)
    nll_manifest = read_json(nll_summary.parent / "artifact_manifest.json")
    nll_artifacts = {
        name: {
            "sha256": sha256_bytes((nll_summary.parent / name).read_bytes()),
            "bytes": (nll_summary.parent / name).stat().st_size,
        }
        for name in nll_manifest["artifacts"]
    }
    nll_manifest_verified = all(
        nll_artifacts[name]["sha256"] == record["sha256"] and nll_artifacts[name]["bytes"] == record["bytes"]
        for name, record in nll_manifest["artifacts"].items()
    )
    provenance["nll_requalification"] = {
        "status": nll["status"], "qualification_pass": nll["qualification_pass"],
        "artifact_manifest_verified": nll_manifest_verified, "artifacts": nll_artifacts,
        "scope": nll["reference_use"]["scope"],
        "margin_superior_to_nll_claim_allowed": nll["reference_use"]["margin_superior_to_nll_claim_allowed"],
    }
    required = [value for arm in arms.values() for value in arm["checks"].values()]
    required += list(cross_checks.values())
    required += [provenance["launch_commit_is_ancestor_of_integration_head"], nll_manifest_verified]
    status = "PASS_READONLY_STEP10000_TERMINAL_AUDIT" if all(required) else "FAIL_READONLY_STEP10000_TERMINAL_AUDIT"
    report = {
        "schema_version": "foldnar.structured_margin_step10000_readonly_audit.v1",
        "audit_status": status,
        "scope": "first_terminal_step10000_validation_only",
        "scientific_conclusion": "NOT_ASSESSED_FROM_A_SINGLE_TERMINAL_VALIDATION_POINT",
        "no_gpu_work_performed": True,
        "checkpoint_tensor_values_recomputed": False,
        "provenance": provenance,
        "cross_arm_checks": cross_checks,
        "arms": arms,
        "risks_and_limits": [
            "validation_log.jsonl stores only the finite aggregate, not the 9143 per-record IDs/losses; full coverage follows from the frozen registered order and the completed unbounded validation loop in the hash-matched trainer.",
            "run_meta.json is the startup snapshot while training is running; step10000 state is taken from resume.pt, best.pt, and validation_log.jsonl.",
            "resume.pt must still be the step10000 checkpoint when this audit runs; a later overwrite fails closed instead of substituting another step.",
            "Checkpoint tensor storages were deliberately not loaded or recomputed; this audit reads checkpoint metadata only and performs no GPU inference.",
            "The first eligible validation point cannot establish convergence, arm superiority, or scientific effectiveness.",
        ],
    }
    for arm in arms.values():
        rows = arm.pop("batch_order_rows_through_10000")
        arm["training_order_digest_through_10000"] = sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        )
    return report


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--code", type=Path, default=Path("/home/rna_jev_margin_candidate"))
    parser.add_argument("--nll-summary", type=Path, default=DEFAULT_NLL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.wait:
        try:
            wait_for_terminal(args.root, args.poll_seconds, args.timeout_seconds)
        except (RuntimeError, TimeoutError) as error:
            print(str(error), file=sys.stderr)
            return 2
    report = build_report(args.root, args.code, args.nll_summary)
    encoded = json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["audit_status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
