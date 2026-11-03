#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import shlex
from pathlib import Path

ALLOWED_CHANGES = {"protocol.code_commit", "protocol.code_sha256"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_json(path, payload):
    atomic_bytes(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode())


def diff_paths(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        assert before.keys() == after.keys(), f"object keys differ at {prefix}"
        return set().union(*(diff_paths(before[key], after[key], f"{prefix}.{key}" if prefix else key)
                             for key in before))
    if isinstance(before, list) and isinstance(after, list):
        assert len(before) == len(after), f"list length differs at {prefix}"
        return set().union(*(diff_paths(left, right, f"{prefix}[{index}]")
                             for index, (left, right) in enumerate(zip(before, after)))) if before else set()
    return set() if before == after else {prefix}


def finalize(result_path, pin_path, backup_path, audit_path):
    pin = json.loads(pin_path.read_text())
    assert pin["status"] == "RUNNING_PROCESS_CODE_IDENTITY_PINNED"
    assert pin["correction_scope"] == "protocol.code_sha256 and protocol.code_commit only; predictions/runs/timing immutable"
    arguments = shlex.split(pin["process_command"])
    out_index = arguments.index("--out") + 1
    assert Path(arguments[out_index]).resolve() == result_path.resolve()
    current_bytes = result_path.read_bytes()
    current = json.loads(current_bytes)
    assert current["status"] == "PASS" and len(current["runs"]) == 3
    if backup_path.exists():
        original = json.loads(backup_path.read_bytes())
    else:
        original = current
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open("xb") as handle:
            handle.write(current_bytes)
    corrected = copy.deepcopy(original)
    corrected["protocol"]["code_commit"] = pin["code_commit"]
    corrected["protocol"]["code_sha256"] = pin["code_sha256"]
    changed = diff_paths(original, corrected)
    assert changed == ALLOWED_CHANGES, changed
    assert diff_paths(current, corrected) <= ALLOWED_CHANGES
    atomic_json(result_path, corrected)
    assert json.loads(result_path.read_text()) == corrected
    audit = {"schema_version": 1, "status": "PASS_CODE_IDENTITY_CORRECTION_ONLY",
             "result": str(result_path), "backup": str(backup_path), "pin": str(pin_path),
             "backup_sha256": sha256(backup_path), "result_sha256": sha256(result_path),
             "pin_sha256": sha256(pin_path), "changed_paths": sorted(changed),
             "code_commit": pin["code_commit"], "code_sha256": pin["code_sha256"]}
    atomic_json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = finalize(args.result, args.pin, args.backup, args.audit)
    print(json.dumps({"status": audit["status"], "result_sha256": audit["result_sha256"],
                      "audit_sha256": sha256(args.audit), "changed_paths": audit["changed_paths"]}, indent=2))


if __name__ == "__main__":
    main()
