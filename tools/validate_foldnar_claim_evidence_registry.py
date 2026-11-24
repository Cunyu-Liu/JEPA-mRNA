#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path, PurePosixPath

STATUSES = {"UNSUPPORTED", "IN_PROGRESS", "SUPPORTED"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CLAIM_FIELDS = {
    "claim_id", "claim", "phenomenon", "intervention", "expected_observation",
    "artifact_ids", "limitation", "current_status", "supported", "abstract_title_eligible",
}


def validate(registry):
    errors = []
    if registry.get("schema_version") != "foldnar.preprint_claim_evidence_registry.v1":
        errors.append("unexpected schema_version")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return errors + ["artifacts must be a non-empty object"]
    for artifact_id, artifact in artifacts.items():
        if not re.fullmatch(r"A\d{2,}", artifact_id):
            errors.append(f"{artifact_id}: malformed artifact ID")
        if not isinstance(artifact, dict):
            errors.append(f"{artifact_id}: artifact must be an object")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path or not PurePosixPath(path).is_absolute():
            errors.append(f"{artifact_id}: path must be absolute POSIX")
        elif ".." in PurePosixPath(path).parts or "//" in path or "~" in path:
            errors.append(f"{artifact_id}: path must be normalized and contain no tilde")
        availability = artifact.get("availability")
        digest = artifact.get("sha256")
        if availability not in {"EXISTING", "PLANNED"}:
            errors.append(f"{artifact_id}: invalid availability")
        if availability == "EXISTING" and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            errors.append(f"{artifact_id}: existing artifact needs a full lowercase SHA-256")
        if availability == "PLANNED" and digest is not None:
            errors.append(f"{artifact_id}: planned artifact hash must be null")
    claims = registry.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a non-empty array")
        claims = []
    seen = set()
    for index, claim in enumerate(claims):
        claim_id = claim.get("claim_id", f"claim[{index}]")
        missing = CLAIM_FIELDS - set(claim)
        if missing:
            errors.append(f"{claim_id}: missing fields {sorted(missing)}")
        if claim_id in seen:
            errors.append(f"{claim_id}: duplicate claim ID")
        seen.add(claim_id)
        refs = claim.get("artifact_ids")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{claim_id}: artifact_ids must be non-empty")
        else:
            errors.extend(f"{claim_id}: unknown artifact {ref}" for ref in refs if ref not in artifacts)
    for location, item in walk_status_objects(registry):
        status = item.get("current_status")
        if status not in STATUSES:
            errors.append(f"{location}: invalid current_status")
            continue
        if item.get("supported") is not (status == "SUPPORTED"):
            errors.append(f"{location}: supported flag contradicts current_status")
        if status != "SUPPORTED" and item.get("abstract_title_eligible") is not False:
            errors.append(f"{location}: non-supported claim cannot be abstract/title eligible")
    baseline = registry.get("baseline_gaps", {})
    reference = Decimal(str(baseline.get("xens2_micro_f1")))
    for row in baseline.get("rows", []):
        expected = Decimal(str(row["micro_f1"])) - reference
        if Decimal(str(row["absolute_gap"])) != expected:
            errors.append(f"baseline gap arithmetic mismatch: {row.get('system')}")
    return errors


def walk_status_objects(value, location="$"):
    if isinstance(value, dict):
        if "current_status" in value:
            yield location, value
        for key, item in value.items():
            yield from walk_status_objects(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_status_objects(item, f"{location}[{index}]")


def validate_markdown(registry, path):
    text = path.read_text(encoding="utf-8")
    errors = []
    for claim in registry["claims"]:
        prefix = f"| {claim['claim_id']} |"
        lines = [line for line in text.splitlines() if line.startswith(prefix)]
        expected = f"| **{claim['current_status']}** |"
        if len(lines) != 1 or not lines[0].endswith(expected):
            errors.append(f"markdown status mismatch: {claim['claim_id']}")
    return errors


def verify_hashes(registry):
    errors = []
    for artifact_id, artifact in registry["artifacts"].items():
        if artifact["availability"] != "EXISTING":
            continue
        path = Path(artifact["path"])
        if not path.is_file():
            errors.append(f"{artifact_id}: missing file {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            errors.append(f"{artifact_id}: SHA-256 mismatch for {path}")
    return errors


def self_test(registry):
    cases = []
    changed = copy.deepcopy(registry)
    changed["claims"][0]["supported"] = True
    cases.append(("in-progress marked supported", changed, "supported flag"))
    changed = copy.deepcopy(registry)
    changed["claims"][2]["abstract_title_eligible"] = True
    cases.append(("unsupported made title eligible", changed, "abstract/title"))
    changed = copy.deepcopy(registry)
    changed["claims"][6]["supported"] = False
    cases.append(("supported status contradicted", changed, "supported flag"))
    changed = copy.deepcopy(registry)
    changed["artifacts"]["A00"]["sha256"] = "abc"
    cases.append(("malformed hash", changed, "SHA-256"))
    changed = copy.deepcopy(registry)
    changed["artifacts"]["A00"]["path"] = "relative/path.json"
    cases.append(("relative path", changed, "absolute POSIX"))
    changed = copy.deepcopy(registry)
    changed["claims"][0]["artifact_ids"] = ["A99"]
    cases.append(("broken artifact reference", changed, "unknown artifact"))
    failures = []
    for name, changed, expected in cases:
        errors = validate(changed)
        if not any(expected in error for error in errors):
            failures.append(f"negative fixture was accepted: {name}")
    if validate(registry):
        failures.append("valid fixture rejected by self-test")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--verify-existing-hashes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate(registry)
    if args.markdown:
        errors.extend(validate_markdown(registry, args.markdown))
    if args.verify_existing_hashes:
        errors.extend(verify_hashes(registry))
    if args.self_test:
        errors.extend(self_test(registry))
    if errors:
        raise SystemExit("FAIL\n" + "\n".join(f"- {error}" for error in errors))
    counts = {status: 0 for status in sorted(STATUSES)}
    for claim in registry["claims"]:
        counts[claim["current_status"]] += 1
    print(f"PASS: {len(registry['claims'])} claims {counts}; {len(registry['artifacts'])} artifacts")
    print("Status gates, artifact references, paths, hashes, gap arithmetic, and requested negative fixtures are valid.")


if __name__ == "__main__":
    main()
