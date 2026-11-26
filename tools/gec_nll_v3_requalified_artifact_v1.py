#!/usr/bin/env python3
"""Finalize a versioned NLL v3 qualification artifact after two independent passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ARMS = ("matched_control", "ensemble_context", "destroy_interval_shuffle", "independent_context")
SCHEMA = "foldnar.nll_v3_requalified_artifact.v1"


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize(primary: dict, second: dict) -> dict:
    per_arm = {}
    for arm in ARMS:
        left = primary["arms"][arm]
        right = second["per_arm"][arm]
        qualified = (left["qualification_pass"] and right["qualification_pass"]
                     and second["primary_agreement"][arm])
        per_arm[arm] = {
            "qualification": "QUALIFIED_STRUCTURE_REFERENCE" if qualified else "NOT_QUALIFIED",
            "qualification_pass": qualified,
            "stop_step": left["validation"]["stop_step"], "best_step": left["validation"]["best_step"],
            "best_validation_loss": left["validation"]["best_validation_loss"],
            "bad_evaluations_from_raw_ledger": left["validation"]["bad_evaluations"],
            "best_resume_model_tensors_equal": right["direct_tensors_and_run_meta"]["all_model_tensors_equal"],
            "predictions_bound_to_best": right["predictions"]["pass"],
            "historical_code_and_data_identity": left["historical_code"]["pass"] and left["data_identity"]["pass"],
            "sole_historical_run_meta_defect": "convergence.bad_evaluations observed 0; raw stopped ledger 5",
            "run_meta_defect_isolated": right["direct_tensors_and_run_meta"]["sole_defect_is_restored_bad_counter"],
        }
    passed = all(row["qualification_pass"] for row in per_arm.values())
    return {
        "schema_version": SCHEMA,
        "status": "PASS_NLL_V3_REQUALIFIED" if passed else "FAIL_NLL_V3_REQUALIFICATION",
        "qualification_pass": passed, "per_arm": per_arm,
        "reference_use": {
            "historical_nll_four_arm_structure_reference_eligible": passed,
            "scope": "structure evaluation reference only",
            "original_scientific_fail_unchanged": True,
            "scientific_gate_recomputed_or_changed": False,
            "margin_superior_to_nll_claim_allowed": False,
            "margin_requirement": "same-MEA recomputation before any superiority comparison",
        },
        "historical_defect": {
            "path": "run_meta.json:convergence.bad_evaluations",
            "mechanism": "best restore replaced stopped bad=5 with best bad=0",
            "sole_qualification_defect": passed,
            "historical_files_rewritten": False,
            "fix_commit_integrated": "6a1178eddc60f9d0675552af022770c164275d02",
        },
        "independent_review": {"second_implementation_pass": second["qualification_pass"],
                               "two_implementations_agree": second["primary_and_second_agree"]},
    }


def run(repo: Path, primary_root: Path) -> dict:
    primary = json.loads((primary_root / "requalification.json").read_text())
    second = json.loads((primary_root / "independent_recheck.json").read_text())
    input_manifest = json.loads((primary_root / "input_manifest.json").read_text())
    if not primary["qualification_pass"] or not second["pass"]:
        raise ValueError("cannot finalize failed or disagreeing qualification reports")
    summary = summarize(primary, second)
    if not summary["qualification_pass"]:
        raise ValueError("per-arm qualification mismatch")
    changed = []
    for name, recorded in input_manifest["inputs"].items():
        actual = identity(Path(recorded["path"]))
        if actual["bytes"] != recorded["bytes"] or actual["sha256"] != recorded["sha256"]:
            changed.append(name)
    if changed:
        raise ValueError(f"source inputs changed during requalification: {changed}")
    if subprocess.run(["git", "diff", "--quiet"], cwd=repo).returncode or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
        raise ValueError("requalification repository has uncommitted tracked changes")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    primary_implementation = repo / "tools/gec_nll_v3_requalification_v1.py"
    if identity(primary_implementation)["sha256"] != primary["code_identity"]["implementation"]["sha256"]:
        raise ValueError("primary implementation differs from executed implementation")
    summary["source_artifact_unchanged"] = True
    summary["code_identity"] = {
        "final_head": head,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "primary_execution_head": primary["code_identity"]["head"],
        "primary": identity(primary_implementation),
        "second": identity(repo / "tools/gec_nll_v3_requalification_independent_v1.py"),
        "finalizer": identity(Path(__file__).resolve()),
    }
    write_json(primary_root / "qualification_summary.json", summary)
    artifacts = {name: identity(primary_root / name) for name in (
        "input_manifest.json", "requalification.json", "independent_recheck.json", "qualification_summary.json",
    )}
    write_json(primary_root / "artifact_manifest.json", {"schema_version": SCHEMA, "artifacts": artifacts})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--primary-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run(args.repo.resolve(), args.primary_root)
        print(json.dumps({"status": summary["status"],
                          "per_arm": {arm: row["qualification"] for arm, row in summary["per_arm"].items()}},
                         indent=2, sort_keys=True))
        return 0 if summary["qualification_pass"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"NLL v3 artifact finalization error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
