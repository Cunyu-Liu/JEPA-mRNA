#!/usr/bin/env python3
"""Read-only round-70 eligibility check; never writes a deck or a final JSON."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from pptx import Presentation

OUT = Path(__file__).resolve().parent
BUILDER = OUT / "add_phase0_structured_margin_result_round70.py"
SPEC = importlib.util.spec_from_file_location("round70_readiness_builder", BUILDER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
MODEL_DEV_VERIFICATION = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/model_dev_v2/verification_round3.json")
MODEL_DEV_REGISTRY = OUT.parent / "manifests/foldnar_v2/model_dev_registry.json"


def check_readiness(deck: Path, evaluation_root: Path, model_dev_verification: Path,
                    model_dev_registry: Path) -> dict:
    deck = deck.resolve()
    deck_hash = hashlib.sha256(deck.read_bytes()).hexdigest()
    M.require(len(Presentation(deck).slides) == 13, "real deck is not the preserved 13-slide source")
    verification = M.load_json(model_dev_verification.resolve())
    checks = {row.get("check"): row.get("passed") for row in verification.get("checks", [])}
    M.require(verification.get("status") == "PASS_INDEPENDENT_MANIFEST_LEAKAGE_HASH_RECONCILIATION",
              "model-dev v2 verification is not passing")
    M.require(checks.get("leakage/cross_80_unchecked/tr1c_vs_chanrg_validation") is True,
              "TR1c train versus CHANRG Validation 80% is no longer explicitly UNCHECKED")
    M.require("cross-TR1c/CHANRG 80%" in verification.get("remaining_unchecked", []),
              "cross-source 80% caveat is absent")
    registry_hash = hashlib.sha256(model_dev_registry.resolve().read_bytes()).hexdigest()
    M.require(registry_hash == verification.get("registry_sha256"), "model-dev v2 registry hash changed")
    bundle = M.validate_final_evaluation(evaluation_root)
    result = {
        "status": "READY_FOR_ONE_ATOMIC_ROUND70_BUILD",
        "ready": True, "ppt_modified": False,
        "deck": str(deck), "deck_sha256_before_and_after": deck_hash,
        "slide_count": 13, "evaluation_root": str(bundle["root"]),
        "scientific_status": bundle["evaluation"]["status"],
        "scientific_overall_pass": bundle["evaluation"]["overall_pass"],
        "verified_source_identities": len(bundle["verified_identities"]),
        "cross_source_caveat": M.CROSS_SOURCE_CAVEAT,
        "model_dev_registry_sha256": registry_hash,
    }
    M.require(hashlib.sha256(deck.read_bytes()).hexdigest() == deck_hash, "read-only check changed deck")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=M.DECK)
    parser.add_argument("--evaluation-root", type=Path, default=M.EVALUATION_ROOT)
    parser.add_argument("--model-dev-verification", type=Path, default=MODEL_DEV_VERIFICATION)
    parser.add_argument("--model-dev-registry", type=Path, default=MODEL_DEV_REGISTRY)
    args = parser.parse_args(argv)
    deck_hash = hashlib.sha256(args.deck.read_bytes()).hexdigest() if args.deck.is_file() else None
    try:
        report = check_readiness(args.deck, args.evaluation_root,
                                 args.model_dev_verification, args.model_dev_registry)
    except (M.EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        report = {"status": "WAITING_FOR_FINAL_HASH_VERIFIED_FOUR_ARM_EVAL", "ready": False,
                  "ppt_modified": False, "reason": str(error),
                  "deck_sha256_before_and_after": deck_hash}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
