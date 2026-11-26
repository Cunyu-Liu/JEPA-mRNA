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
BUILDER = OUT / "add_phase0_structured_margin_result_round71.py"
SPEC = importlib.util.spec_from_file_location("round71_readiness_builder", BUILDER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
MODEL_DEV_REGISTRY = OUT.parent / "manifests/foldnar_v2/model_dev_registry.json"


def check_readiness(deck: Path, evaluation_root: Path, cross_source_root: Path,
                    model_dev_registry: Path) -> dict:
    deck = deck.resolve()
    deck_hash = hashlib.sha256(deck.read_bytes()).hexdigest()
    M.require(deck_hash == M.SOURCE_DECK_SHA256, "deck is not the frozen crossover round70 source")
    M.require(len(Presentation(deck).slides) == 14, "real deck is not the preserved 14-slide source")
    cross_source = M.validate_cross_source_boundary(cross_source_root)
    registry = model_dev_registry.resolve()
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    recorded_registry = cross_source["summary"].get("model_dev_v2_registry", {})
    M.require(recorded_registry.get("sha256") == registry_hash
              and recorded_registry.get("bytes") == registry.stat().st_size
              and cross_source["summary"].get("model_dev_v2_registry_modified") is False,
              "model-dev v2 registry changed or is not bound to the cross-source summary")
    bundle = M.validate_final_evaluation(evaluation_root)
    result = {
        "status": "READY_FOR_ONE_ATOMIC_ROUND71_BUILD",
        "ready": True, "ppt_modified": False,
        "deck": str(deck), "deck_sha256_before_and_after": deck_hash,
        "slide_count": 14, "evaluation_root": str(bundle["root"]),
        "scientific_status": bundle["evaluation"]["status"],
        "scientific_overall_pass": bundle["evaluation"]["overall_pass"],
        "verified_source_identities": len(bundle["verified_identities"]),
        "exposure_sensitivity": {
            **bundle["exposure"]["identity"], "embedded_field": "preregistered_exposure_sensitivity",
            "embedded_matches": True, "cluster_dev_counts": M.EXPOSURE_COUNTS,
            "unknown_ensemble_minus_control": bundle["exposure"]["data"]["strata"]["NOT_DETECTED_UNKNOWN"]["ensemble_vs_controls"]["matched_control"],
        },
        "cross_source_boundary": {
            "code_commit": cross_source["code_commit"],
            "summary_sha256": cross_source["summary_identity"]["sha256"],
            "display_boundary": cross_source["display_boundary"],
            "gec_state": cross_source["gec_qualification"]["state_after_this_screen"],
        },
        "model_dev_registry_sha256": registry_hash,
    }
    M.require(hashlib.sha256(deck.read_bytes()).hexdigest() == deck_hash, "read-only check changed deck")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=M.DECK)
    parser.add_argument("--evaluation-root", type=Path, default=M.EVALUATION_ROOT)
    parser.add_argument("--cross-source-root", type=Path, default=M.CROSS_SOURCE_ROOT)
    parser.add_argument("--model-dev-registry", type=Path, default=MODEL_DEV_REGISTRY)
    args = parser.parse_args(argv)
    deck_hash = hashlib.sha256(args.deck.read_bytes()).hexdigest() if args.deck.is_file() else None
    try:
        report = check_readiness(args.deck, args.evaluation_root,
                                 args.cross_source_root, args.model_dev_registry)
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
