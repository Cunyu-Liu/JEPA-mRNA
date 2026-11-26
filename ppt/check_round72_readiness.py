#!/usr/bin/env python3
"""Read-only round72 eligibility check including margin exposure and independent-report gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from pptx import Presentation

OUT = Path(__file__).resolve().parent
BUILDER = OUT / "add_external_results_round72.py"
SPEC = importlib.util.spec_from_file_location("round72_readiness_builder", BUILDER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def check_readiness(deck: Path, margin_verification: Path,
                    margin_independent_verification: Path = M.MARGIN_INDEPENDENT_VERIFICATION,
                    panel_registry: Path = M.PANEL_REGISTRY,
                    system_metrics: Path = M.PANEL_SYSTEM_METRICS,
                    panel_independent_verification: Path = M.PANEL_INDEPENDENT_VERIFICATION,
                    panel_labels: Path = M.PANEL_LABELS,
                    longseq_root: Path = M.LONGSEQ_ROOT) -> dict:
    deck = deck.resolve()
    deck_hash = hashlib.sha256(deck.read_bytes()).hexdigest() if deck.is_file() else None
    margin = M.validate_margin(deck, margin_verification, margin_independent_verification)
    external = M.validate_external_raw(panel_registry, system_metrics,
                                       panel_independent_verification, panel_labels)
    longseq = M.validate_longseq(longseq_root)
    M.require(len(Presentation(deck).slides) == 15, "round72 readiness requires exactly 15 slides")
    M.require(hashlib.sha256(deck.read_bytes()).hexdigest() == deck_hash,
              "read-only round72 readiness changed deck")
    return {
        "status": "READY_FOR_ONE_ATOMIC_ROUND72_BUILD", "ready": True, "ppt_modified": False,
        "deck": str(deck), "deck_sha256_before_and_after": deck_hash, "slide_count": 15,
        "added_slide_numbers": [35, 36], "prior_slide_xml_preserved_required": 15,
        "margin_scientific_status": margin["report"]["scientific_status"],
        "margin_verification": M.identity(margin["verification_path"]),
        "margin_independent_verification": M.identity(margin["independent_verification_path"]),
        "exposure_sensitivity": {**margin["exposure_identity"],
                                 "embedded_field": "preregistered_exposure_sensitivity",
                                 "embedded_matches": True,
                                 "cluster_dev_counts": margin["cluster_dev_counts"],
                                 "unknown_ensemble_minus_control": margin["unknown_ensemble_minus_control"]},
        "verified_exposure_registry_identities": len(margin["exposure_registry_identities"]),
        "external_artifact_count": len(external["paths"]),
        "official_reference_panel": {"dataset_id": external["panel"]["dataset_id"],
                                     "sha256": external["panel"]["sha256"],
                                     "full_coverage_systems": external["coverage"]["full_coverage_scored"],
                                     "old_derived_label_ranking_used": False,
                                     "bpfold_replay_author_release_conflated": False},
        "longseq_artifact_count": len(longseq["paths"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=M.DECK)
    parser.add_argument("--margin-verification", type=Path, default=M.MARGIN_VERIFICATION)
    parser.add_argument("--margin-independent-verification", type=Path,
                        default=M.MARGIN_INDEPENDENT_VERIFICATION)
    parser.add_argument("--panel-registry", type=Path, default=M.PANEL_REGISTRY)
    parser.add_argument("--system-metrics", type=Path, default=M.PANEL_SYSTEM_METRICS)
    parser.add_argument("--panel-independent-verification", type=Path,
                        default=M.PANEL_INDEPENDENT_VERIFICATION)
    parser.add_argument("--panel-labels", type=Path, default=M.PANEL_LABELS)
    parser.add_argument("--longseq-root", type=Path, default=M.LONGSEQ_ROOT)
    args = parser.parse_args(argv)
    deck_hash = hashlib.sha256(args.deck.read_bytes()).hexdigest() if args.deck.is_file() else None
    try:
        report = check_readiness(args.deck, args.margin_verification,
                                 args.margin_independent_verification, args.panel_registry,
                                 args.system_metrics, args.panel_independent_verification,
                                 args.panel_labels, args.longseq_root)
    except (M.EligibilityError, OSError, KeyError, TypeError, ValueError) as error:
        report = {"status": "WAITING_FOR_HASH_VERIFIED_MARGIN_EXPOSURE_AND_INDEPENDENT_REPORT",
                  "ready": False, "ppt_modified": False, "reason": str(error),
                  "deck_sha256_before_and_after": deck_hash}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
