#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from longseq_acceptance_contract import freeze_targets, validate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate full-length raw-pair predictions; never runs a model.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSON list of per-target records, not summary rows")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="JSON execution protocol; exact and approximate runs are separate files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text())
    protocol = json.loads(args.protocol.read_text())
    if not isinstance(records, list) or not isinstance(protocol, dict):
        raise ValueError("predictions must be a list and protocol must be an object")
    targets = freeze_targets(args.repo.resolve())["targets"]
    report = validate_predictions(records, targets, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "execution_column": report["execution_column"],
                      "coverage": report["coverage"], "micro_pair_recall": report["micro_pair_recall"]}, indent=2))
    if report["status"] != "PASS_FULL_ACCEPTANCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
