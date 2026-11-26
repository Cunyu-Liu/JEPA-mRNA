#!/usr/bin/env python3
"""Independent consistency checks for the frozen GEC residual audit artifacts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--independent-reeval", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    errors = []
    checks = {}
    for arm in ("ensemble_context", "destroy_interval_shuffle"):
        summary = json.loads((root / f"{arm}_summary.json").read_text())
        if summary["scope"] != {
            "panel_id": "tr1c_cluster80_dev_v1", "records": 9143,
            "external_test_sets": False, "limited_smoke": False,
        }:
            errors.append(f"{arm}: scope")
        invariants = summary["invariants"]
        if invariants["s1_equals_s0_plus_delta_max_abs"] != 0.0:
            errors.append(f"{arm}: S1 identity")
        if invariants["zero_init_delta_max_abs"] != 0.0:
            errors.append(f"{arm}: zero init")
        if invariants["historical_mea_prediction_mismatches"] != 0:
            errors.append(f"{arm}: prediction replay")
        if invariants["native_nll_abs_difference"] > 1e-6:
            errors.append(f"{arm}: NLL replay")
        raw = root / f"{arm}.jsonl.gz"
        if sha256(raw) != summary["raw_diagnostics"]["sha256"]:
            errors.append(f"{arm}: raw hash")
        counts = Counter()
        families = Counter()
        spans = Counter()
        probe_count = 0
        sequence_count = 0
        for sequence in rows(raw):
            sequence_count += 1
            probe_count += sequence["stopgrad_probe"] is not None
            for pair in sequence["true_pairs"]:
                counts[pair["class"]] += 1
                families[str(pair["family"])] += 1
                spans[pair["span"]] += 1
        if sequence_count != 9143:
            errors.append(f"{arm}: raw sequence count")
        for label, count in counts.items():
            if summary["groups"]["true_all"][label] != count:
                errors.append(f"{arm}: class count {label}")
        for family, count in families.items():
            if summary["families"][family]["truth_pairs"] != count:
                errors.append(f"{arm}: family count {family}")
        for span, count in spans.items():
            if summary["groups"][f"span:{span}"]["truth_pairs"] != count:
                errors.append(f"{arm}: span count {span}")
        if probe_count != summary["stopgrad_forward_sensitivity"]["probe_count"]:
            errors.append(f"{arm}: stopgrad probe count")
        checks[arm] = {
            "sequences": sequence_count, "true_pairs": sum(counts.values()),
            "classes": dict(counts), "families": len(families), "stopgrad_probes": probe_count,
            "cuda_allocated_peak_bytes": summary["execution"]["cuda_allocated_peak_bytes"],
            "native_nll_abs_difference": invariants["native_nll_abs_difference"],
        }
    independent = json.loads(Path(args.independent_reeval).read_text())
    expected = {
        "micro_pair_f1": -0.009355788655,
        "macro_pair_f1": -0.009706658361,
        "micro_stem_f1": -0.023240929251,
        "macro_stem_f1": -0.015810540044,
    }
    observed = independent["comparisons"]["matched_control"]["metrics"]
    for metric, value in expected.items():
        if abs(observed[metric]["delta"] - value) > 5e-13:
            errors.append(f"independent seed0 failure changed: {metric}")
    if independent["overall_pass"] is not False:
        errors.append("independent seed0 overall pass changed")
    checks["independent_seed0_failure"] = {
        metric: observed[metric]["delta"] for metric in expected
    }
    comparison = json.loads((root / "comparison.json").read_text())
    if comparison["scope"] != "registered TR1c cluster80 dev only; no external test":
        errors.append("comparison scope")
    report = {
        "schema_version": "foldnar.gec_residual_error_audit.verification.v1",
        "status": "PASS" if not errors else "FAIL", "errors": errors, "checks": checks,
        "comparison_sha256": sha256(root / "comparison.json"),
        "independent_reeval_sha256": sha256(Path(args.independent_reeval)),
        "external_test_sets": False,
    }
    temporary = root / f"verification.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "verification.json")
    print(json.dumps(report, indent=2, allow_nan=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
