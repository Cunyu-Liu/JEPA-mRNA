#!/usr/bin/env python3
"""Build a machine-readable result registry from a verified CHANRG report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARM_META = {
    "s0": ("chanrg_r2d_s0_gt", "single", "CHANRG train; legacy overlap-disclosed", ("s0",)),
    "s1": ("chanrg_r2d_s1_gt", "single", "CHANRG train; legacy overlap-disclosed", ("s1",)),
    "s2": ("chanrg_r2d_s2_gt", "single", "CHANRG train; legacy overlap-disclosed", ("s2",)),
    "ens2": ("chanrg_ens2_gt", "ensemble_2_heads", "CHANRG train; legacy overlap-disclosed", ("s0", "s1")),
    "ens3": ("chanrg_ens3_gt", "ensemble_3_heads", "CHANRG train; legacy overlap-disclosed", ("s0", "s1", "s2")),
    "board_r2d": ("zero_shot_r2d_tr1c", "single", "bpRNA tr1c; legacy board", ()),
    "board_xens2": ("zero_shot_xens2_tr1c", "ensemble", "bpRNA tr1c; legacy board", ()),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_rulings() -> list[dict]:
    return [
        {"id": "archiveii_clean_reference", "verdict": "negative_not_unique_TS2_failure",
         "ours": {"model": "xens2", "micro_f1": 0.7760},
         "reference": {"model": "Kirigami", "micro_f1": 0.7872},
         "evidence": "tables/sota_vs_ours.md; records/DECISION_TRAINING_LOG.md"},
        {"id": "bprna_new_sota_boundary", "verdict": "selected_UFold_win_only_not_all_system_SOTA",
         "ours": {"model": "r2dtr1_two_seed_ensemble", "micro_f1": 0.6132},
         "references": {"UFold": 0.6106, "EternaFold": 0.6854,
                        "ViennaRNA_centroid": 0.6770, "MXfold2": 0.6688, "eFold": 0.6640},
         "evidence": "tables/sota_vs_ours.md; spec/spec.md round-29"},
        {"id": "testsetb_withdrawal", "verdict": "old_zero_shot_claim_withdrawn",
         "reason": "TR0 shares 247 sequences with TestSetB; use clean-181 and retain full-428 separately",
         "clean_reference": {"xens2": 0.8370, "RiNALMo_ft": 0.8177},
         "evidence": "tables/overlap_audit.md; records/DECISION_TRAINING_LOG.md"},
        {"id": "C1-a", "verdict": "FAIL",
         "reason": "RNAformer unrecalibrated ECE 0.0015 outperforms our recalibrated 0.0031",
         "evidence": "spec/benchmark_decision.md; records/DECISION_TRAINING_LOG.md"},
        {"id": "C2_cascade", "verdict": "negative_not_a_contribution",
         "ours": {"casc_at_20k": 0.5011, "flat_ff": 0.5958, "flat_big": 0.6425},
         "evidence": "spec/spec.md; records/DECISION_TRAINING_LOG.md"},
        {"id": "S1_S3_speed_gates", "verdict": "RETIRED",
         "reason": "50x McCaskill and L512<=10ms are not valid acceptance gates; report only matched-protocol measurements",
         "evidence": "spec/spec.md round-29 and S1/S3 table"},
        {"id": "contrastive_A0_A5", "verdict": "NOT_RUN",
         "reason": "Jev RLCD/ensembles are not structure-aware contrastive-learning arms",
         "evidence": "local RNA-SSP contrastive spec v3.0 handoff addendum"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("errors"):
        raise SystemExit("refusing to build a registry from a failed verification")
    if "s2" not in report.get("required_arms", []):
        raise SystemExit("registry requires the complete three-seed report")

    rows = []
    for arm, by_split in report["results"].items():
        model, aggregation, corpus, checkpoint_arms = ARM_META[arm]
        for split, metric in by_split.items():
            source_path = Path(metric["path"])
            source = json.loads(source_path.read_text(encoding="utf-8"))
            legacy = arm.startswith("board_")
            rows.append({
                "model": model,
                "aggregation": aggregation,
                "corpus": corpus,
                "split": split,
                "n_sequences": metric["n_sequences"],
                "metrics": {key: metric[key] for key in
                            ("micro_f1", "macro_f1", "macro_inf", "precision", "recall")},
                "evidence_type": "measured",
                "device": metric["device"],
                "legacy_schema": legacy,
                "decode": source.get("decode", "not_recorded_legacy" if legacy else None),
                "gt": "project canonical-nested; verify per source result" if not legacy
                      else "not_recorded_legacy",
                "scorer": "evaluate_decision strict pair identity" if not legacy
                          else "legacy project scorer; read source script",
                "training_convergence": "step_cap_20000_not_convergence_validated"
                                        if not legacy else "not_recorded_legacy",
                "overlap_status": "legacy overlap disclosed; not clean OOD" if not legacy
                                  else "legacy board; see overlap audit",
                "checkpoint_sha256": {name: report["checkpoint_sha256"][name]["sha256"]
                                      for name in checkpoint_arms},
                "source_result": str(source_path),
                "source_result_sha256": sha256(source_path),
            })
    rows.sort(key=lambda row: (row["model"], row["split"]))
    registry = {
        "schema_version": 1,
        "generated_by": "tools/build_result_registry.py",
        "source_verification": {"path": str(args.report), "sha256": sha256(args.report)},
        "status_note": "CHANRG evidence registry; not a submission-ready global benchmark",
        "contrastive_learning_status": "A0-A5 not run",
        "rows": rows,
        "seed_statistics": report["seed_statistics"],
        "comparisons": report["comparisons"],
        "contract_rulings": contract_rulings(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    assert len(checked["rows"]) == len(rows) == 28
    for row, source in zip(rows, checked["rows"]):
        assert row == source
    temporary.replace(args.out)
    print(f"RESULT_REGISTRY_PASS rows={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
