#!/usr/bin/env python3
"""Freeze the experimentally validated xens2-TurnerOff-MEA candidate identity.

This archives configuration and evidence, not the large score/weight files.  The
original xens2 identity is deliberately left unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


EVIDENCE_FILES = (
    "decode_xens2_vl0/frozen_decode_config.json",
    "decode_xens2_vl0/decode_run_manifest.json",
    "decode_xens2_decision/final_route_decision.json",
    "decode_xens2_decision/decode_candidate_decision.csv",
    "decode_xens2_decision/independent_route_decision_verification.json",
    "decode_xens2_external/route_summary.json",
    "decode_xens2_external/independent_external_raw_dp_verification.json",
    "final_pair_report/final_pair_diagnostic_decision.json",
    "final_pair_report/final_structured_decoder_summary.csv",
    "raw_scores_chunk64/raw_xens2_full_verification.json",
    "final_pair_milestone_manifest.json",
)

EXPECTED_WEIGHTS = {
    "plana0": {
        "path": "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt",
        "sha256": "bd917b6380738e65ce46020eb52ab12333d2cd7a475ab066ed37379dc71828cd",
        "weight": 0.35,
    },
    "plana1": {
        "path": "/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt",
        "sha256": "c10fb303b589346b806ff15ba1f4bb789af1ada3ac78a8010246f6d85c3296c8",
        "weight": 0.35,
    },
    "r2d": {
        "path": "/root/rna_jepa_data/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt",
        "sha256": "47a823514111d116825f5eeef5e39d992abe45b3844d47cf174468bef3b345a3",
        "weight": 0.30,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_candidate(artifact_root: Path, out: Path, weight_specs=None, verify_weights: bool = True) -> dict:
    artifact_root = artifact_root.resolve()
    weight_specs = EXPECTED_WEIGHTS if weight_specs is None else weight_specs
    required = [artifact_root / name for name in EVIDENCE_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze evidence: {missing}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze directory: {out}")

    decision = load_json(artifact_root / "final_pair_report/final_pair_diagnostic_decision.json")
    config = load_json(artifact_root / "decode_xens2_vl0/frozen_decode_config.json")
    run_manifest = load_json(artifact_root / "decode_xens2_vl0/decode_run_manifest.json")
    raw_verification = load_json(artifact_root / "raw_scores_chunk64/raw_xens2_full_verification.json")
    route_verification = load_json(artifact_root / "decode_xens2_decision/independent_route_decision_verification.json")
    external_verification = load_json(artifact_root / "decode_xens2_external/independent_external_raw_dp_verification.json")

    passing = [row for row in decision["passing_candidates"] if row["score_variant"] == "xens2_off" and row["decoder"] == "mea"]
    if len(passing) != 1:
        raise ValueError("exactly one xens2_off MEA passing candidate is required")
    result = passing[0]
    mea = config["config"]["xens2_off"]["mea"]
    if float(mea["temperature"]) != 2.0 or float(mea["gamma"]) != 4.0:
        raise ValueError(f"unexpected frozen xens2_off MEA parameters: {mea}")
    if decision.get("turner_on_passing_candidates") != []:
        raise ValueError("Turner-on boundary changed; refusing to reuse the frozen identity")
    if raw_verification.get("status") != "PASS" or raw_verification.get("n_sequences") != 708:
        raise ValueError("raw fusion verification is not complete")
    if route_verification.get("status") != "PASS" or external_verification.get("status") != "PASS":
        raise ValueError("route or external verification is not PASS")
    if not run_manifest.get("official"):
        raise ValueError("VL0 decode run is not official")

    weights = {}
    for name, spec in weight_specs.items():
        path = Path(spec["path"])
        actual = sha256(path) if verify_weights else spec["sha256"]
        if actual != spec["sha256"]:
            raise ValueError(f"weight SHA256 mismatch for {name}: {actual} != {spec['sha256']}")
        weights[name] = {"path": str(path), "sha256": actual, "ensemble_weight": float(spec["weight"])}

    candidate = {
        "schema_version": 1,
        "candidate_id": "xens2_turneroff_mea_v1",
        "display_name": "xens2-TurnerOff-MEA",
        "status": "FROZEN_EXPERIMENTAL_CANDIDATE",
        "base_identity": "xens2",
        "distinct_from_base_identity": True,
        "production_repair": False,
        "sota_claim": False,
        "further_external_tuning_allowed": False,
        "score": {
            "variant": "xens2_off",
            "turner_prior": "off",
            "formula": "0.35*plana0_off + 0.35*plana1_off + 0.30*r2d_off",
            "weights": weights,
            "raw_root": str(artifact_root / "raw_scores_chunk64/xens2"),
            "raw_verification": "PASS; n=708; max_formula_error=0.0",
        },
        "decoder": {
            "name": "inside_outside_posterior_mea",
            "temperature": float(mea["temperature"]),
            "gamma": float(mea["gamma"]),
            "legal_pair_definition": "AU/UA, GC/CG, GU/UG; j-i>3",
            "structure_class": "nested",
        },
        "selection": {
            "development_dataset": "bprna_vl0",
            "n_development": 196,
            "policy": config["selection_policy"],
            "external_protocol": "eight frozen 64-per-split sensitivity panels; one frozen application",
            "external_is_full_split_sota": False,
        },
        "evidence": result,
        "verification": {
            "route": {"status": route_verification["status"], "n_checks": route_verification.get("n_checks")},
            "external": {"status": external_verification["status"], "n_checks": external_verification.get("n_checks")},
            "raw": raw_verification,
        },
        "boundaries": [
            "This is a new candidate identity, not an update to the original xens2 result.",
            "Evidence is restricted to Turner-off scoring plus MEA; no Turner-on production repair is claimed.",
            "External panels are sensitivity evidence, not full-split SOTA.",
            "Residual frequent-seen-environment FN and CHANRG task-definition ceilings remain unresolved.",
        ],
    }

    evidence_out = out / "evidence"
    evidence_out.mkdir(parents=True, exist_ok=False)
    archived = []
    for name in EVIDENCE_FILES:
        source = artifact_root / name
        target = evidence_out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = sha256(source)
        if sha256(target) != source_hash:
            raise IOError(f"archived evidence hash mismatch: {name}")
        archived.append({"source": str(source), "archive": str(target.relative_to(out)), "sha256": source_hash})

    candidate_path = out / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": candidate["candidate_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": sha256(candidate_path),
        "freeze_code_sha256": sha256(code_path),
        "archived_evidence": archived,
        "large_artifacts": "Referenced in place by absolute path and SHA256; not duplicated.",
    }
    manifest_path = out / "FREEZE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-weight-hash-verification", action="store_true")
    args = parser.parse_args()
    manifest = freeze_candidate(args.artifact_root, args.out, verify_weights=not args.skip_weight_hash_verification)
    print(json.dumps({"status": manifest["status"], "candidate_id": manifest["candidate_id"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
