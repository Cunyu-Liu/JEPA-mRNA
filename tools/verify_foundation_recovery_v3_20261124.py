#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_foundation")
REPORT = ROOT / "reports/foundation_bprna_new_metrics_verification_v3.json"

EXPECTED = {
    "rinalmo_author_bprna_ft": (51448, 42537, 83736, 0.5474065010373996, 0.38057758314593443, 0.4489961556755058, 0.44123738936817275),
    "nucleicbert_author_secstr_full_ft": (15926, 15754, 119258, 0.5027146464646465, 0.11780979997632857, 0.19088599098667178, 0.18515300147695582),
    "nucleicbert_project_frozen_encoder_tr1c": (60809, 156240, 74375, 0.2801625439416906, 0.4498239436619718, 0.34527713189848763, 0.3611124978084714),
    "structrfm_author_bprna1m_ssp": (73864, 62616, 61320, 0.5412075029308323, 0.5463960231980116, 0.5437893868896873, 0.5338105702966888),
    "rnafm_project_frozen_encoder_crf": (37568, 77317, 97616, 0.3270052661357009, 0.27790271037992664, 0.3004610727439227, 0.3201747208074672),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def main():
    document = json.loads(REPORT.read_text())
    assert document["status"] == "PASS_FIVE_FULL_PREDICTION_IDENTITIES_ERNIE_BLOCKED"
    assert document["dataset"]["n_sequences"] == 5388
    assert document["dataset"]["n_truth_pairs"] == 135184
    assert document["dataset"]["sha256"] == "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
    assert set(document["systems"]) == set(EXPECTED)
    for system, expected in EXPECTED.items():
        record = document["systems"][system]
        metrics = record["metrics"]
        assert sha256(record["artifact"]) == record["artifact_sha256"]
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == expected[:3]
        for key, value in zip(("micro_precision", "micro_recall", "micro_f1", "macro_f1"), expected[3:]):
            close(metrics[key], value)
        assert metrics["n_with_prediction"] == metrics["n_dataset_sequences"] == 5388
        assert metrics["coverage"] == 1.0
        assert metrics["missing_names"] == metrics["extra_names"] == []
        assert record["model_form"] == "single_model_not_ensemble"
        rows = [json.loads(line) for line in (ROOT / f"reports/{system}_bprna_new_per_sequence_v3.jsonl").read_text().splitlines()]
        predictions = [json.loads(line) for line in (ROOT / f"predictions/{system}/bprna_new_v3.jsonl").read_text().splitlines()]
        assert len(rows) == len(predictions) == 5388
        assert len({row["name"] for row in rows}) == len({row["name"] for row in predictions}) == 5388
        assert [row["name"] for row in rows] == [row["name"] for row in predictions]
        counts = tuple(sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        assert counts == expected[:3]
        close(math.fsum(row["f1"] for row in rows) / 5388, expected[-1])
        for row, prediction in zip(rows, predictions):
            assert row["n_predicted_pairs"] == len(prediction["pairs"])
    blocked = document["blocked"]["ernie_rna_author_bprna_new_protocol"]
    assert blocked["metrics"] is None
    assert blocked["coverage"] == 0.0
    assert blocked["n_with_prediction"] == 0
    assert blocked["n_dataset_sequences"] == 5388
    assert document["weights"]["rinalmo_author_bprna_ft"]["status"] == "PASS_OFFICIAL_BYTES_SIZE_MD5_SHA256"
    assert document["weights"]["nucleicbert_author_secstr_full_ft"]["status"] == "PASS_OFFICIAL_BYTES_SIZE_MD5_SHA256"
    exposure = document["exposure_and_homology"]["rinalmo_known_source_audit"]
    assert exposure["exact_overlap_bprna_new"]["exact_overlap_rows_any_known_source"] == 5381
    assert exposure["homology_80pct_bidirectional_coverage_bprna_new"]["detected_by_any_source"] == 5388
    assert exposure["homology_80pct_bidirectional_coverage_bprna_new"]["not_detected_by_any_source"] == 0
    assert not list(ROOT.rglob("*.tmp"))
    assert not list(ROOT.rglob("*.part"))
    core_files = [
        ROOT / "scripts/recover_foundation_baselines_v3_20261124.py",
        ROOT / "scripts/test_recover_foundation_baselines_v3_20261124.py",
        ROOT / "scripts/verify_foundation_recovery_v3_20261124.py",
        ROOT / "reports/foundation_baseline_recovery_v3_report_20261124.md",
        REPORT,
        ROOT / "official_assets/rinalmo/zenodo_15043668.json",
        ROOT / "official_assets/nucleicbert/zenodo_16989562.json",
        ROOT / "official_assets/nucleicbert/secstr.yml",
        ROOT / "official_assets/structrfm/release_v0.0.8.json",
        ROOT / "official_assets/rnafm/ss_prediction_bprna_yml_api.decoded.txt",
        ROOT / "official_assets/ernie_rna/readme.decoded.md",
        ROOT / "official_assets/ernie_rna/bprna_new_testseqs_api.decoded.txt",
    ]
    result = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": [
            "five full prediction identities: exact counts, micro/macro metrics, 5388/5388 coverage, names, normalized pairs, and source hashes",
            "ERNIE-RNA null metrics and zero usable full-project coverage without summary substitution",
            "official RiNALMo/NucleicBERT checkpoint verification status",
            "RiNALMo 5381 exact and 5388 known-source homology exposure counts with limitations retained",
            "no temporary/partial download files and core deliverable hashes",
        ],
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in core_files],
    }
    out = ROOT / "reports/delivery_verification_v3.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "n_core_files": len(core_files), "report": str(out), "sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
