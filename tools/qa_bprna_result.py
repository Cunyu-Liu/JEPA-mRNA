#!/usr/bin/env python3
"""External artifact QA for bpRNA smoke/full result contracts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expect-smoke", action="store_true")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result["status"] == ("SMOKE_PASS" if args.expect_smoke else "PASS")
    assert result["scientific_eligibility"] is (not args.expect_smoke)
    protocol = result["protocol"]
    assert protocol["sample_scope"] == ("engineering_smoke_subset_not_scientific_result" if args.expect_smoke else "full_official_protocol")
    assert protocol["device"] == "cuda" and protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["data_card_sha256"] == "dc370656ebce5d4ed1941dae07c31fc387025ce5d0e38feb4c09f343c4f1c9d0"
    assert protocol["pretrained_sha256"] == "0b3f73b5c80d267f0b7e92655b28f16088e3d5a595c6dae631b52142aa38bcda"
    assert protocol["family"] == "TS0" and len(protocol["official_threshold_grid"]) == 29
    expected_counts = {"train": 10814, "valid": 1300, "test": 1305}
    for split, full_count in expected_counts.items():
        split_protocol = protocol["split"][split]
        assert split_protocol["n_full"] == full_count
        assert split_protocol["n_used"] == (2 if args.expect_smoke else full_count)
    assert protocol["code_sha256"] == sha256("/home/rna_jev/tools/rinalmo_bprna_dual_round33.py")
    assert protocol["cache_meta_sha256"] == sha256(protocol["cache_meta"])
    cache_meta = json.loads(Path(protocol["cache_meta"]).read_text())
    assert cache_meta["extractor_sha256"] == protocol["code_sha256"]
    assert cache_meta["pretrained_sha256"] == protocol["pretrained_sha256"]
    assert cache_meta["pretrained_config_sha256"] == protocol["pretrained_config_sha256"]
    assert cache_meta["preflight_sha256"] == protocol["implementation_sha256"]["preflight"]
    embs_path = Path(protocol["cache_meta"]).with_suffix(".embs.npy")
    assert cache_meta["embedding_content_sha256"] == sha256(embs_path)
    for head_name in ("official_head", "jev_head"):
        runs = result["heads"][head_name]["runs"]
        assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
        values = []
        for run in runs:
            assert run["device"] == "cuda" and run["converged"] and run["early_stopped"]
            assert run["held_out_family"] == "TS0"
            for split, full_count in expected_counts.items():
                assert run[f"n_{split}"] == (2 if args.expect_smoke else full_count)
            assert run["independent_recheck"] == "PASS"
            assert run["epochs_run"] == run["best_epoch"] + 32
            assert len(run["validation_history"]) == run["epochs_run"]
            best = next(point for point in run["validation_history"] if point["epoch"] == run["best_epoch"])
            assert best["validation_bce"] == run["best_validation_bce"]
            assert sha256(run["checkpoint"]) == run["checkpoint_sha256"]
            assert sha256(run["prediction_file"]) == run["prediction_sha256"]
            with Path(run["prediction_file"]).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            test = [record for record in records if record["split"] == "test"]
            valid = [record for record in records if record["split"] == "valid"]
            assert len(test) == run["n_test"] and len(valid) == run["n_valid"]
            assert len({record["id"] for record in test}) == len(test)
            assert all(record["head"] == head_name and record["seed"] == run["seed"] for record in records)
            assert np.isclose(np.mean([record["f1"] for record in test]), run["test_macro_f1"], atol=1e-12)
            assert np.isclose(np.mean([record["f1"] for record in valid]), run["validation_macro_f1"], atol=1e-12)
            values.append(run["test_macro_f1"])
        assert np.isclose(np.mean(values), result["heads"][head_name]["test_macro_f1_mean"], atol=1e-12)
        assert np.isclose(np.std(values, ddof=1), result["heads"][head_name]["test_macro_f1_sample_sd"], atol=1e-12)
    comparison = result["comparison"]
    official = np.array([run["test_macro_f1"] for run in result["heads"]["official_head"]["runs"]])
    jev = np.array([run["test_macro_f1"] for run in result["heads"]["jev_head"]["runs"]])
    assert np.isclose(np.mean(jev - official), comparison["paired_delta_mean"], atol=1e-12)
    print("BPRNA_RESULT_EXTERNAL_QA_PASS", json.dumps({
        "scope": "smoke" if args.expect_smoke else "full",
        "family": protocol["family"], "runs": 10,
        "result_sha256": sha256(args.result),
        "scientific_eligibility": result["scientific_eligibility"]}), flush=True)


if __name__ == "__main__":
    main()
