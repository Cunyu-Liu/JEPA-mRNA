#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    output = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output.append((str(row.get("name") or row.get("id") or len(output)),
                               {tuple(pair) for pair in row.get("gt_pairs") or row.get("pairs") or []}))
    return output


def validate_result(path, data_path, audit_path, checkpoint_sha):
    result = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    assert result["model"] == "plana"
    assert result["status"] in ("PASS", "EMPTY_AFTER_DELEAK")
    protocol = result["protocol"]
    assert Path(protocol["data"]) == data_path
    assert Path(protocol["clean_audit"]) == audit_path
    assert protocol["data_sha256"] == sha256(data_path) == audit["clean_sha256"]
    assert protocol["n_sequences"] == audit["clean_n"]
    assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
    assert protocol["device_type"] == "cuda" and protocol["cached_embeddings"] is False
    assert protocol["sample_scope"] == "full_source_file" and protocol["n_length_filtered"] == 0
    assert protocol["model_report"]["weights_sha256"] == checkpoint_sha
    expected = records(data_path)
    observed = result["predictions"]
    assert len(observed) == len(expected)
    tp = fp = fn = 0
    macro = []
    canonical = []
    for (name, gt), prediction in zip(expected, observed):
        assert prediction["name"] == name
        pred = {tuple(pair) for pair in prediction["pairs"]}
        assert len(pred) == len(prediction["pairs"])
        assert all(isinstance(i, int) and isinstance(j, int) and i < j for i, j in pred)
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
        macro.append(2 * len(pred & gt) / max(1, 2 * len(pred & gt) + len(pred - gt) + len(gt - pred)))
        canonical.append({"name": name, "pairs": sorted(map(list, pred))})
    prediction_sha = hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    if not expected:
        assert result["status"] == "EMPTY_AFTER_DELEAK" and result["runs"] == []
        return {"clean_n": 0, "status": "EMPTY_AFTER_DELEAK", "tp": 0, "fp": 0, "fn": 0,
                "strict_micro_f1": None, "predictions_sha256": prediction_sha,
                "result_sha256": sha256(path)}
    assert result["status"] == "PASS" and len(result["runs"]) == 3
    assert len(result["f1_runs"]) == 3 and len(set(result["f1_runs"])) == 1
    for run in result["runs"]:
        assert run["n_sequences"] == len(expected)
        assert (run["tp"], run["fp"], run["fn"]) == (tp, fp, fn)
        assert math.isclose(run["strict_micro_f1"], f1, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(run["strict_macro_f1"], statistics.mean(macro), rel_tol=0, abs_tol=1e-12)
        assert run["predictions_sha256"] == prediction_sha
    return {"clean_n": len(expected), "status": "PASS", "tp": tp, "fp": fp, "fn": fn,
            "strict_micro_f1": f1, "strict_macro_f1": statistics.mean(macro),
            "predictions_sha256": prediction_sha, "result_sha256": sha256(path)}


def summary(values):
    if any(value is None for value in values):
        return None
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    per_split = {}
    missing = []
    n_complete = 0
    for split in SPLITS:
        rows = []
        for seed in range(3):
            arm = f"xens2_plana_distill_s{seed}"
            qa = json.loads((args.root / "qa" / f"{arm}_vl0" / "qa_summary.json").read_text())
            meta = json.loads((args.root / "runs" / arm / "run_meta.json").read_text())
            assert qa["status"] == "PASS" and meta["converged"] and meta["early_stopped"]
            checkpoint_sha = qa["checkpoint_sha256"]
            result_path = args.root / f"plana_s{seed}" / "clean" / f"{split}.json"
            if not result_path.is_file():
                missing.append(str(result_path))
                continue
            row = validate_result(result_path, args.clean_root / f"{split}.jsonl",
                                  args.clean_root / f"{split}.audit.json", checkpoint_sha)
            row.update({"seed": seed, "checkpoint_sha256": checkpoint_sha})
            rows.append(row)
        if len(rows) == 3:
            n_complete += 1
            per_split[split] = {"seeds": rows, "clean_n": {row["clean_n"] for row in rows}.pop(),
                                "strict_micro_f1": summary([row["strict_micro_f1"] for row in rows])}
        elif rows:
            per_split[split] = {"seeds": rows, "status": "INCOMPLETE_THREE_SEED"}
    complete = n_complete == len(SPLITS) and not missing
    output = {"schema_version": 1,
              "status": "PASS_8_SPLIT_3_TRAINING_SEEDS" if complete else "PARTIAL_NOT_SCIENTIFIC_CONCLUSION",
              "model": "single_plana_xens2_KL_student",
              "independent_evaluation": "three trained seeds; three deterministic inference repeats per seed",
              "n_training_seeds": 3, "n_inference_repeats_per_seed": 3,
              "complete_split_count": n_complete, "required_split_count": len(SPLITS),
              "concurrent_throughput_eligible_for_speed_claim": False,
              "per_split": per_split, "missing": missing}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    checked = json.loads(temporary.read_text())
    assert checked["complete_split_count"] == n_complete
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "complete_split_count": n_complete,
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
