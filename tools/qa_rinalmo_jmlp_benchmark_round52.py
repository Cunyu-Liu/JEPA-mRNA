import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from eval.ss.metrics import PairLevelMetrics
from qa_rinalmo_vl0_structure_round52 import micro_metrics
from rnajepa.harness import valid_pair_mask


SPLITS = ("bprna_ts0", "bprna_new", "archiveii_embok_clean", "ref_pdb_ts1",
          "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_result(path, checkpoint, data, split, seed):
    path, checkpoint, data = Path(path), Path(checkpoint), Path(data)
    result = json.loads(path.read_text())
    assert result["device"] == "cuda" and result["decoder"] == "nussinov"
    assert result["repeats"] == 3 and result["score_forward_count_per_sequence"] == 1
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["data_sha256"] == sha256_file(data)
    run_meta = json.loads((checkpoint.parent / "run_meta.json").read_text())
    assert run_meta["arm"] == f"round52_rinalmo_length_s{seed}" and run_meta["seed"] == seed
    assert run_meta["status"] == "completed" and run_meta["converged"]
    assert run_meta["best_step"] == result["checkpoint_step"] and run_meta["scorer"] == "pairmlp"
    records = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert result["raw_n"] == len(records)
    complete = result["status"] == "PASS"
    assert complete or result["status"] == "PARTIAL_RINALMO_CUDA_FAILURE"
    assert result["n"] == len(result["per_sequence"])
    assert result["n"] + result["n_long_unsupported"] == len(records)
    assert complete == (result["n_long_unsupported"] == 0)
    assert all(item["index"] == result["n"] for item in result["long_unsupported"])
    assert len(result["predictions"]) == 3
    assert all(len(repeat) == result["n"] for repeat in result["predictions"])
    tp = fp = fn = 0
    per_sequence_f1 = []
    for index, (source, row) in enumerate(zip(records, result["per_sequence"])):
        sequence = source["seq"].upper().replace("T", "U")
        ground_truth = [tuple(pair) for pair in source["pairs"]]
        assert row["name"] == source.get("name", "") and row["length"] == len(sequence)
        repeat_pairs = []
        for repeat in result["predictions"]:
            prediction = repeat[index]
            assert prediction["name"] == row["name"]
            repeat_pairs.append([tuple(pair) for pair in prediction["pred_pairs"]])
        assert repeat_pairs[0] == repeat_pairs[1] == repeat_pairs[2]
        metrics = PairLevelMetrics.from_pairs(repeat_pairs[0], ground_truth,
                                              L=len(sequence), mask=valid_pair_mask(sequence))
        assert (row["tp"], row["fp"], row["fn"]) == (metrics.tp, metrics.fp, metrics.fn)
        assert math.isclose(row["f1"], metrics.f1, rel_tol=0, abs_tol=1e-12)
        tp += metrics.tp
        fp += metrics.fp
        fn += metrics.fn
        per_sequence_f1.append(metrics.f1)
    micro = micro_metrics(tp, fp, fn)
    for key, value in micro.items():
        assert math.isclose(value, result["micro"][key], rel_tol=0, abs_tol=1e-12)
    macro = math.fsum(per_sequence_f1) / len(per_sequence_f1) if per_sequence_f1 else 0.0
    if per_sequence_f1:
        assert math.isclose(macro, result["macro_f1"], rel_tol=0, abs_tol=1e-12)
    return {"file": path.name, "split": split, "seed": seed, "complete": complete,
            "raw_n": len(records), "n": result["n"], "n_long_unsupported": result["n_long_unsupported"],
            "long_unsupported": result["long_unsupported"], "micro": micro, "macro_f1": macro,
            "checkpoint_sha256": result["checkpoint_sha256"], "result_sha256": sha256_file(path),
            "repeat_predictions_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    control = json.loads((root / "audits/round52/rinalmo_jmlp_training_qa.json").read_text())
    assert control["status"] == "PASS_JMLP_THREE_SEED_TRAINING_QA" and control["base"] == "length"
    manifest_path = root / "audits/round52/rinalmo_tr0_clean_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "PASS_RINALMO_TR0_CLEAN_ASSETS"
    clean_sha256 = {row["split"]: row["clean_sha256"] for row in manifest["rows"]}
    results = {}
    for seed in (42, 2026, 3407):
        for split in SPLITS:
            data = (Path("/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl") if split == "rfam_fam"
                    else root / "data/rinalmo_tr0_clean_round52" / f"{split}.jsonl")
            if split != "rfam_fam":
                assert sha256_file(data) == clean_sha256[split]
            path = root / "evals/round52/jmlp_benchmark" / f"rinalmo_jmlp_length_s{seed}_{split}.json"
            checkpoint = root / "runs" / f"round52_rinalmo_length_s{seed}/best.pt"
            results[f"{seed}.{split}"] = validate_result(path, checkpoint, data, split, seed)
    aggregates = {}
    for split in SPLITS:
        cells = [results[f"{seed}.{split}"] for seed in (42, 2026, 3407)]
        complete = all(cell["complete"] for cell in cells)
        values = [cell["micro"]["f1"] for cell in cells]
        aggregates[split] = {"complete_all_three": complete, "seed_micro_f1": values,
                             "mean_micro_f1": statistics.mean(values),
                             "population_sd_micro_f1": statistics.pstdev(values),
                             "raw_n": [cell["raw_n"] for cell in cells],
                             "n_long_unsupported": [cell["n_long_unsupported"] for cell in cells]}
    complete = all(cell["complete"] for cell in results.values())
    result = {
        "status": "PASS_JMLP_MAIN8_RFAM_INDEPENDENT_QA" if complete else "PARTIAL_JMLP_MAIN8_RFAM_INDEPENDENT_QA",
        "scientific_conclusion": False, "complete": complete,
        "results": results, "aggregates": aggregates,
        "control_sha256": sha256_file(root / "audits/round52/rinalmo_jmlp_training_qa.json"),
        "clean_manifest_sha256": sha256_file(manifest_path), "qa_sha256": sha256_file(__file__),
        "limitations": [
            "Frozen-model descriptive evaluation only; no test set was used for tuning, early stopping, calibration or arm selection.",
            "Rfam is the raw official set with RiNALMo pretraining-contact risk, not a clean OOD guarantee.",
            "Concurrent wall times are not formal speed results; calibration/confidence fields are not independently accepted here.",
            "Long-sequence failures and partial cells are retained and never converted to full-coverage or SOTA claims.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "complete": complete, "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
