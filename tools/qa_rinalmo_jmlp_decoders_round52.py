import argparse
import hashlib
import json
import statistics
from pathlib import Path

from qa_rinalmo_jmlp_benchmark_round52 import SPLITS, sha256_file, validate_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["greedy", "turner0"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    for seed in (42, 2026, 3407):
        for split in SPLITS:
            data = (Path("/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl") if split == "rfam_fam"
                    else root / "data/rinalmo_tr0_clean_round52" / f"{split}.jsonl")
            path = root / f"evals/round52/jmlp_{args.mode}" / f"rinalmo_jmlp_length_s{seed}_{split}.json"
            checkpoint = root / "runs" / f"round52_rinalmo_length_s{seed}/best.pt"
            results[f"{seed}.{split}"] = validate_result(
                path, checkpoint, data, split, seed, mode=args.mode)
    aggregates = {}
    for split in SPLITS:
        cells = [results[f"{seed}.{split}"] for seed in (42, 2026, 3407)]
        values = [cell["micro"]["f1"] for cell in cells]
        aggregates[split] = {"complete_all_three": all(cell["complete"] for cell in cells),
                             "seed_micro_f1": values, "mean_micro_f1": statistics.mean(values),
                             "population_sd_micro_f1": statistics.pstdev(values),
                             "raw_n": [cell["raw_n"] for cell in cells],
                             "n_long_unsupported": [cell["n_long_unsupported"] for cell in cells]}
    complete = all(cell["complete"] for cell in results.values())
    result = {
        "status": f"PASS_JMLP_{args.mode.upper()}_MAIN8_RFAM_INDEPENDENT_QA" if complete else f"PARTIAL_JMLP_{args.mode.upper()}_MAIN8_RFAM_INDEPENDENT_QA",
        "scientific_conclusion": False, "complete": complete, "mode": args.mode,
        "results": results, "aggregates": aggregates, "qa_sha256": sha256_file(__file__),
        "limitations": [
            "Decoder/physical-prior ablation only; no test set was used for tuning, early stopping, calibration or selection.",
            "Rfam retains pretraining-contact risk; concurrent wall times are not formal speed results.",
            "Long-sequence failures and empty buckets are retained; no SOTA claim is licensed.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "complete": complete,
                      "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
