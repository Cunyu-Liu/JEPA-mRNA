import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length_breakdown(result):
    from structrfm_metrics_round48 import length_bucket
    buckets = [{"n": 0, "tp": 0, "fp": 0, "fn": 0, "f1_values": []} for _ in range(4)]
    for row in result["per_sequence"]:
        bucket = buckets[length_bucket(row["length"])]
        bucket["n"] += 1
        for key in ("tp", "fp", "fn"):
            bucket[key] += row[key]
        bucket["f1_values"].append(row["f1"])
    output = []
    for bucket in buckets:
        denominator = 2 * bucket["tp"] + bucket["fp"] + bucket["fn"]
        output.append({"n": bucket["n"], "tp": bucket["tp"], "fp": bucket["fp"],
                       "fn": bucket["fn"], "micro_f1": 2 * bucket["tp"] / denominator if denominator else None,
                       "macro_f1": float(np.mean(bucket["f1_values"])) if bucket["n"] else None})
    return output


def span_breakdown(result, minimum_span):
    key = f"span_{minimum_span}"
    counts = {name: sum(row[key][name] for row in result["per_sequence"])
              for name in ("tp", "fp", "fn")}
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return {**counts, "micro_f1": 2 * counts["tp"] / denominator if denominator else None,
            "macro_f1": float(np.mean([row[key]["f1"] for row in result["per_sequence"]]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--run-prefix", default="")
    args = parser.parse_args()
    experiment = args.experiment or ("structrfm_full" if args.full_model else "structrfm_frozen")
    run_prefix = args.run_prefix or ("structrfm_jev_s" if args.full_model else "structrfm_frozen_jev_s")
    clean = args.root / "data/structrfm_pretrain_clean"
    splits = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1",
              "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb",
              "rfam_fam", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
    results = {}
    for split in splits:
        audit = json.loads((clean / f"{split}.audit.json").read_text())
        cells = []
        for seed in range(3):
            path = args.root / f"eval_decision/{experiment}/s{seed}/{split}.json"
            result = json.loads(path.read_text())
            checkpoint = args.root / f"runs/{run_prefix}{seed}/best.pt"
            assert result["device"] == "cuda"
            assert result["repeats"] == 3 and len(result["predictions"]) == 3
            if args.full_model:
                assert result["encoder_load"]["strict_identity"]
            assert result["data_sha256"] == audit["clean_sha256"]
            assert result["checkpoint_sha256"] == sha256(checkpoint)
            assert result["raw_n"] == audit["clean_n"]
            assert result["n"] + result["n_long_unsupported"] == result["raw_n"]
            assert result["status"] in {"PASS", "PARTIAL_STRUCTRFM_POSITION_LIMIT"}
            tp = sum(row["tp"] for row in result["per_sequence"])
            fp = sum(row["fp"] for row in result["per_sequence"])
            fn = sum(row["fn"] for row in result["per_sequence"])
            assert (tp, fp, fn) == tuple(result["micro"][key] for key in ("tp", "fp", "fn"))
            macro = float(np.mean([row["f1"] for row in result["per_sequence"]]))
            macro_inf = float(np.mean([row["inf"] for row in result["per_sequence"]]))
            assert abs(macro - result["macro_f1"]) < 1e-12
            assert abs(macro_inf - result["macro_inf"]) < 1e-12
            expected_inf = float(np.sqrt(result["micro"]["precision"] * result["micro"]["recall"]))
            assert abs(expected_inf - result["micro_inf"]) < 1e-12
            assert result["calibration"]["n"] > 0
            assert all(np.isfinite(result["calibration"][key]) for key in ("ece", "brier", "nll"))
            assert all(0.0 <= row["structure_conf"] <= 1.0 for row in result["per_sequence"])
            expected = result["predictions"][0]
            assert result["predictions"][1] == expected and result["predictions"][2] == expected
            assert len(expected) == result["n"]
            cells.append({"seed": seed, "f1": result["micro"]["f1"],
                          "macro_f1": macro, "tp": tp, "fp": fp, "fn": fn,
                          "micro_inf": result["micro_inf"], "macro_inf": macro_inf,
                          "ece": result["calibration"]["ece"], "brier": result["calibration"]["brier"],
                          "nll": result["calibration"]["nll"], "length_buckets": length_breakdown(result),
                          "span_30": span_breakdown(result, 30), "span_100": span_breakdown(result, 100),
                          "precision": result["micro"]["precision"],
                          "recall": result["micro"]["recall"],
                          "result_sha256": sha256(path), "checkpoint_sha256": result["checkpoint_sha256"],
                          "n": result["n"], "n_long_unsupported": result["n_long_unsupported"],
                          "status": result["status"]})
        f1_values = [cell["f1"] for cell in cells]
        macro_values = [cell["macro_f1"] for cell in cells]
        results[split] = {"n": audit["clean_n"], "cells": cells,
                          "f1_mean": float(np.mean(f1_values)),
                          "f1_sd": float(np.std(f1_values)),
                          "macro_f1_mean": float(np.mean(macro_values)),
                          "macro_f1_sd": float(np.std(macro_values)),
                          "macro_inf_mean": float(np.mean([cell["macro_inf"] for cell in cells])),
                          "micro_inf_mean": float(np.mean([cell["micro_inf"] for cell in cells])),
                          "ece_mean": float(np.mean([cell["ece"] for cell in cells])),
                          "brier_mean": float(np.mean([cell["brier"] for cell in cells])),
                          "nll_mean": float(np.mean([cell["nll"] for cell in cells])),
                          "n_long_unsupported": sum(cell["n_long_unsupported"] for cell in cells)}
    status = "FULL" if args.full_model else "FROZEN"
    summary = {"status": f"PASS_STRUCTRFM_{status}_ABC_13_SPLITS_3_SEEDS_3_REPEATS",
               "protocol": "own-clean; each seed independently decoded three times; not an ensemble",
               "splits": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": summary["status"], "out": str(args.out),
                      "sha256": sha256(args.out), "splits": results}, indent=2))


if __name__ == "__main__":
    main()
