import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()
    experiment = "structrfm_full" if args.full_model else "structrfm_frozen"
    run_prefix = "structrfm_jev_s" if args.full_model else "structrfm_frozen_jev_s"
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
            assert abs(macro - result["macro_f1"]) < 1e-12
            expected = result["predictions"][0]
            assert result["predictions"][1] == expected and result["predictions"][2] == expected
            assert len(expected) == result["n"]
            cells.append({"seed": seed, "f1": result["micro"]["f1"],
                          "macro_f1": macro, "tp": tp, "fp": fp, "fn": fn,
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
