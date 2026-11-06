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
    args = parser.parse_args()
    clean = args.root / "data/structrfm_pretrain_clean"
    splits = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1",
              "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb",
              "rfam_fam", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
    results = {}
    for split in splits:
        audit = json.loads((clean / f"{split}.audit.json").read_text())
        cells = []
        for seed in range(3):
            path = args.root / f"eval_decision/structrfm_frozen/s{seed}/{split}.json"
            result = json.loads(path.read_text())
            checkpoint = args.root / f"runs/structrfm_frozen_jev_s{seed}/best.pt"
            assert result["repeats"] == 3 and len(result["predictions"]) == 3
            assert result["data_sha256"] == audit["clean_sha256"]
            assert result["checkpoint_sha256"] == sha256(checkpoint)
            assert result["raw_n"] == audit["clean_n"]
            assert result["n"] + result["n_long_unsupported"] == result["raw_n"]
            assert result["status"] in {"PASS", "PARTIAL_STRUCTRFM_POSITION_LIMIT"}
            tp = sum(row["tp"] for row in result["per_sequence"])
            fp = sum(row["fp"] for row in result["per_sequence"])
            fn = sum(row["fn"] for row in result["per_sequence"])
            assert (tp, fp, fn) == tuple(result["micro"][key] for key in ("tp", "fp", "fn"))
            expected = result["predictions"][0]
            assert result["predictions"][1] == expected and result["predictions"][2] == expected
            assert len(expected) == result["n"]
            cells.append({"seed": seed, "f1": result["micro"]["f1"],
                          "precision": result["micro"]["precision"],
                          "recall": result["micro"]["recall"],
                          "result_sha256": sha256(path), "checkpoint_sha256": result["checkpoint_sha256"],
                          "n": result["n"], "n_long_unsupported": result["n_long_unsupported"],
                          "status": result["status"]})
        f1_values = [cell["f1"] for cell in cells]
        results[split] = {"n": audit["clean_n"], "cells": cells,
                          "f1_mean": float(np.mean(f1_values)),
                          "f1_sd": float(np.std(f1_values)),
                          "n_long_unsupported": sum(cell["n_long_unsupported"] for cell in cells)}
    summary = {"status": "PASS_STRUCTRFM_FROZEN_ABC_13_SPLITS_3_SEEDS_3_REPEATS",
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
