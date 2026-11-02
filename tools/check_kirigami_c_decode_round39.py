#!/usr/bin/env python3
import json
import time
from pathlib import Path

from kirigami.utils import dict2db as reference_dict2db
from kirigami_fast_decode_round39 import dict2db as fast_dict2db


def main():
    root = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark/kirigami/raw")
    for split in ("bprna_ts0", "bprna_new"):
        result = json.loads((root / f"{split}.json").read_text())
        records = {row["name"]: row["seq"] for row in
                   (json.loads(line) for line in Path(result["data"]).open() if line.strip())}
        started = time.perf_counter()
        exact = total = 0
        for row in result["per_sequence"]:
            pairs = {tuple(pair) for pair in row["pred_pairs"]}
            reference = reference_dict2db(pairs, len(records[row["name"]]))
            fast = fast_dict2db(pairs, len(records[row["name"]]))
            exact += fast == reference
            total += 1
            if fast != reference and total <= 3:
                print("MISMATCH", split, row["name"], reference, fast)
        print(split, "n", total, "dot_bracket_exact", exact, "dot_bracket_identity", exact / total,
              "elapsed", time.perf_counter() - started, flush=True)


if __name__ == "__main__":
    main()
