#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_kirigami_round39 import parse_dot_pairs
from kirigami.utils import _embed_fasta, mat2db as reference_mat2db, outer_concat
from kirigami_fast_decode_round39 import mat2db as fast_mat2db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/kirigami"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--longest", action="store_true")
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    rows = {row["name"]: row for row in (json.loads(line) for line in Path(result["data"]).open() if line.strip())}
    names = [row["name"] for row in result["per_sequence"]
             if ((args.min_length is None or len(rows[row["name"]]["seq"]) >= args.min_length) and
                 (args.max_length is None or len(rows[row["name"]]["seq"]) <= args.max_length))]
    if args.longest:
        selected = sorted(range(len(names)), key=lambda index: len(rows[names[index]]["seq"]), reverse=True)[:args.limit]
    else:
        selected = list(range(min(args.limit, len(names))))
    model = torch.hub.load(str(args.repo), "kirigami", source="local", pretrained=True,
                           checkpoint_path=str(args.weights)).cuda().eval()
    output = []
    for index in selected:
        name = names[index]
        seq = rows[name]["seq"]
        feat = outer_concat(_embed_fasta(seq)).cuda()
        with torch.no_grad():
            raw = model.model(feat)
            processed = model.post_proc(raw, feat)
            processed[processed < model.threshold.item()] = 0
        reference = set(parse_dot_pairs(reference_mat2db(processed)))
        fast = set(parse_dot_pairs(fast_mat2db(processed)))
        union = reference | fast
        row = {"name": name, "length": len(seq), "reference_n": len(reference), "fast_n": len(fast),
               "pair_identity": len(reference & fast) / max(1, len(union)),
               "exact": reference == fast,
               "reference_pairs": sorted(reference), "fast_pairs": sorted(fast)}
        output.append(row)
        print(json.dumps({key: value for key, value in row.items() if not key.endswith("pairs")}, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema_version": 1, "status": "DIAGNOSTIC", "n": len(output),
                                    "exact_count": sum(row["exact"] for row in output),
                                    "sequences": output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
