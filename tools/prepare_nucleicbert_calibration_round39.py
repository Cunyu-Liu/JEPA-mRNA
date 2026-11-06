#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_nucleicbert_ft_round39 import validate as validate_ft
from aggregate_nucleicbert_frozen_round39 import validate as validate_frozen
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from rnajepa.harness import valid_pair_mask
from calibrate_xens2_round39 import confidence_for_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--system", choices=["nucleicbert_ft", "nucleicbert_frozen"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    audit = validate_ft(args.source, False) if args.system == "nucleicbert_ft" else validate_frozen(args.source, clean=True)
    if source["status"] == "EMPTY_AFTER_DELEAK":
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "system": args.system,
                   "source_result": str(args.source), "result_sha256": sha256(args.source)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return
    rows = [json.loads(line) for line in Path(source["data"]).read_text().splitlines() if line.strip()]
    assert len(rows) == source["n_sequences"]
    assert sha256(Path(source["data"])) == source["data_sha256"]
    assert sha256(Path(source["weights"])) == source["weights_sha256"]
    names, offsets, probabilities, labels = [], [0], [], []
    per_sequence = []
    with np.load(source["pair_probabilities"], allow_pickle=False) as archive:
        source_probabilities = archive["probabilities"]
        for index, (row, prediction) in enumerate(zip(rows, source["predictions"])):
            name = row["name"]
            length = min(len(row["seq"]), 598)
            mask = valid_pair_mask(row["seq"][:length])
            probability = source_probabilities[archive["offsets"][index]:archive["offsets"][index + 1]]
            assert len(probability) == np.triu(mask, k=1).sum()
            truth = {(i, j) for i, j in row.get("gt_pairs") or row.get("pairs") or [] if 0 <= i < j < length}
            expected = np.zeros((length, length), dtype=np.uint8)
            for i, j in truth:
                expected[i, j] = expected[j, i] = 1
            label = expected[np.triu(mask, k=1)]
            assert len(label) == len(probability)
            pred = {tuple(pair) for pair in prediction["pairs"]}
            f1 = 2 * len(pred & truth) / max(1, 2 * len(pred & truth) + len(pred - truth) + len(truth - pred))
            confidence = confidence_for_pairs(pred, probability, mask)
            names.append(name)
            offsets.append(offsets[-1] + len(probability))
            probabilities.append(probability)
            labels.append(label)
            per_sequence.append({"name": name, "structure_f1": f1,
                                 "mean_selected_pair_probability": confidence,
                                 "pred_pairs": prediction["pairs"],
                                 "n_predicted_pairs": len(pred), "n_gt_pairs": len(truth)})
    saved = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
             "probabilities": np.concatenate(probabilities), "labels": np.concatenate(labels)}
    payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": args.system,
               "data": source["data"], "data_sha256": source["data_sha256"], "n_sequences": len(rows),
               "clean_audit": source["clean_audit"], "weights": source["weights"],
               "weights_sha256": source["weights_sha256"],
               "protocol": {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                            "evaluation_scope": source["evaluation_scope"], "candidate_pairs": "strict upper valid_pair_mask",
                            "n_bins": 10, "structure_confidence": "mean native probability of decoded pairs"},
               "n_truncated_sequences": source["n_truncated_sequences"],
               "source_result": str(args.source), "source_result_sha256": sha256(args.source),
               "source_probabilities": source["pair_probabilities"],
               "source_probabilities_sha256": source["pair_probabilities_sha256"],
               "f1_runs": [run["strict_micro_f1"] for run in source["repeats"]],
               "calibration": metrics(saved["probabilities"], saved["labels"]),
               "confidence": selective_metrics([row["mean_selected_pair_probability"] for row in per_sequence],
                                                [row["structure_f1"] for row in per_sequence]),
               "per_sequence": per_sequence, "independent_audit": audit,
               "code_sha256": sha256(Path(__file__).resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    with args.out.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **saved)
    args.out.with_suffix(".npz.tmp").replace(arrays_path)
    payload["pair_arrays"] = str(arrays_path)
    payload["pair_arrays_sha256"] = sha256(arrays_path)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
