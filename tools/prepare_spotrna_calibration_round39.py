#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_spotrna_round39 import validate
from calibrate_bpfold_round39 import flatten_probability
from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    audit = validate(args.source, args.split, True)
    if source["status"] == "EMPTY_AFTER_DELEAK":
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "system": "spotrna",
                   "source_result": str(args.source), "result_sha256": sha256(args.source)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload))
        return
    rows = [json.loads(line) for line in Path(source["data"]).read_text().splitlines() if line.strip()]
    names, offsets, probabilities, labels = [], [0], [], []
    confidence_values, structure_f1 = [], []
    per_sequence = []
    with np.load(source["probabilities"], allow_pickle=True) as archive:
        for row, saved, matrix in zip(rows, source["per_sequence"], archive["probabilities"]):
            assert row["name"] == saved["name"]
            flat_probability, flat_labels = flatten_probability(row, matrix)
            pred = {tuple(pair) for pair in saved["pred_pairs"]}
            confidence = float(np.mean([matrix[i, j] for i, j in pred])) if pred else 0.0
            names.append(row["name"])
            offsets.append(offsets[-1] + len(flat_probability))
            probabilities.append(flat_probability)
            labels.append(flat_labels)
            confidence_values.append(confidence)
            structure_f1.append(saved["f1"])
            per_sequence.append({"name": row["name"], "structure_f1": saved["f1"],
                                 "mean_selected_pair_probability": confidence,
                                 "pred_pairs": saved["pred_pairs"], "n_predicted_pairs": len(pred),
                                 "n_gt_pairs": len(row["pairs"])})
    saved_arrays = {"names": np.array(names), "offsets": np.array(offsets, dtype=np.int64),
                    "probabilities": np.concatenate(probabilities), "labels": np.concatenate(labels)}
    arrays_path = args.out.with_suffix(".npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **saved_arrays)
    payload = {"schema_version": 1, "status": "PASS_NATIVE_PROBABILITY_EVALUATION", "system": "spotrna",
               "distribution": "MultiMolecule converted checkpoint; not an original-author file",
               "data": source["data"], "data_sha256": source["data_sha256"], "n_sequences": len(rows),
               "clean_audit": source["clean_audit"],
               "protocol": {"device_type": "cuda", "cached_embeddings": False, "repeats": 3,
                            "scope": "existing verified raw-sequence official MultiMolecule use_postprocess=True contact_map; CPU conversion only",
                            "candidate_pairs": "strict upper valid_pair_mask", "n_bins": 10,
                            "structure_confidence": "mean native contact-map probability of all official threshold-selected pairs, including noncanonical pairs excluded from the common ECE mask; zero if none"},
               "weights": source["weights"], "weights_sha256": source["weights_sha256"],
               "source_result": str(args.source), "source_result_sha256": sha256(args.source),
               "source_probabilities": source["probabilities"], "source_probabilities_sha256": source["probabilities_sha256"],
               "f1_runs": [run["metrics"]["micro_f1"] for run in source["repeats"]],
               "calibration": metrics(saved_arrays["probabilities"], saved_arrays["labels"]),
               "confidence": selective_metrics(confidence_values, structure_f1),
               "per_sequence": per_sequence, "independent_audit": audit,
               "code_sha256": sha256(Path(__file__).resolve())}
    args.out.with_suffix(".npz.tmp").replace(arrays_path)
    payload["pair_arrays"] = str(arrays_path)
    payload["pair_arrays_sha256"] = sha256(arrays_path)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "split": args.split, "out": str(args.out),
                      "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
