#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_plana_round39 import selective_metrics, sha256
from evaluate_spotrna_vl0_calibration_round39 import metrics
from verify_bpfold_calibration_round39 import validate_result as validate_bpfold
from verify_xens2_calibration_round39 import validate_result as validate_xens2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xens2", type=Path, required=True)
    parser.add_argument("--bpfold", "--baseline", dest="bpfold", type=Path, required=True)
    parser.add_argument("--baseline-system", choices=["bpfold", "spotrna", "rnaformer", "rinalmo_ft", "nucleicbert_ft", "nucleicbert_frozen", "kirigami", "mxfold2", "vienna_mfe", "vienna_centroid", "vienna_mea", "eternafold", "contrafold"], default="bpfold")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    system = args.baseline_system
    xens2 = json.loads(args.xens2.read_text())
    bpfold = json.loads(args.bpfold.read_text())
    xens2_audit = validate_xens2(args.xens2)
    if system == "spotrna":
        from verify_spotrna_calibration_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    elif system in {"rnaformer", "rinalmo_ft"}:
        from verify_rnaformer_calibration_round39 import validate_result
        assert args.baseline_summary is not None
        bpfold_audit = validate_result(args.bpfold, args.baseline_summary)
    elif system in {"nucleicbert_ft", "nucleicbert_frozen"}:
        from verify_nucleicbert_calibration_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    elif system == "kirigami":
        from calibrate_kirigami_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    elif system == "mxfold2":
        from calibrate_mxfold2_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    elif system in {"vienna_mfe", "vienna_centroid", "vienna_mea"}:
        from calibrate_vienna_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    elif system in {"eternafold", "contrafold"}:
        from calibrate_contrafold_round39 import validate_result
        bpfold_audit = validate_result(args.bpfold)
    else:
        bpfold_audit = validate_bpfold(args.bpfold)
    assert Path(xens2["data"]).stem == Path(bpfold["data"]).stem
    raw_key = f"{system}_raw"
    confidence_key = "bpfold_CI" if system == "bpfold" else f"{system}_native"
    with np.load(xens2["pair_arrays"], allow_pickle=False) as x_archive, np.load(bpfold["pair_arrays"], allow_pickle=False) as b_archive:
        xa = {key: x_archive[key] for key in x_archive.files}
        ba = {key: b_archive[key] for key in b_archive.files}
        x_names, b_names = xa["names"].tolist(), ba["names"].tolist()
        b_name_set = set(b_names)
        common = [name for name in x_names if name in b_name_set]
        x_lookup = {name: index for index, name in enumerate(x_names)}
        b_lookup = {name: index for index, name in enumerate(b_names)}
        gt_protocol_mismatch_excluded = 0
        if system in {"nucleicbert_ft", "nucleicbert_frozen"}:
            matched = []
            for name in common:
                xi, bi = x_lookup[name], b_lookup[name]
                xs, xe = xa["offsets"][xi:xi + 2]
                bs, be = ba["offsets"][bi:bi + 2]
                if xe - xs == be - bs and np.array_equal(xa["labels"][xs:xe], ba["labels"][bs:be]):
                    matched.append(name)
                else:
                    gt_protocol_mismatch_excluded += 1
            common = matched
        if not common:
            payload = {"schema_version": 1, "status": "EMPTY_MATCHED_CALIBRATION_INTERSECTION",
                       "split": Path(xens2["data"]).stem, "matched_n": 0, "baseline_system": system,
                       "xens2_clean_n": xens2["n_sequences"], "baseline_clean_n": bpfold["n_sequences"],
                       "bpfold_clean_n": bpfold["n_sequences"],
                       "gt_protocol_mismatch_excluded": gt_protocol_mismatch_excluded,
                       "main_ranking_boundary": "no shared name+sequence+evaluated-GT samples; no ECE or confidence metric is defined",
                       "xens2_result_sha256": sha256(args.xens2), "baseline_result_sha256": sha256(args.bpfold),
                       "calibration": {}, "structure_confidence": {}}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out.with_suffix(args.out.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.out)
            print(json.dumps({"status": payload["status"], "split": payload["split"],
                              "out": str(args.out), "sha256": sha256(args.out)}, indent=2))
            return
        x_lookup = {name: index for index, name in enumerate(x_names)}
        b_lookup = {name: index for index, name in enumerate(b_names)}
        probabilities = {key: [] for key in ["xens2_raw", "xens2_calibrated", raw_key]}
        if system == "bpfold":
            probabilities["bpfold_post"] = []
        structure = {key: [] for key in ["xens2_f1", "xens2_raw_confidence", "xens2_calibrated_confidence",
                                        "baseline_f1", "baseline_confidence"]}
        for name in common:
            xi, bi = x_lookup[name], b_lookup[name]
            xs, xe = xa["offsets"][xi:xi + 2]
            bs, be = ba["offsets"][bi:bi + 2]
            assert xe - xs == be - bs and np.array_equal(xa["labels"][xs:xe], ba["labels"][bs:be])
            probabilities["xens2_raw"].append(xa["raw_probabilities"][xs:xe])
            probabilities["xens2_calibrated"].append(xa["calibrated_probabilities"][xs:xe])
            probabilities[raw_key].append(ba["probabilities"][bs:be])
            if system == "bpfold":
                probabilities["bpfold_post"].append(ba["post_probabilities"][bs:be])
            xrow = xens2["per_sequence"][xi]
            brow = bpfold["per_sequence"][bi]
            assert xrow["name"] == brow["name"] == name
            structure["xens2_f1"].append(xrow["structure_f1"])
            structure["xens2_raw_confidence"].append(xrow["mean_selected_pair_probability"])
            structure["xens2_calibrated_confidence"].append(xrow["calibrated_confidence"])
            structure["baseline_f1"].append(brow["structure_f1"])
            structure["baseline_confidence"].append(brow["CI"] if system == "bpfold" else brow["mean_selected_pair_probability"])
        matched_labels = np.concatenate([xa["labels"][xa["offsets"][x_lookup[name]]:xa["offsets"][x_lookup[name] + 1]]
                                         for name in common])
        calibration = {key: metrics(np.concatenate(values), matched_labels) for key, values in probabilities.items()}
        confidence = {
            "xens2_raw": selective_metrics(structure["xens2_raw_confidence"], structure["xens2_f1"]),
            "xens2_calibrated": selective_metrics(structure["xens2_calibrated_confidence"], structure["xens2_f1"]),
            confidence_key: selective_metrics(structure["baseline_confidence"], structure["baseline_f1"]),
        }
    payload = {"schema_version": 1, "status": "PASS_MATCHED_CALIBRATION_SENSITIVITY_NOT_MAIN_LEADERBOARD",
               "split": Path(xens2["data"]).stem, "matched_n": len(common), "baseline_system": system,
               "xens2_clean_n": xens2["n_sequences"], "baseline_clean_n": bpfold["n_sequences"],
               "bpfold_clean_n": bpfold["n_sequences"],
               "gt_protocol_mismatch_excluded": gt_protocol_mismatch_excluded,
               "candidate_pairs": "same name+sequence+evaluated GT, strict upper valid_pair_mask, 10 bins",
               "main_ranking_boundary": "native own-clean leaderboard is primary; this matched analysis is auxiliary only",
               "postprocess_boundary": ("BPfold _post.npy is constrained/postprocessed and may be binary; its ECE is diagnostic, not evidence of independently calibrated marginal probabilities" if system == "bpfold" else
                                        "native SPOT contact map; common ECE mask excludes noncanonical pairs but native confidence retains them" if system == "spotrna" else
                                        f"native {system} pair probabilities; official native decode retained; common valid-pair mask only for comparable ECE"),
               "xens2_result_sha256": sha256(args.xens2), "baseline_result_sha256": sha256(args.bpfold),
               "bpfold_result_sha256": sha256(args.bpfold),
               "xens2_independent_audit": xens2_audit, "baseline_independent_audit": bpfold_audit,
               "bpfold_independent_audit": bpfold_audit,
               "calibration": calibration, "structure_confidence": confidence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": payload["status"], "split": payload["split"], "matched_n": len(common),
                      "out": str(args.out), "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
