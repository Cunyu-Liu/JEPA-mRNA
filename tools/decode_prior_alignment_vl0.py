#!/usr/bin/env python3
"""Select C1-C3 MEA settings on VL0 and evaluate the four controlled identities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_audit import GAMMAS, TEMPERATURES, aggregate, tune_vl0  # noqa: E402
from pair_decode_common import decode_mea  # noqa: E402
from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask  # noqa: E402
from rnajepa.prior_alignment import load_contract, sha256_file  # noqa: E402
from rnajepa.prior_structure_metrics import pair_and_stem_counts, prf  # noqa: E402


def _macro(rows, level, metric):
    values = [prf(row[level]["tp"], row[level]["fp"], row[level]["fn"])[metric] for row in rows]
    return float(np.mean(values)) if values else None


def _summary(rows):
    output = {}
    for arm in ("C0", "C1", "C2", "C3"):
        selected = [row for row in rows if row["arm"] == arm]
        row = {"arm": arm, "n_sequences": len(selected)}
        for level in ("pair", "stem"):
            tp = sum(x[level]["tp"] for x in selected)
            fp = sum(x[level]["fp"] for x in selected)
            fn = sum(x[level]["fn"] for x in selected)
            micro = prf(tp, fp, fn)
            row[f"micro_{level}_precision"] = micro["precision"]
            row[f"micro_{level}_recall"] = micro["recall"]
            row[f"micro_{level}_f1"] = micro["f1"]
            for metric in ("precision", "recall", "f1"):
                row[f"macro_{level}_{metric}"] = _macro(selected, level, metric)
        output[arm] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-on-export", type=Path, required=True)
    parser.add_argument("--train-off-export", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=REPO / "configs/prior_alignment_2x2_v1.json")
    parser.add_argument("--contract-mode", choices=["smoke", "pilot", "full"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    on_meta = json.loads((args.train_on_export / "export_metadata.json").read_text())
    off_meta = json.loads((args.train_off_export / "export_metadata.json").read_text())
    checks = {
        "train_on_identity": on_meta["train_trajectory"] == "train_on",
        "train_off_identity": off_meta["train_trajectory"] == "train_off",
        "same_shared_init": on_meta["shared_init_sha256"] == off_meta["shared_init_sha256"],
        "different_terminal_checkpoints": on_meta["checkpoint"]["sha256"] != off_meta["checkpoint"]["sha256"],
        "same_contract": on_meta["contract_sha256"] == off_meta["contract_sha256"] == sha256_file(args.contract),
        "same_seed": on_meta["seed"] == off_meta["seed"],
        "same_terminal_step": on_meta["checkpoint_step"] == off_meta["checkpoint_step"],
        "same_record_count": on_meta["n_selected"] == off_meta["n_selected"],
        "correct_mode": on_meta["contract_mode"] == off_meta["contract_mode"] == args.contract_mode,
        "no_old_external_panels": not on_meta["old_external_panels_used"] and not off_meta["old_external_panels_used"],
        "contract_temperature_grid": list(TEMPERATURES) == contract["decode_selection"]["temperature_grid"],
        "contract_gamma_grid": list(GAMMAS) == contract["decode_selection"]["mea_gamma_grid"],
    }
    expected_step = 3200 if args.contract_mode == "pilot" else 20000 if args.contract_mode == "full" else None
    if expected_step is not None:
        checks["expected_terminal_step"] = on_meta["checkpoint_step"] == expected_step
    if args.contract_mode != "smoke":
        checks["complete_vl0"] = on_meta["n_selected"] == 196
    if not all(checks.values()):
        raise ValueError(f"decode identity gate failed: {checks}")
    args.out.mkdir(parents=True, exist_ok=True)
    tuning_root = args.out / "tuning_input"
    tuning_dir = tuning_root / "bprna_vl0"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    on_files = {path.stem: path for path in (args.train_on_export / "bprna_vl0").glob("*.npz")}
    off_files = {path.stem: path for path in (args.train_off_export / "bprna_vl0").glob("*.npz")}
    if set(on_files) != set(off_files):
        raise ValueError("train-on/off VL0 sequence identities differ")
    for sequence_hash in sorted(on_files):
        with np.load(on_files[sequence_hash], allow_pickle=False) as on, np.load(off_files[sequence_hash], allow_pickle=False) as off:
            if str(on["seq"]) != str(off["seq"]) or not np.array_equal(on["gt_pairs"], off["gt_pairs"]):
                raise ValueError(f"sequence or GT mismatch for {sequence_hash}")
            np.savez_compressed(tuning_dir / f"{sequence_hash}.npz",
                                C1_off=np.asarray(on["plana_off"], dtype=np.float32),
                                C2_off=np.asarray(off["plana_off"], dtype=np.float32),
                                C3_on=np.asarray(off["plana_on"], dtype=np.float32),
                                name=on["name"], seq=on["seq"], gt_pairs=on["gt_pairs"])
    dp_rows, rank_rows, entries, variant_config = tune_vl0(tuning_root)
    pd.DataFrame(dp_rows).to_csv(args.out / "vl0_grid_per_sequence.csv", index=False)
    grid_summary = aggregate(dp_rows, ["dataset_id", "score_variant", "decoder", "temperature", "gamma"])
    pd.DataFrame(grid_summary).to_csv(args.out / "vl0_grid_summary.csv", index=False)
    decode_config = {
        "C0": {"score_variant": "train_on/plana_on", "decoder": "MAP", "temperature": 1.0, "gamma": None},
        "C1": {"score_variant": "train_on/plana_off", "decoder": "MEA", **variant_config["C1_off"]["mea"]},
        "C2": {"score_variant": "train_off/plana_off", "decoder": "MEA", **variant_config["C2_off"]["mea"]},
        "C3": {"score_variant": "train_off/plana_on", "decoder": "MEA", **variant_config["C3_on"]["mea"]},
    }
    selected_rows = []
    for index, sequence_hash in enumerate(sorted(on_files), 1):
        with np.load(on_files[sequence_hash], allow_pickle=False) as on, np.load(off_files[sequence_hash], allow_pickle=False) as off:
            seq = str(on["seq"])
            truth = {tuple(map(int, pair)) for pair in on["gt_pairs"]}
            mask = valid_pair_mask(seq)
            sources = {"C0": np.asarray(on["plana_on"], dtype=np.float64),
                       "C1": np.asarray(on["plana_off"], dtype=np.float64),
                       "C2": np.asarray(off["plana_off"], dtype=np.float64),
                       "C3": np.asarray(off["plana_on"], dtype=np.float64)}
            for arm, scores in sources.items():
                config = decode_config[arm]
                if arm == "C0":
                    predicted = nussinov_map(scores, mask)
                else:
                    _, posterior = inside_outside(scores / config["temperature"], mask)
                    if not np.isfinite(posterior).all():
                        raise ValueError(f"non-finite posterior for {arm} {sequence_hash}")
                    predicted = decode_mea(posterior, mask, config["gamma"])
                row = {"dataset_id": "bprna_vl0", "sequence_sha256": sequence_hash,
                       "name": str(on["name"]), "arm": arm,
                       "train_trajectory": "train_on" if arm in ("C0", "C1") else "train_off",
                       "decoder": config["decoder"], "temperature": config["temperature"],
                       "gamma": config["gamma"], "n_truth_pairs": len(truth),
                       "n_predicted_pairs": len(predicted)}
                row.update(pair_and_stem_counts(predicted, truth))
                selected_rows.append(row)
        if index % 20 == 0:
            print(f"[prior-decode] selected {index}/{len(on_files)}", flush=True)
    with (args.out / "vl0_selected_per_sequence.jsonl").open("w") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": 1, "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract), "mode": args.contract_mode,
        "development_dataset": "bprna_vl0", "n_sequences": len(on_files),
        "shared_init_sha256": on_meta["shared_init_sha256"],
        "train_on_checkpoint_sha256": on_meta["checkpoint"]["sha256"],
        "train_off_checkpoint_sha256": off_meta["checkpoint"]["sha256"],
        "terminal_step": on_meta["checkpoint_step"], "seed": on_meta["seed"],
        "old_external_panels_used": False, "decode_config": decode_config,
        "arms": _summary(selected_rows), "identity_checks": checks,
        "interpretation_limit": "development-only controlled comparison; no production repair or SOTA claim",
    }
    (args.out / "vl0_selected_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "frozen_decode_config.json").write_text(json.dumps({
        "development_dataset": "bprna_vl0", "contract_sha256": sha256_file(args.contract),
        "selection_policy": contract["decode_selection"]["selection_policy"],
        "config": decode_config, "old_external_panels_may_select_parameters": False,
    }, indent=2) + "\n")
    (args.out / "decode_verification.json").write_text(json.dumps({
        "status": "PASS", "checks": checks, "n_sequences": len(on_files),
        "n_selected_rows": len(selected_rows), "finite_inside_outside": True,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
