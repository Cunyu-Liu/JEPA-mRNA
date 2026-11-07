import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_next_base(candidates):
    return min(candidates, key=lambda row: (row["mean_best_vl0"], row["experiment"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "audits/round48/ripe_preregistered.json"
    manifest = json.loads(manifest_path.read_text())
    paired_path = root / "eval_decision/ripe_paired_bootstrap.json"
    paired = json.loads(paired_path.read_text())
    assert paired["status"] == "PASS_RIPE_PAIRED_MICRO_F1_BOOTSTRAP"
    assert paired["manifest_sha256"] == sha256(manifest_path)
    for relative, expected in paired["source_hashes"].items():
        assert sha256(root / relative) == expected, relative
    sources = {str(manifest_path.relative_to(root)): sha256(manifest_path),
               str(paired_path.relative_to(root)): sha256(paired_path)}
    baseline = manifest["selected"]
    candidates = [{"experiment": baseline["experiment"], "run_prefix": baseline["run_prefix"],
                   "mean_best_vl0": baseline["mean_best_vl0"], "ripe_mode": None,
                   "ripe_encoder": False}]
    arm_rows = {}
    for arm in manifest["arms"]:
        name = arm["arm"]
        assert name in paired["comparisons"] and name in paired["long_subsets"]
        experiment_dir = root / "eval_decision" / name
        summary_path = experiment_dir / "summary.json"
        sota_path = experiment_dir / "sota_comparison.json"
        confidence_path = experiment_dir / "confidence_summary.json"
        summary = json.loads(summary_path.read_text())
        sota = json.loads(sota_path.read_text())
        confidence = json.loads(confidence_path.read_text())
        assert summary["status"].startswith("PASS_")
        assert sota["status"] == "PASS_STRUCTRFM_INDEPENDENT_COMPARISON"
        assert sota["result_sha256"] == sha256(summary_path)
        assert len(confidence["splits"]) == 13
        for path in (summary_path, sota_path, confidence_path):
            sources[str(path.relative_to(root))] = sha256(path)
        for ablation in ("greedy", "turner0"):
            path = root / f"eval_decision/{name}_{ablation}/summary.json"
            assert json.loads(path.read_text())["status"].startswith("PASS_")
            sources[str(path.relative_to(root))] = sha256(path)
        losses = []
        for seed in range(3):
            run = root / "runs" / f"{name}_s{seed}"
            meta = json.loads((run / "run_meta.json").read_text())
            assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
            assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
            assert meta["environment"]["resolved_device"].startswith("cuda")
            assert meta["ripe"]["encoder"] == arm["encoder"]
            assert meta["ripe"]["mode"] == (None if arm["encoder"] else arm["mode"])
            assert meta["scorer"] == manifest["training"]["scorer"]
            checkpoint_hash = sha256(run / "best.pt")
            losses.append(meta["best_validation_loss"])
            for split in summary["splits"].values():
                assert split["cells"][seed]["checkpoint_sha256"] == checkpoint_hash
        candidates.append({"experiment": name, "run_prefix": f"{name}_s",
                           "mean_best_vl0": statistics.fmean(losses),
                           "ripe_mode": arm["mode"], "ripe_encoder": arm["encoder"]})
        arm_rows[name] = {
            "main_sota_any_wins": sota["main_sota_any_wins"],
            "main_sota_external_single_wins": sota["main_sota_external_single_wins"],
            "rfam_f1": sota["rfam"]["ours_f1_mean"], "rfam_sota": sota["rfam"]["sota"],
            "chanrg_measured_wins": sota["chanrg_measured_wins"],
            "mean_best_vl0": candidates[-1]["mean_best_vl0"],
            "length_and_span": {split: {"n_long_unsupported": summary["splits"][split]["n_long_unsupported"],
                                         "cells": [{"length_buckets": cell["length_buckets"],
                                                    "span_30": cell["span_30"], "span_100": cell["span_100"]}
                                                   for cell in summary["splits"][split]["cells"]]}
                                for split in ("archiveii", "rfam_fam")}}
    selected = select_next_base(candidates)
    decision = {"status": "PASS_RIPE_ROUTE_COMPLETE", "complete": True,
                "allow_pair_jepa": True,
                "selection_rule": "R0 and all RIPE arms ranked only by three-seed mean best VL0 loss; test metrics excluded",
                "candidates": candidates, "next_base": selected,
                "arms": arm_rows, "source_hashes": sources,
                "main_8_sota_observed": any(row["main_sota_any_wins"] == 8 for row in arm_rows.values()),
                "long_sequence_sota_claim": False,
                "best_confidence_claim": False, "exclusive_speed_sota_claim": False,
                "boundary": "route completion permits preregistered Pair-JEPA; long, confidence and speed claims require independent final comparisons",
                "next": "Pair-JEPA J-mlp/J1/J1-noEMA/J-random on the VL0-selected base"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(decision, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": decision["status"], "out": str(args.out),
                      "sha256": sha256(args.out), "next_base": selected}, indent=2))


if __name__ == "__main__":
    main()
