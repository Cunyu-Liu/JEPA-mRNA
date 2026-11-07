#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from compare_ripe_paired_round48 import MAIN8, common_rows, compare_scope, subset


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_assessment(rows):
    significant_positive_b = False
    significant_negative_b = False
    significant_negative_main = False
    for row in rows:
        if row["status"] != "PASS" or row.get("small_sample_descriptive"):
            continue
        if row["split"] in MAIN8:
            if row.get("p_holm_main8", 1) < 0.05 and row["delta_micro_f1"] < 0:
                significant_negative_main = True
        elif row["p_two_sided"] < 0.05:
            significant_positive_b |= row["delta_micro_f1"] > 0
            significant_negative_b |= row["delta_micro_f1"] < 0
    return {"b_group_improvement": significant_positive_b and not significant_negative_b,
            "significant_negative_b": significant_negative_b,
            "main8_significant_regression": significant_negative_main,
            "meets_preregistered_b_route": significant_positive_b and not significant_negative_b and not significant_negative_main}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    assert args.bootstrap >= 1000
    manifest_path = args.root / "data/rna_complex_round49/d1_preparation/comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    baseline = manifest["selected"]
    config_path = args.root / "data/rna_complex_round49/d1_preparation/d1_config.json"
    config = json.loads(config_path.read_text())
    assert manifest["d1_config_sha256"] == sha256(config_path)
    paired_path = args.root / "eval_decision/complex_d1_paired_bootstrap.json"
    paired = json.loads(paired_path.read_text())
    assert paired["manifest_sha256"] == sha256(manifest_path)
    for relative, expected in paired["source_hashes"].items():
        assert sha256(args.root / relative) == expected
    rows = paired["comparisons"]["complex_d1"]
    sources = {str(manifest_path.relative_to(args.root)): sha256(manifest_path),
               str(config_path.relative_to(args.root)): sha256(config_path),
               str(paired_path.relative_to(args.root)): sha256(paired_path)}
    for name in [baseline["experiment"], "complex_d1"]:
        for suffix in ["summary.json", "sota_comparison.json", "confidence_summary.json"]:
            path = args.root / "eval_decision" / name / suffix
            assert json.loads(path.read_text())["status"].startswith("PASS_")
            sources[str(path.relative_to(args.root))] = sha256(path)
    for decoder in ["greedy", "turner0"]:
        path = args.root / f"eval_decision/complex_d1_{decoder}/summary.json"
        assert json.loads(path.read_text())["status"].startswith("PASS_")
        sources[str(path.relative_to(args.root))] = sha256(path)
    losses = []
    for seed in range(3):
        meta_path = args.root / f"runs/complex_d1_s{seed}/run_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
        assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
        assert meta["data"] == config["d1"]["data"]
        assert meta["objective"]["teacher"] == config["d1"]["teacher_dir"]
        losses.append(meta["best_validation_loss"])
        if meta["scorer"] == "pairjepa":
            for split in paired["source_hashes"]:
                if split.startswith("eval_decision/complex_d1/"):
                    result = json.loads((args.root / split).read_text())
                    assert result["pairjepa_teacher_forward_guard"] is True
    holdout = []
    for seed in range(3):
        paths = [args.root / f"eval_decision/{name}/s{seed}/complex_holdout.json"
                 for name in ["complex_d0", "complex_d1"]]
        left, right = [json.loads(path.read_text()) for path in paths]
        for path in paths:
            sources[str(path.relative_to(args.root))] = sha256(path)
        assert right["data_sha256"] == config["complex_holdout_sha256"]
        left_rows, right_rows, extra = common_rows(left, right)
        digest = hashlib.sha256(f"complex_d1:holdout:{seed}".encode()).digest()
        row = {"seed": seed, "left_supported_n": left["n"], "right_supported_n": right["n"],
               "right_only_supported_n": len(extra),
               **compare_scope(left_rows, right_rows, int.from_bytes(digest[:8], "little"), args.bootstrap)}
        holdout.append(row)
        for scope in ["length_200_400", "length_over_400", "span_30", "span_100"]:
            scoped = subset(left_rows, scope)
            other = subset(right_rows, scope)
            if scoped:
                digest = hashlib.sha256(f"complex_d1:{scope}:{seed}".encode()).digest()
                holdout.append({"seed": seed, "scope": scope,
                                **compare_scope(scoped, other, int.from_bytes(digest[:8], "little"), args.bootstrap)})
    mean_loss = sum(losses) / 3
    candidates = [{"experiment": baseline["experiment"], "run_prefix": baseline["run_prefix"],
                   "mean_best_vl0": baseline["mean_best_vl0"]},
                  {"experiment": "complex_d1", "run_prefix": "complex_d1_s", "mean_best_vl0": mean_loss}]
    selected = min(candidates, key=lambda row: (row["mean_best_vl0"], row["experiment"] != baseline["experiment"]))
    assessment = route_assessment(rows)
    output = {"status": "PASS_COMPLEX_D1_COMPLETE", "complete": True,
              "selection_rule": "D0 and D1 ranked only by three-seed mean best VL0; test metrics excluded",
              "next_base": selected, "candidates": candidates,
              **assessment, "holdout": holdout, "source_hashes": sources,
              "allow_lineardesign": True,
              "main_8_sota_observed": json.loads((args.root / "eval_decision/complex_d1/sota_comparison.json").read_text())["main_sota_any_wins"] == 8,
              "long_sequence_sota_claim": False, "best_confidence_claim": False,
              "exclusive_speed_sota_claim": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "next_base": selected,
                      "assessment": assessment, "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
