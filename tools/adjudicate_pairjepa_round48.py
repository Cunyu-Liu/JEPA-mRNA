import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from adjudicate_ripe_round48 import select_next_base


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifacts(root, experiment, sources):
    base = root / "eval_decision" / experiment
    summary_path = base / "summary.json"
    sota_path = base / "sota_comparison.json"
    confidence_path = base / "confidence_summary.json"
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
        path = root / f"eval_decision/{experiment}_{ablation}/summary.json"
        assert json.loads(path.read_text())["status"].startswith("PASS_")
        sources[str(path.relative_to(root))] = sha256(path)
    return summary, sota


def diagnostics(run, seed):
    path = run / "jepa_diagnostics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert row["masked_pairs"] > 0 and math.isfinite(row["loss"]) and row["loss"] >= 0
        for key in ("target_variance", "prediction_variance",
                    "target_effective_rank", "prediction_effective_rank"):
            assert math.isfinite(row[key]) and row[key] >= 0, (seed, key)
    return {"n_records": len(rows), "last": rows[-1],
            "minimum_target_effective_rank": min(row["target_effective_rank"] for row in rows),
            "minimum_prediction_effective_rank": min(row["prediction_effective_rank"] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "audits/round48/pairjepa_preregistered.json"
    manifest = json.loads(manifest_path.read_text())
    paired_path = root / "eval_decision/pairjepa_paired_bootstrap.json"
    paired = json.loads(paired_path.read_text())
    assert paired["status"] == "PASS_PAIRJEPA_PAIRED_MICRO_F1_BOOTSTRAP"
    assert paired["manifest_sha256"] == sha256(manifest_path)
    for relative, expected in paired["source_hashes"].items():
        assert sha256(root / relative) == expected, relative
    sources = {str(manifest_path.relative_to(root)): sha256(manifest_path),
               str(paired_path.relative_to(root)): sha256(paired_path)}
    baseline = manifest["baseline"]
    _, baseline_sota = validate_artifacts(root, baseline["experiment"], sources)
    candidates = [{"experiment": baseline["experiment"], "run_prefix": baseline["run_prefix"],
                   "mean_best_vl0": baseline["mean_best_vl0"],
                   "ripe_mode": baseline["ripe_mode"], "ripe_encoder": baseline["ripe_encoder"]}]
    arm_rows = {"J0": {"experiment": baseline["experiment"],
                        "main_sota_any_wins": baseline_sota["main_sota_any_wins"],
                        "rfam_f1": baseline_sota["rfam"]["ours_f1_mean"]}}
    for arm in manifest["arms"]:
        if arm.get("reuse_baseline"):
            arm_rows[arm["arm"]] = {"reuse_J0": True}
            continue
        name = arm["arm"]
        assert name in paired["comparisons"] and name in paired["long_subsets"]
        summary, sota = validate_artifacts(root, name, sources)
        losses = []
        diagnostic_rows = {}
        for seed in range(3):
            run = root / "runs" / f"{name}_s{seed}"
            meta = json.loads((run / "run_meta.json").read_text())
            assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
            assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
            assert meta["scorer"] == arm["scorer"]
            assert meta["objective"]["lambda_jepa"] == arm["lambda_jepa"]
            assert meta["objective"]["jepa_target"] == arm["target"]
            assert meta["objective"].get("jepa_context") == (
                "context_v1" if arm["scorer"] == "pairjepa" else None)
            assert meta["ripe"]["encoder"] == manifest["ripe_encoder"]
            expected_mode = None if manifest["ripe_encoder"] else manifest["ripe_mode"]
            assert meta["ripe"]["mode"] == expected_mode
            checkpoint_hash = sha256(run / "best.pt")
            losses.append(meta["best_validation_loss"])
            for split in summary["splits"].values():
                assert split["cells"][seed]["checkpoint_sha256"] == checkpoint_hash
            if arm["scorer"] == "pairjepa":
                diagnostic_rows[f"s{seed}"] = diagnostics(run, seed)
                for result_path in (root / "eval_decision" / name / f"s{seed}").glob("*.json"):
                    result = json.loads(result_path.read_text())
                    assert result["pairjepa_teacher_forward_guard"] is True
        candidates.append({"experiment": name, "run_prefix": f"{name}_s",
                           "mean_best_vl0": statistics.fmean(losses),
                           "ripe_mode": manifest["ripe_mode"], "ripe_encoder": manifest["ripe_encoder"]})
        arm_rows[name] = {"main_sota_any_wins": sota["main_sota_any_wins"],
                          "main_sota_external_single_wins": sota["main_sota_external_single_wins"],
                          "rfam_f1": sota["rfam"]["ours_f1_mean"], "rfam_sota": sota["rfam"]["sota"],
                          "chanrg_measured_wins": sota["chanrg_measured_wins"],
                          "mean_best_vl0": candidates[-1]["mean_best_vl0"],
                          "jepa_diagnostics": diagnostic_rows}
    selected = select_next_base(candidates)
    output = {"status": "PASS_PAIRJEPA_ROUTE_COMPLETE", "complete": True,
              "allow_complex_data_route": True,
              "selection_rule": "J0 and newly trained controls ranked only by three-seed mean best VL0; test metrics excluded",
              "candidates": candidates, "next_base": selected, "arms": arm_rows,
              "source_hashes": sources,
              "anti_collapse_rule": "finite masked losses, variances and effective ranks required; low ranks retained without an arbitrary pass threshold",
              "inference_rule": "all Pair-JEPA main results passed fail-closed teacher-forward guard",
              "main_8_sota_observed": any(row.get("main_sota_any_wins") == 8 for row in arm_rows.values()),
              "long_sequence_sota_claim": False, "best_confidence_claim": False,
              "exclusive_speed_sota_claim": False,
              "next": "RNA complex D0/D1 with entry/cluster leakage control; LinearDesign remains after D0/D1 adjudication"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out),
                      "sha256": sha256(args.out), "next_base": selected}, indent=2))


if __name__ == "__main__":
    main()
