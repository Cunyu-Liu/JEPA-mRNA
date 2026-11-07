import argparse
import hashlib
import json
from pathlib import Path


ARMS = {"S1": ("structrfm_frozen", "structrfm_frozen_jev_s"),
        "S2": ("structrfm_full", "structrfm_jev_s"),
        "S3": ("structrfm_s3", "structrfm_s3_pairmlp490_s")}
ABLATIONS = ("greedy", "turner0")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_decision(root):
    root = Path(root)
    sources = {}
    arm_rows = {}
    metadata = {}
    for arm, (experiment, prefix) in ARMS.items():
        base = root / "eval_decision" / experiment
        required = {"summary": base / "summary.json",
                    "sota": base / "sota_comparison.json",
                    "confidence": base / "confidence_summary.json"}
        for kind, path in required.items():
            assert path.is_file(), path
            sources[str(path.relative_to(root))] = file_sha256(path)
        summary = json.loads(required["summary"].read_text())
        sota = json.loads(required["sota"].read_text())
        confidence = json.loads(required["confidence"].read_text())
        assert summary["status"].startswith("PASS_")
        assert sota["status"] == "PASS_STRUCTRFM_INDEPENDENT_COMPARISON"
        assert len(confidence["splits"]) == 13
        assert sota["result_sha256"] == sources[str(required["summary"].relative_to(root))]
        unsupported = {split: summary["splits"][split]["n_long_unsupported"]
                       for split in ("archiveii", "rfam_fam")}
        arm_rows[arm] = {"main_any_wins": sota["main_sota_any_wins"],
                         "main_external_single_wins": sota["main_sota_external_single_wins"],
                         "rfam_f1": sota["rfam"]["ours_f1_mean"],
                         "rfam_sota": sota["rfam"]["sota"],
                         "chanrg_measured_wins": sota["chanrg_measured_wins"],
                         "unsupported": unsupported}
        metadata[arm] = [json.loads((root / f"runs/{prefix}{seed}/run_meta.json").read_text())
                         for seed in range(3)]
        for seed, meta in enumerate(metadata[arm]):
            assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
        for ablation in ABLATIONS:
            path = root / f"eval_decision/{experiment}_{ablation}/summary.json"
            assert path.is_file() and json.loads(path.read_text())["status"].startswith("PASS_")
            sources[str(path.relative_to(root))] = file_sha256(path)
    paired_path = root / "eval_decision/structrfm_paired_bootstrap.json"
    paired = json.loads(paired_path.read_text())
    assert paired["status"] == "PASS_STRUCTRFM_S1_S2_S3_PAIRED_MICRO_F1_BOOTSTRAP"
    assert set(paired["comparisons"]) == {"S2_minus_S1", "S3_minus_S1", "S3_minus_S2"}
    sources[str(paired_path.relative_to(root))] = file_sha256(paired_path)
    candidates = [(meta["best_validation_loss"], arm, seed)
                  for arm, rows in metadata.items() for seed, meta in enumerate(rows)]
    best_loss, best_arm, best_seed = min(candidates)
    negative = all(row["main_any_wins"] == 0 and row["main_external_single_wins"] == 0
                   for row in arm_rows.values())
    return {"status": "PASS_ROUTE1_ADJUDICATION_WITH_NEGATIVE_SOTA" if negative else
                      "PASS_ROUTE1_ADJUDICATION_WITH_CANDIDATE",
            "route": "structRFM+JEV", "complete": True, "allow_ripe": True,
            "single_model_protocol": "one backbone, one head, one checkpoint per seed; no prediction averaging",
            "arms": arm_rows,
            "vl0_reference": {"arm": best_arm, "seed": best_seed,
                              "best_validation_loss": best_loss,
                              "selection": "minimum best VL0 loss only; not test-selected"},
            "long_sequence_sota_claim": False,
            "best_confidence_claim": False,
            "exclusive_speed_sota_claim": False,
            "paired_bootstrap_sha256": sources["eval_decision/structrfm_paired_bootstrap.json"],
            "source_hashes": sources,
            "next_route": "RIPE independent head and encoder arms with three-seed A/B/C",
            "boundary": "negative route-one evidence still permits preregistered RIPE; no 8/8, long-sequence, confidence or exclusive-speed claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = build_decision(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(decision, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": decision["status"], "out": str(args.out),
                      "sha256": file_sha256(args.out),
                      "vl0_reference": decision["vl0_reference"]}, indent=2))


if __name__ == "__main__":
    main()
