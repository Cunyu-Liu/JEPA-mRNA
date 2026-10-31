#!/usr/bin/env python3
"""Aggregate one complete RiNALMo TE/EL cell-line task over ten official folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

FOLDS = tuple(range(10))
SEEDS = tuple(range(5))
HEADS = ("official_head", "jev_head")
QA_MARKER = "REGRESSION_RESULT_EXTERNAL_QA_PASS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_counts(frame: pd.DataFrame, fold: int) -> dict[str, int]:
    splits = list(KFold(n_splits=10, shuffle=True, random_state=42).split(frame))
    test = set(splits[fold][1])
    valid = set(splits[fold - 1][1])
    train = set(range(len(frame))) - test - valid
    return {"n_train": len(train), "n_validation": len(valid), "n_test": len(test)}


def verify_qa_log(path: Path, result_sha256: str) -> str:
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if (QA_MARKER in line and '"scope": "full"' in line and '"runs": 10' in line
                and '"cuda_device": "NVIDIA GeForce RTX 4090"' in line and result_sha256 in line):
            matches.append(line)
    if not matches:
        raise RuntimeError(f"no current full GPU-QA record in {path}")
    return sha256(path)


def load_fold(root: Path, task: str, cell_line: str, fold: int) -> tuple[dict, dict]:
    name = f"rinalmo_{task}_{cell_line}_fold{fold}_round33_full_v4"
    result_path = root / "eval_decision" / name / "result.json"
    qa_watch = "round34_qa_watch" if fold in (0, 1) else "round34_qa_watch_folds2_9"
    qa_log = root / "audits" / qa_watch / f"{name}.qa.log"
    result_sha256 = sha256(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = result["protocol"]
    if result.get("status") != "PASS":
        raise RuntimeError(f"fold is not PASS: {result_path}")
    if (protocol["task"] != task or protocol["cell_line"] != cell_line
            or protocol["fold"] != fold or protocol["device"] != "cuda"
            or protocol["sample_scope"] != "full_official_protocol"
            or protocol["seeds"] != list(SEEDS)):
        raise RuntimeError(f"protocol identity mismatch: {result_path}")
    source = Path(protocol["source_data"])
    if source != root / "data/rinalmo/te_el_official" / f"{cell_line}_sequence.csv":
        raise RuntimeError(f"unexpected source data: {source}")
    if sha256(source) != protocol["source_data_sha256"]:
        raise RuntimeError(f"source data hash mismatch: {source}")
    frame = pd.read_csv(source)
    expected_counts = split_counts(frame, fold)
    fold_output = {"fold": fold, "result": str(result_path), "result_sha256": result_sha256,
                   "external_gpu_qa_log": str(qa_log), "heads": {}}
    fold_output["external_gpu_qa_log_sha256"] = verify_qa_log(qa_log, result_sha256)
    for head in HEADS:
        runs = result["heads"][head]["runs"]
        if len(runs) != 5 or [run["seed"] for run in runs] != list(SEEDS):
            raise RuntimeError(f"seed contract mismatch: {head} fold={fold}")
        saved_runs = []
        for run in runs:
            if (run["device"] != "cuda" or not run["converged"] or not run["early_stopped"]
                    or run["independent_recheck"] != "PASS"
                    or run["epochs_run"] != run["best_epoch"] + 32
                    or len(run["validation_history"]) != run["epochs_run"]
                    or any(run[key] != value for key, value in expected_counts.items())):
                raise RuntimeError(f"convergence/count contract mismatch: {head} fold={fold} seed={run['seed']}")
            best = next(point for point in run["validation_history"] if point["epoch"] == run["best_epoch"])
            if best["validation_loss"] != run["best_validation_loss"]:
                raise RuntimeError(f"best checkpoint is not bound to validation history: {head} fold={fold}")
            if sha256(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {head} fold={fold} seed={run['seed']}")
            prediction_path = Path(run["prediction_file"])
            if sha256(prediction_path) != run["prediction_sha256"]:
                raise RuntimeError(f"prediction hash mismatch: {head} fold={fold} seed={run['seed']}")
            with np.load(prediction_path, allow_pickle=False) as saved:
                predictions = saved["predictions"].copy()
                truth = saved["truth"].copy()
            if predictions.shape != (expected_counts["n_test"],) or truth.shape != predictions.shape:
                raise RuntimeError(f"prediction shape mismatch: {head} fold={fold} seed={run['seed']}")
            spearman = float(stats.spearmanr(predictions, truth)[0])
            r2 = float(1 - np.sum((truth - predictions) ** 2) / np.sum((truth - truth.mean()) ** 2))
            pcc = float(stats.pearsonr(predictions, truth)[0])
            if not np.isclose(spearman, run["test_spearman"], rtol=0, atol=1e-10):
                raise RuntimeError(f"Spearman mismatch: {head} fold={fold} seed={run['seed']}")
            if not np.isclose(r2, run["test_r2"], rtol=0, atol=1e-10) or not np.isclose(pcc, run["test_pcc"], rtol=0, atol=1e-10):
                raise RuntimeError(f"secondary metric mismatch: {head} fold={fold} seed={run['seed']}")
            saved_runs.append({"seed": run["seed"], "spearman": spearman, "r2": r2, "pcc": pcc})
        fold_output["heads"][head] = {
            "runs": saved_runs,
            "spearman_seed_mean": float(np.mean([run["spearman"] for run in saved_runs])),
            "spearman_seed_sd": float(np.std([run["spearman"] for run in saved_runs], ddof=1)),
        }
    identity = {key: protocol[key] for key in (
        "source_data_sha256", "pretrained_sha256", "pretrained_config_sha256",
        "preflight_sha256", "embedding_cache_meta", "embedding_cache_meta_sha256", "code_sha256")}
    return fold_output, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024"))
    parser.add_argument("--task", required=True, choices=("te", "el"))
    parser.add_argument("--cell-line", required=True, choices=("Muscle", "pc3", "HEK"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite aggregate: {args.out}")

    fold_outputs, identities = [], []
    for fold in FOLDS:
        fold_output, identity = load_fold(args.root, args.task, args.cell_line, fold)
        fold_outputs.append(fold_output)
        identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("data/cache/code identity differs across folds")

    heads_output = {}
    seed_cv_deltas = []
    for head in HEADS:
        seed_cv = [float(np.mean([fold_output["heads"][head]["runs"][seed]["spearman"]
                                  for fold_output in fold_outputs])) for seed in SEEDS]
        fold_means = [fold_output["heads"][head]["spearman_seed_mean"] for fold_output in fold_outputs]
        heads_output[head] = {
            "seed_cv_spearman": seed_cv,
            "cv_spearman_mean": float(np.mean(fold_means)),
            "cv_spearman_fold_sd": float(np.std(fold_means, ddof=1)),
            "cv_r2_mean": float(np.mean([fold_output["heads"][head]["runs"][seed]["r2"]
                                         for fold_output in fold_outputs for seed in SEEDS])),
            "cv_pcc_mean": float(np.mean([fold_output["heads"][head]["runs"][seed]["pcc"]
                                          for fold_output in fold_outputs for seed in SEEDS])),
        }
    for seed in SEEDS:
        seed_cv_deltas.append(heads_output["jev_head"]["seed_cv_spearman"][seed]
                              - heads_output["official_head"]["seed_cv_spearman"][seed])
    delta = np.asarray(seed_cv_deltas)
    delta_sd = float(np.std(delta, ddof=1))
    test = stats.ttest_rel(
        heads_output["jev_head"]["seed_cv_spearman"],
        heads_output["official_head"]["seed_cv_spearman"], alternative="greater")
    statistics_ok = bool(delta_sd > 0 and np.isfinite(test.statistic) and np.isfinite(test.pvalue))
    comparison = {
        "primary_metric": "spearman",
        "aggregation": "mean over ten official folds within each seed, then five-seed paired comparison",
        "seed_cv_delta_jev_minus_official": seed_cv_deltas,
        "paired_delta_mean": float(np.mean(delta)),
        "paired_delta_sample_sd": delta_sd,
        "one_sided_ttest_rel_greater_t_statistic": float(test.statistic) if np.isfinite(test.statistic) else None,
        "one_sided_ttest_rel_greater_p_value": float(test.pvalue) if np.isfinite(test.pvalue) else None,
        "statistics_status": "ok" if statistics_ok else "undefined_zero_variance",
        "significant_positive": bool(statistics_ok and np.mean(delta) > 0 and test.pvalue < 0.05),
        "inference_scope": "paired training-seed effects after official ten-fold averaging; no test-set population inference",
    }
    payload = {"schema_version": 1, "status": "PASS", "scientific_eligibility": True,
               "protocol": {"task": args.task, "cell_line": args.cell_line, "folds": list(FOLDS),
                            "seeds": list(SEEDS), "n_fold_results": 10, "n_runs_per_head": 50,
                            "identity": identities[0]},
               "heads": heads_output, "comparison": comparison, "fold_results": fold_outputs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    if checked["status"] != "PASS" or len(checked["fold_results"]) != 10:
        raise RuntimeError("aggregate JSON reread failed")
    temporary.replace(args.out)
    print("TE_EL_CV_AGGREGATE_PASS", json.dumps({
        "task": args.task, "cell_line": args.cell_line, "folds": 10,
        "runs": 100, "delta_spearman": comparison["paired_delta_mean"],
        "significant_positive": comparison["significant_positive"],
        "aggregate_sha256": sha256(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
