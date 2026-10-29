#!/usr/bin/env python3
"""Independently reload a round-34 SSP checkpoint and reproduce every saved prediction on GPU."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import nb_secstr_dual_round34 as pipeline


def independent_metrics(predicted, truth):
    pred = {tuple(pair) for pair in predicted}
    gt = {tuple(pair) for pair in truth}
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 0.0 if precision + recall < 1e-5 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for independent QA")
    result_path = args.run_dir / "run.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "PASS":
        raise RuntimeError("run result is not PASS")
    if result["code_sha256"] != pipeline.sha256(Path(pipeline.__file__).resolve()):
        raise RuntimeError("training code changed after result creation")
    full = result["limit"] == 0
    if full and not (result["scientific_eligibility"] and result["converged"] and result["early_stopped"]):
        raise RuntimeError("full result lacks convergence/eligibility evidence")
    if not full and result["scientific_eligibility"]:
        raise RuntimeError("smoke incorrectly marked scientifically eligible")
    checkpoint_path = Path(result["checkpoint"])
    if pipeline.sha256(checkpoint_path) != result["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head = pipeline.build_head(result["head"], args.device)
    head.load_state_dict(checkpoint["model_state"], strict=True)
    head.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries = {}
    for split in ("validation", "ts0", "archiveii600"):
        role = split
        records, evidence = pipeline.load_records(role, result["limit"])
        expected_evidence = result["data"][role]
        if evidence["jsonl_sha256"] != expected_evidence["jsonl_sha256"]:
            raise RuntimeError(f"{split}: source data changed")
        expected_cache = {e["path"]: e["sha256"] for e in expected_evidence["cache_entries"]}
        actual_cache = {e["path"]: e["sha256"] for e in evidence["cache_entries"]}
        if expected_cache != actual_cache:
            raise RuntimeError(f"{split}: embedding manifest changed")
        for path, digest in actual_cache.items():
            if pipeline.sha256(Path(path)) != digest:
                raise RuntimeError(f"{split}: embedding shard hash mismatch: {path}")
        reproduced = pipeline.predict(result["head"], head, records, args.device, result["threshold"])
        prediction_path = args.run_dir / f"{split}_predictions.jsonl"
        saved = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        if len(saved) != len(reproduced) or len(saved) != result["predictions"][split]["rows"]:
            raise RuntimeError(f"{split}: prediction count mismatch")
        if pipeline.sha256(prediction_path) != result["predictions"][split]["sha256"]:
            raise RuntimeError(f"{split}: prediction file hash mismatch")
        metrics = []
        for old, new in zip(saved, reproduced):
            if old["name"] != new["name"] or old["gt_pairs"] != new["gt_pairs"] or old["pred_pairs"] != new["pred_pairs"]:
                raise RuntimeError(f"{split}: checkpoint re-inference mismatch: {old['name']}")
            p, r, f = independent_metrics(old["pred_pairs"], old["gt_pairs"])
            for key, value in (("precision", p), ("recall", r), ("f1", f)):
                if not math.isclose(old[key], value, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"{split}: independent metric mismatch: {old['name']}")
            metrics.append((p, r, f))
        summary = {
            "rows": len(saved), "macro_precision": float(np.mean([m[0] for m in metrics])),
            "macro_recall": float(np.mean([m[1] for m in metrics])), "macro_f1": float(np.mean([m[2] for m in metrics])),
            "prediction_sha256": pipeline.sha256(prediction_path), "gpu_reinference_exact": True,
        }
        for key in ("macro_precision", "macro_recall", "macro_f1"):
            if not math.isclose(summary[key], result["predictions"][split][key], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"{split}: aggregate {key} mismatch")
        summaries[split] = summary
        print(f"QA_SPLIT_PASS {split} rows={len(saved)} macro_f1={summary['macro_f1']:.8f}", flush=True)
    payload = {
        "schema_version": 1, "status": "PASS", "run": str(result_path),
        "run_sha256": pipeline.sha256(result_path), "head": result["head"], "seed": result["seed"],
        "scope": "full" if full else "engineering_smoke", "scientific_eligibility": full,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint_sha256": result["checkpoint_sha256"], "splits": summaries,
        "code_sha256": pipeline.sha256(Path(pipeline.__file__).resolve()),
    }
    pipeline.atomic_json(args.run_dir / "qa.json", payload)
    checked = json.loads((args.run_dir / "qa.json").read_text())
    assert checked["status"] == "PASS" and set(checked["splits"]) == set(summaries)
    print(f"NB_SECSTR_EXTERNAL_GPU_QA_PASS {args.run_dir} sha256={pipeline.sha256(args.run_dir / 'qa.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
