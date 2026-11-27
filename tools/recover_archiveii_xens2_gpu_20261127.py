#!/usr/bin/env python3
"""Record-level frozen xens2 recovery with per-ID checkpoints and GPU peak evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "eval"))
import recover_archiveii_full3966_cpu_baselines_20261127 as base
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.train_decision import EmbeddingStore
from xens2_eval import load_plana, load_r2d, plana_scores, r2d_scores

PLAN_PATH = ROOT / "source_data/archiveii_full3966_cpu_recovery_20261127/recovery_plan.json"
DEFAULT_WORK = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/recovery/archiveii_xens2_gpu_20261127")
CKPTS = {
    "plana_s0": Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt"),
    "plana_s1": Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt"),
    "r2d_s0": Path("/root/rna_jepa_data/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt"),
}
EMBEDDING_DIR = Path("/root/rna_jepa_data/embeddings/rinalmo-giga")
EMBEDDING_SHARD = EMBEDDING_DIR / "archiveii.shard0of1.npz"
SCHEMA = "archiveii.xens2.gpu_recovery.v1"


def checkpoint_identities():
    return {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": base.sha256(path)}
            for name, path in CKPTS.items()}


def protocol_block_reason(row, store):
    if set(row["seq"]) - set("ACGU"):
        return "REJECTED_NON_ACGU"
    if len(row["seq"]) > 1024:
        return "BLOCKED_PROTOCOL_LENGTH_GT1024"
    if store.get(row["seq"]) is None:
        return "BLOCKED_PROTOCOL_EMBEDDING_MISSING"
    return None


def blocked_record(row, reason):
    return {"schema_version": SCHEMA, "name": row["name"], "sequence_sha256": row["sequence_sha256"],
            "sequence_length": len(row["seq"]), "status": reason, "pairs": None, "pair_set_sha256": None,
            "non_acgu_characters": sorted(set(row["seq"]) - set("ACGU"))}


def success_record(row, pairs, device_evidence):
    return {"schema_version": SCHEMA, "name": row["name"], "sequence_sha256": row["sequence_sha256"],
            "sequence_length": len(row["seq"]), "status": "PASS", "pairs": pairs,
            "pair_set_sha256": base.value_sha(pairs), "non_acgu_characters": [],
            "prediction_identity": "xens2_three_checkpoint_ensemble_w0.7_plana_s1_weight0.5",
            "device_evidence": device_evidence}


def validate_saved(record, row):
    if record["name"] != row["name"] or record["sequence_sha256"] != row["sequence_sha256"]:
        raise ValueError("xens2 checkpoint identity mismatch")
    if record["status"] == "PASS":
        pairs = base.pair_set(record["pairs"], len(row["seq"]))
        if pairs != [tuple(pair) for pair in record["pairs"]] or record["pair_set_sha256"] != base.value_sha(pairs):
            raise ValueError("xens2 checkpoint pair mismatch")
    elif record["pairs"] is not None:
        raise ValueError("blocked xens2 checkpoint has pairs")


def consolidate(work_root, rows):
    records, remaining = [], []
    for row in rows:
        path = base.record_path(work_root, "xens2", row["name"])
        if not path.is_file():
            remaining.append(row["name"])
            continue
        record = json.loads(path.read_text())
        validate_saved(record, row)
        records.append(record)
    base.write_gzip_jsonl(work_root / "xens2/predictions.jsonl.gz", records)
    coverage = {"schema_version": SCHEMA + ".coverage", "n_target": len(rows), "n_completed": len(records),
                "n_pass": sum(row["status"] == "PASS" for row in records),
                "n_blocked": sum(row["status"] != "PASS" for row in records),
                "blocked_reason_counts": {reason: sum(row["status"] == reason for row in records)
                    for reason in sorted({row["status"] for row in records if row["status"] != "PASS"})},
                "coverage": len(records) / len(rows), "remaining_count": len(remaining), "remaining_names": remaining,
                "predictions_sha256": base.sha256(work_root / "xens2/predictions.jsonl.gz"),
                "status": "COMPLETE" if not remaining else "PARTIAL"}
    base.write_json(work_root / "xens2/coverage.json", coverage)
    return coverage


def run(work_root, rows, probe=False):
    lock = base.acquire_lock(work_root, "xens2", " ".join(sys.argv))
    try:
        device = "cuda"
        store = EmbeddingStore.from_dir(str(EMBEDDING_DIR), split="archiveii")
        if probe:
            eligible = [row for row in rows if not protocol_block_reason(row, store)]
            if not eligible:
                raise ValueError("no eligible probe record")
            rows = [max(eligible, key=lambda row: len(row["seq"]))]
        identities = checkpoint_identities()
        embedding_sha256 = base.sha256(EMBEDDING_SHARD)
        enc0, head0 = load_plana(CKPTS["plana_s0"], device)
        enc1, head1 = load_plana(CKPTS["plana_s1"], device)
        model = load_r2d(CKPTS["r2d_s0"], device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for index, row in enumerate(rows):
            path = base.record_path(work_root, "xens2", row["name"])
            if path.is_file():
                validate_saved(json.loads(path.read_text()), row)
                continue
            reason = protocol_block_reason(row, store)
            if reason:
                record = blocked_record(row, reason)
            else:
                sequence = row["seq"]
                s0 = plana_scores(enc0, head0, sequence, device).cpu().numpy()
                s1 = plana_scores(enc1, head1, sequence, device).cpu().numpy()
                s2 = r2d_scores(model, sequence, store.get(sequence), device).cpu().numpy()
                for array in (s1, s2):
                    if array.shape != s0.shape:
                        array = array[:s0.shape[0], :s0.shape[1]]
                scores = 0.7 * (0.5 * s0 + 0.5 * s1) + 0.3 * s2
                mask = valid_pair_mask(sequence)
                scores = np.where(mask, scores, -np.inf)
                pairs = [tuple(pair) for pair in nussinov_map(scores, mask)]
                evidence = {"device": device, "device_name": torch.cuda.get_device_name(),
                            "checkpoint_sha256": identities,
                            "embedding_shard": str(EMBEDDING_SHARD), "embedding_shard_sha256": embedding_sha256}
                record = success_record(row, pairs, evidence)
            base.write_json(path, record)
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                coverage = consolidate(work_root, rows)
                print(json.dumps({"completed": coverage["n_completed"], "pass": coverage["n_pass"],
                                  "blocked": coverage["n_blocked"], "remaining": coverage["remaining_count"]}), flush=True)
        peak_mib = torch.cuda.max_memory_allocated() / (1 << 20)
        metadata = {"schema_version": SCHEMA + ".metadata", "device": device,
                    "device_name": torch.cuda.get_device_name(), "torch_peak_mib": peak_mib,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoints": identities,
                    "embedding_shard": str(EMBEDDING_SHARD), "embedding_shard_sha256": embedding_sha256,
                    "w_plana": 0.7, "plana_s1_weight": 0.5,
                    "protocol": "frozen xens2_eval; len>1024 or missing embeddings explicitly blocked, never substituted"}
        base.write_json(work_root / "xens2/predictor_metadata.json", metadata)
        coverage = consolidate(work_root, rows)
        if probe:
            base.write_json(work_root / "xens2/probe_result.json", {**metadata, "coverage": coverage})
        return coverage, metadata
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if args.status_only:
        print(json.dumps(consolidate(args.work_root, plan["rows"]), sort_keys=True))
        return
    coverage, metadata = run(args.work_root, plan["rows"], args.probe)
    print(json.dumps({"coverage": coverage, "torch_peak_mib": metadata["torch_peak_mib"]}, sort_keys=True))


if __name__ == "__main__":
    main()
