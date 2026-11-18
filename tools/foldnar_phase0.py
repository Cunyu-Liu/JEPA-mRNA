#!/usr/bin/env python3
"""Freeze the FoldNAR environment and build data/prediction/baseline registries."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
JSONL = Path("/root/rna_jepa_data/ss_data/jsonl")
EVAL = Path("/root/rna_jepa_data/eval_decision")
EXT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
MAIN_SPLITS = ("archiveii_embok_clean", "bprna_new", "bprna_ts0", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb")
CHANRG_SPLITS = ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, content_hash: bool = True) -> dict:
    rec = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        rec["bytes"] = path.stat().st_size
        rec["sha256"] = sha256(path) if content_hash else None
        rec["hash_scope"] = "full_content" if content_hash else "not_computed"
    return rec


def jsonl_counts(path: Path) -> dict:
    parsed = 0
    ids: set[str] = set()
    duplicate_ids = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed += 1
            identity = str(row.get("name", row.get("id", f"row:{parsed}")))
            if identity in ids:
                duplicate_ids += 1
            ids.add(identity)
    newlines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            newlines += block.count(b"\n")
    return {"json_parse": parsed, "binary_newlines": newlines, "unique_ids": len(ids), "duplicate_ids": duplicate_ids}


def manifest_count(path: Path) -> int | None:
    manifest = path.with_suffix(".manifest.json")
    if not manifest.is_file():
        return None
    obj = json.loads(manifest.read_text())
    for key in ("n_records", "n_sequences", "count", "written", "n"):
        if isinstance(obj.get(key), int):
            return obj[key]
    for key in ("counts", "summary"):
        if isinstance(obj.get(key), dict):
            for subkey in ("n_records", "n_sequences", "count", "written", "n"):
                if isinstance(obj[key].get(subkey), int):
                    return obj[key][subkey]
    return None


def data_specs() -> list[tuple[str, str, str]]:
    specs = [
        ("bprna_tr1c", "train", "bpRNA_TR1c"),
        ("bprna_vl0", "val", "bpRNA_VL0"),
        ("ref_tr_bprna", "train", "TR0/TR1 bpRNA reference"),
        ("ref_tr_inter", "train", "inter-molecular experimental"),
        ("ref_tr_intra", "train", "intra-molecular experimental"),
        ("ref_tr_experimental", "train", "experimental reference union"),
        ("ref_syn_train", "train", "synthetic reference"),
        ("chanrg_train_train", "train", "CHANRG train"),
        ("chanrg_train_validation", "val", "CHANRG validation"),
        ("ncrna_rfamnovel_train", "train", "ncRNA/RfamNovel train"),
        ("ncrna_rfamnovel_val", "val", "ncRNA/RfamNovel validation"),
    ]
    specs += [(name, "test", name) for name in MAIN_SPLITS]
    specs += [(name, "test", name) for name in CHANRG_SPLITS]
    specs += [("rfam_temporal", "test", "Rfam temporal"), ("rfam_fam", "test", "Rfam family"),
              ("ncrna_rfamnovel_test", "test", "ncRNA/RfamNovel test")]
    return [(name, role, desc) for name, role, desc in specs]


def build_data_registry() -> list[dict]:
    rows = []
    for name, role, description in data_specs():
        path = JSONL / f"{name}.jsonl"
        rec = {"dataset_id": name, "role": role, "description": description, **file_info(path)}
        if path.is_file():
            counts = jsonl_counts(path)
            rec["n_sequences"] = counts["json_parse"]
            rec["count_verification"] = {**counts, "adjacent_manifest": manifest_count(path)}
            rec["count_consistent"] = counts["json_parse"] == counts["binary_newlines"] and (
                rec["count_verification"]["adjacent_manifest"] is None
                or rec["count_verification"]["adjacent_manifest"] == counts["json_parse"]
            )
            rec["status"] = "FROZEN" if rec["count_consistent"] else "QA_MISMATCH"
        else:
            rec["status"] = "BLOCKED"
            rec["block_reason"] = "JSONL not found"
        rows.append(rec)
    # The frozen cache is registered by an index hash, not a misleading partial content hash.
    cache = Path("/root/rna_jepa_data/embeddings/rinalmo-giga")
    index_rows = []
    if cache.is_dir():
        for p in sorted(cache.rglob("*.npz")):
            st = p.stat()
            index_rows.append(f"{p.relative_to(cache)}\t{st.st_size}\t{st.st_mtime_ns}")
    index_payload = "\n".join(index_rows).encode()
    rows.append({
        "dataset_id": "rinalmo_giga_embedding_cache", "role": "derived_cache",
        "description": "RiNALMo-giga frozen embedding store; index covers every NPZ",
        "path": str(cache), "exists": cache.is_dir(), "n_files": len(index_rows),
        "index_sha256": hashlib.sha256(index_payload).hexdigest(), "hash_scope": "path_size_mtime_index",
        "status": "FROZEN" if cache.is_dir() else "BLOCKED",
    })
    return rows


def prediction_row(model: str, split: str, path: Path, repeat: int | None = None) -> dict:
    identity = f"{model}/{split}" + (f"/repeat{repeat}" if repeat is not None else "")
    rec = {"prediction_id": identity, "model": model, "split": split, "repeat": repeat, **file_info(path)}
    if not path.is_file():
        rec.update({"status": "BLOCKED", "block_reason": "registered prediction file missing"})
        return rec
    try:
        obj = json.loads(path.read_text())
        per = obj.get("per_sequence")
        if per is None and isinstance(obj.get("predictions"), list):
            per = obj["predictions"]
        if not isinstance(per, list):
            raise ValueError("no per_sequence/predictions list")
        rec["n_sequences"] = len(per)
        rec["n_sequences_field"] = obj.get("n_sequences", obj.get("n"))
        rec["n_skipped"] = obj.get("n_skipped")
        rec["data_path"] = obj.get("data") or str(JSONL / f"{split}.jsonl")
        rec["prediction_schema"] = "per_sequence" if "per_sequence" in obj else "predictions"
        rec["model_metadata"] = {k: obj.get(k) for k in (
            "tag", "checkpoint", "plana_ckpt", "plana_ckpt2", "r2d_ckpt", "w_plana",
            "plana_s1_weight", "system", "distribution", "weights_sha256", "threshold",
        ) if k in obj}
        rec["count_consistent"] = rec["n_sequences_field"] is None or rec["n_sequences_field"] == len(per)
        rec["status"] = "FROZEN" if rec["count_consistent"] else "QA_MISMATCH"
    except Exception as exc:
        rec.update({"status": "BLOCKED", "block_reason": f"schema/read error: {exc}"})
    return rec


def build_prediction_registry() -> list[dict]:
    rows = []
    for split in MAIN_SPLITS:
        rows.append(prediction_row("xens2", split, EVAL / "xens2_board_4090" / f"{split}.json"))
        rows.append(prediction_row("rinalmo_2d_s0", split, EVAL / "nb_jev_r2dtr1c_s0_step20000" / split / "result.json"))
        for model, directory in (("ufold", "ufold_mm_corrected"), ("mxfold2", "mxfold2"),
                                 ("rinalmo_ft", "rinalmo_ft"), ("eternafold", "eternafold")):
            rows.append(prediction_row(model, split, EXT / directory / "raw" / f"{split}.json"))
        # The verified full ArchiveII MXfold2 result lives in raw_full.
        if split == "archiveii_embok_clean":
            rows[-3] = prediction_row("mxfold2", split, EXT / "mxfold2" / "raw_full" / f"{split}.json")
    for split in CHANRG_SPLITS:
        rows.append(prediction_row("xens2", split, EVAL / split / "xens2" / "result.json"))
        rows.append(prediction_row("rinalmo_2d_s0", split, EVAL / split / "r2d" / "result.json"))
        rows.append(prediction_row("chanrg_ens3", split, EVAL / "chanrg_ens3_gt" / split / "result.json"))
    for repeat in range(3):
        rows.append(prediction_row("xens2", "rfam_fam", EVAL / "round42_rfam" / "xens2" / f"repeat{repeat}.json", repeat))
    rows.append({"prediction_id": "xens2/rfam_temporal", "model": "xens2", "split": "rfam_temporal",
                 "repeat": None, "status": "BLOCKED", "block_reason": "no frozen per-sequence prediction found"})
    return rows


def build_baseline_registry() -> list[dict]:
    source = REPO / "source_data"
    entries = [
        ("xens2", "frozen_ensemble", "VL0-selected Plan-A s0/s1 x RiNALMo-2D s0",
         ART / "artifacts/p0/xens2_bprna_new_replay/run_manifest.json", "COMPLETE_RAW"),
        ("rinalmo_2d_s0", "own_best_single", "RiNALMo-2D frozen embedding downstream model",
         ART / "manifests/p0/xens2_r2d_identity.json", "ARCHIVEII_FULL_BLOCKED"),
        ("ufold", "external_replay", "MultiMolecule conversion; official-weight provenance limitation retained",
         source / "appendix_leakage/round39_ufold_raw_summary.json", "RAW_COMPLETE_CLEAN_BLOCKED"),
        ("mxfold2", "external_replay", "Official MXfold2 raw full-length replay",
         source / "appendix_leakage/round39_mxfold2_raw_summary.json", "RAW_COMPLETE_CLEAN_PENDING"),
        ("rinalmo_ft", "external_replay", "Full fine-tuned RiNALMo downstream baseline",
         source / "appendix_leakage/round39_rinalmo_ft_raw_summary.json", "RAW_COMPLETE_CLEAN_PENDING"),
        ("eternafold", "external_replay", "Official EternaFold serial CPU parameters",
         source / "appendix_leakage/round39_eternafold_raw_summary.json", "RAW_COMPLETE"),
        ("efold", "external_replay", "eFold 0.1.3; 0.1.2 records are not interchangeable",
         source / "fig05_speed/round38_baseline_assets.json", "TS0_ONLY"),
        ("spot_rna", "external_replay", "Converted weights; not original-author distribution",
         source / "appendix_leakage/round39_spotrna_raw_summary.json", "SUMMARY_ONLY_IN_E0"),
        ("vienna", "external_reference", "ViennaRNA reference",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("rnastructure", "external_reference", "RNAstructure reference",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("contrafold", "external_reference", "CONTRAfold reference",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("kirigami", "external_reference", "Kirigami reference",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("linearfold", "external_reference", "LinearFold reference",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("nucleicbert_frozen", "external_reference", "NucleicBERT frozen baseline",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
        ("nucleicbert_ft", "external_reference", "NucleicBERT fine-tuned baseline",
         REPO / "records/BASELINE_RESULTS.md", "NOT_REGISTERED_FOR_E0"),
    ]
    rows = []
    for model, role, version, evidence, status in entries:
        rows.append({"model": model, "role": role, "version_protocol": version,
                     "evidence": file_info(evidence), "status": status,
                     "prediction_registry_scope": "E0 raw" if status.startswith(("COMPLETE", "RAW")) else "limited/blocked"})
    return rows


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--out-phase", default="phase0")
    args = parser.parse_args()
    out = args.artifact_root / "artifacts/foldnar_v1" / args.out_phase
    manifests = args.repo / "manifests/foldnar_v1"
    targets = [manifests / name for name in ("data_registry.json", "prediction_registry.json", "baseline_registry.json")]
    targets.append(out / "environment.json")
    if any(p.exists() for p in targets):
        raise FileExistsError("refusing to overwrite a frozen Phase-0 registry")
    out.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    data = build_data_registry()
    predictions = build_prediction_registry()
    baselines = build_baseline_registry()
    for path, rows in zip(targets, (data, predictions, baselines)):
        payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "n_rows": len(rows), "rows": rows}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # Three independent registry checks: declared rows, successful file/JSON counts, and reloaded rows.
    verification = {
        "data": {
            "declared_rows": len(data),
            "jsonl_parse_total": sum(r.get("n_sequences", 0) for r in data),
            "binary_newline_total": sum(r.get("count_verification", {}).get("binary_newlines", 0) for r in data),
            "reloaded_rows": len(json.loads(targets[0].read_text())["rows"]),
            "mismatches": [r["dataset_id"] for r in data if r.get("status") == "QA_MISMATCH"],
        },
        "predictions": {
            "declared_rows": len(predictions),
            "frozen_rows": sum(r["status"] == "FROZEN" for r in predictions),
            "blocked_rows": sum(r["status"] == "BLOCKED" for r in predictions),
            "per_sequence_total": sum(r.get("n_sequences", 0) for r in predictions),
            "reloaded_rows": len(json.loads(targets[1].read_text())["rows"]),
            "mismatches": [r["prediction_id"] for r in predictions if r.get("status") == "QA_MISMATCH"],
        },
        "baselines": {
            "declared_rows": len(baselines),
            "existing_evidence": sum(r["evidence"]["exists"] for r in baselines),
            "reloaded_rows": len(json.loads(targets[2].read_text())["rows"]),
        },
    }
    verification["pass"] = (
        verification["data"]["declared_rows"] == verification["data"]["reloaded_rows"]
        and verification["predictions"]["declared_rows"] == verification["predictions"]["reloaded_rows"]
        and verification["baselines"]["declared_rows"] == verification["baselines"]["reloaded_rows"]
        and verification["baselines"]["declared_rows"] == verification["baselines"]["existing_evidence"]
        and verification["data"]["jsonl_parse_total"] == verification["data"]["binary_newline_total"]
        and not verification["data"]["mismatches"] and not verification["predictions"]["mismatches"]
    )
    (out / "registry_verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    weight_paths = [
        ART.parent / "rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt",
        ART.parent / "rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt",
        ART.parent / "rna_jev_data/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt",
        ART.parent / "rna_jev_data/ckpts_1014/rinalmo_r2dtr1c_b4_s1_step20000.pt",
        ART.parent / "rna_jev_data/ckpts_1014/rinalmo_r2dtr1c_b4_s2_step20000.pt",
        Path("/root/rna_jepa_data/backbone/rinalmo-giga/model.safetensors"),
    ]
    code_paths = [args.repo / p for p in (
        "src/rnajepa/fold_structure.py", "tools/foldnar_phase0.py", "tools/xens2_eval.py", "eval/ss/evaluate_decision.py",
        "eval/ss/metrics.py", "configs/ss/decision_r2d.yaml",
        "paper/2026-11-18-rna-jev-strict-scientific-engineering-audit.md",
    )]
    snapshot = out / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in code_paths:
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".md"}:
            shutil.copy2(path, snapshot / path.name)
    env = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "repo": str(args.repo),
        "branch": run(["git", "-C", str(args.repo), "branch", "--show-current"]),
        "head": run(["git", "-C", str(args.repo), "rev-parse", "HEAD"]),
        "freeze_head": run(["git", "-C", str(args.repo), "rev-parse", "HEAD"]),
        "git_status": run(["git", "-C", str(args.repo), "status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.import_module(name).__version__ for name in ("numpy", "scipy", "pandas", "pyarrow", "zss")},
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]),
        "code_inputs": [file_info(p) for p in code_paths],
        "weights": [file_info(p, content_hash=p.is_file()) for p in weight_paths],
        "registries": [file_info(p) for p in targets[:3]],
        "verification": verification,
    }
    targets[3].write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS" if verification["pass"] else "FAIL",
                      "data_rows": len(data), "prediction_rows": len(predictions),
                      "frozen_predictions": verification["predictions"]["frozen_rows"],
                      "blocked_predictions": verification["predictions"]["blocked_rows"],
                      "baseline_rows": len(baselines), "out": str(out)}, indent=2))
    return 0 if verification["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
