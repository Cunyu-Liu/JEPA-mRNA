#!/usr/bin/env python3
"""Export identity-bound raw pair scores for no-training decode diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
sys.path.insert(0, str(REPO / "tools"))
from xens2_eval import load_plana, load_r2d, plana_scores, r2d_scores  # noqa: E402
from rnajepa.train_decision import EmbeddingStore  # noqa: E402

SPLITS = ("bprna_vl0", "archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam")
EXPECTED_CHECKPOINT_SHA256 = {
    "rinalmo_r2dtr1c_b4_s0_step20000.pt": "47a823514111d116825f5eeef5e39d992abe45b3844d47cf174468bef3b345a3",
    "plana_giga_tr1c_s0_step20000.pt": "bd917b6380738e65ce46020eb52ab12333d2cd7a475ab066ed37379dc71828cd",
    "plana_giga_tr1c_s1_step20000.pt": "c10fb303b589346b806ff15ba1f4bb789af1ada3ac78a8010246f6d85c3296c8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resilient_r2d_scores(model, seq, emb, requested_device, prior_weight_override=None):
    """Use CPU after a CUDA OOM; keep the model on CPU to avoid unsafe restore attempts."""
    current_device = str(next(model.parameters()).device)
    device = "cpu" if current_device == "cpu" else requested_device
    try:
        scores = r2d_scores(model, seq, emb, device, prior_weight_override=prior_weight_override)
        used = "cpu_persistent" if device == "cpu" else str(device)
        return scores.detach().float().cpu().numpy(), used
    except torch.OutOfMemoryError:
        if device == "cpu":
            raise
        model.to("cpu")
        torch.cuda.empty_cache()
        scores = r2d_scores(model, seq, emb, "cpu", prior_weight_override=prior_weight_override)
        return scores.detach().float().cpu().numpy(), "cpu_oom_fallback"


def resilient_plana_scores(encoder, head, seq, requested_device, prior_weight_override=None):
    current_device = str(next(encoder.parameters()).device)
    device = "cpu" if current_device == "cpu" else requested_device
    try:
        scores = plana_scores(encoder, head, seq, device, prior_weight_override=prior_weight_override)
        used = "cpu_persistent" if device == "cpu" else str(device)
        return scores.detach().float().cpu().numpy(), used
    except torch.OutOfMemoryError:
        if device == "cpu":
            raise
        encoder.to("cpu")
        head.to("cpu")
        torch.cuda.empty_cache()
        scores = plana_scores(encoder, head, seq, "cpu", prior_weight_override=prior_weight_override)
        return scores.detach().float().cpu().numpy(), "cpu_oom_fallback"


def select_records(registry, dataset_id):
    rows = []
    with open(registry[dataset_id]["path"], encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            name = str(row.get("name", row.get("id", f"row:{index + 1}")))
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()
            selection_hash = hashlib.sha256(f"{dataset_id}|{name}|{seq_hash}".encode()).hexdigest()
            rows.append({"dataset_id": dataset_id, "row_index": index, "name": name, "seq": seq,
                         "sequence_sha256": seq_hash, "selection_sha256": selection_hash,
                         "gt_pairs": row.get("pairs", [])})
    if dataset_id != "bprna_vl0":
        rows = sorted(rows, key=lambda r: r["selection_sha256"])[:64]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(SPLITS))
    ap.add_argument("--members", choices=("r2d", "xens2"), default="r2d")
    ap.add_argument("--r2d-ckpt", default="/root/rna_jepa_data/ckpts_1014/rinalmo_r2dtr1c_b4_s0_step20000.pt")
    ap.add_argument("--plana-ckpt", default="/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s0_step20000.pt")
    ap.add_argument("--plana-ckpt2", default="/teams/HoulabData_1788502323/liucunyu/rna_jev_data/weights_20261013/ckpts/plana_giga_tr1c_s1_step20000.pt")
    ap.add_argument("--embedding-dir", default="/root/rna_jepa_data/embeddings/rinalmo-giga")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-chunk-size", type=int, default=64,
                    help="evaluation-only pair representation chunk; does not change the objective")
    ap.add_argument("--out", type=Path, default=ART / "raw_scores")
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    registry = {r["dataset_id"]: r for r in json.loads((REPO / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]}
    datasets = args.datasets.split(",")
    selected = {d: select_records(registry, d) for d in datasets}
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "rule": "all VL0; other splits first 64 by sha256(dataset_id|name|sequence_sha256)",
                 "members": args.members, "limit_per_dataset_smoke": args.limit_per_dataset or None,
                 "datasets": {d: [{k: v for k, v in r.items() if k not in ("seq", "gt_pairs")} for r in rows]
                              for d, rows in selected.items()}}
    (args.out / f"selection_{args.members}.json").write_text(json.dumps(selection, indent=2) + "\n")

    required = [Path(args.r2d_ckpt)]
    if args.members == "xens2":
        required += [Path(args.plana_ckpt), Path(args.plana_ckpt2)]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"identity-bound checkpoints missing: {missing}")
    checkpoint_hashes = {}
    for path in required:
        digest = sha256(path)
        expected = EXPECTED_CHECKPOINT_SHA256[path.name]
        if digest != expected:
            raise ValueError(f"checkpoint identity mismatch: {path} sha256={digest} expected={expected}")
        checkpoint_hashes[path.name] = digest

    import torch
    r2d = load_r2d(args.r2d_ckpt, args.device)
    r2d.head.chunk_size = args.head_chunk_size
    enc0 = enc1 = head0 = head1 = None
    if args.members == "xens2":
        enc0, head0 = load_plana(args.plana_ckpt, args.device)
        enc1, head1 = load_plana(args.plana_ckpt2, args.device)
        head0.chunk_size = head1.chunk_size = args.head_chunk_size
    score_keys = ["r2d_on", "r2d_off"] if args.members == "r2d" else ["plana0_on", "plana0_off", "plana1_on", "plana1_off", "r2d_on", "r2d_off", "xens2_on", "xens2_off"]
    metadata = {"schema_version": 1, "members": args.members, "device": args.device,
                "evaluation_head_chunk_size": args.head_chunk_size,
                "checkpoints": {p.name: {"path": str(p), "sha256": checkpoint_hashes[p.name]} for p in required},
                "w_plana": 0.7 if args.members == "xens2" else None,
                "plana_s1_weight": 0.5 if args.members == "xens2" else None,
                "datasets": {}}
    for dataset_id, records in selected.items():
        split_dir = args.out / dataset_id
        split_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = EmbeddingStore.from_dir(args.embedding_dir, split=dataset_id)
            store_error = None
        except Exception as exc:
            store = None
            store_error = repr(exc)
        info = {"n_selected": len(records), "n_exported": 0, "n_resumed": 0,
                "n_skipped_length": 0, "n_skipped_embedding": 0, "embedding_store_error": store_error,
                "files": []}
        for record_index, record in enumerate(records):
            if args.limit_per_dataset and record_index >= args.limit_per_dataset:
                break
            seq, name = record["seq"], record["name"]
            if len(seq) > 1024:
                info["n_skipped_length"] += 1
                continue
            out_path = split_dir / f"{record['sequence_sha256']}.npz"
            if args.resume and out_path.exists():
                with np.load(out_path, allow_pickle=False) as old:
                    if str(old["name"]) == name and str(old["seq"]) == seq and all(k in old for k in score_keys):
                        if "r2d_on_device" not in old or "evaluation_head_chunk_size" not in old:
                            payload = {k: old[k] for k in old.files}
                            payload.update({"r2d_on_device": np.array("cuda"), "r2d_off_device": np.array("cuda"),
                                            "evaluation_head_chunk_size": np.array(args.head_chunk_size)})
                            np.savez_compressed(out_path, **payload)
                            info["n_metadata_backfilled"] = info.get("n_metadata_backfilled", 0) + 1
                        info["n_resumed"] += 1
                        continue
            emb = store.get(seq) if store is not None else None
            if emb is None:
                info["n_skipped_embedding"] += 1
                continue
            r2d_on, on_device = resilient_r2d_scores(r2d, seq, emb, args.device)
            r2d_off, off_device = resilient_r2d_scores(r2d, seq, emb, args.device, prior_weight_override=0)
            arrays = {"r2d_on": r2d_on, "r2d_off": r2d_off}
            if args.members == "xens2":
                p0on, p0on_device = resilient_plana_scores(enc0, head0, seq, args.device)
                p0off, p0off_device = resilient_plana_scores(enc0, head0, seq, args.device, prior_weight_override=0)
                p1on, p1on_device = resilient_plana_scores(enc1, head1, seq, args.device)
                p1off, p1off_device = resilient_plana_scores(enc1, head1, seq, args.device, prior_weight_override=0)
                arrays.update({"plana0_on": p0on, "plana0_off": p0off, "plana1_on": p1on, "plana1_off": p1off})
                arrays["xens2_on"] = .7 * (.5 * p0on + .5 * p1on) + .3 * r2d_on
                arrays["xens2_off"] = .7 * (.5 * p0off + .5 * p1off) + .3 * r2d_off
            payload = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
            if args.members == "xens2":
                payload.update({"plana0_on_device": np.array(p0on_device), "plana0_off_device": np.array(p0off_device),
                                "plana1_on_device": np.array(p1on_device), "plana1_off_device": np.array(p1off_device)})
            payload.update({"name": np.array(name), "seq": np.array(seq),
                            "sequence_sha256": np.array(record["sequence_sha256"]),
                            "selection_sha256": np.array(record["selection_sha256"]),
                            "r2d_on_device": np.array(on_device), "r2d_off_device": np.array(off_device),
                            "evaluation_head_chunk_size": np.array(args.head_chunk_size),
                            "gt_pairs": np.asarray(record["gt_pairs"], dtype=np.int32).reshape(-1, 2)})
            if on_device.startswith("cpu"):
                info["n_cpu_on"] = info.get("n_cpu_on", 0) + 1
            if off_device.startswith("cpu"):
                info["n_cpu_off"] = info.get("n_cpu_off", 0) + 1
            np.savez_compressed(out_path, **payload)
            info["n_exported"] += 1
            info["files"].append({"path": str(out_path), "name": name, "sequence_sha256": record["sequence_sha256"],
                                  "r2d_on_device": on_device, "r2d_off_device": off_device})
            if info["n_exported"] % 25 == 0:
                print(f"[raw-export] {dataset_id} {info['n_exported']} exported", flush=True)
        metadata["datasets"][dataset_id] = info
        (args.out / f"metadata_{args.members}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["code"] = str(Path(__file__).resolve())
    metadata["code_sha256"] = sha256(Path(__file__).resolve())
    metadata["git_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    (args.out / f"metadata_{args.members}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"members": args.members, "datasets": {d: {k: v for k, v in x.items() if k != "files"} for d, x in metadata["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
