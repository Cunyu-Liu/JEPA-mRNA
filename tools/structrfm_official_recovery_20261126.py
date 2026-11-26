#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_REPOSITORY = "https://github.com/heqin-zhu/structRFM"
RELEASE_TAG = "v0.0.8"
RELEASE_COMMIT = "35c5f416e0f8395c459ae65c691200a2ddff3eb3"
INFERENCE_COMMIT = "f0c1add19f8dfc786f16bc4abb7aa775dbadddbf"
ASSET_NAME = "structRFM_SSP_bpRNA1m.pt"
ASSET_BYTES = 348179698
ASSET_SHA256 = "689e09e0ad3b51af19b03cb32ea3be5237384c3c9aa5716dff29989098e12145"
ASSET_URL = f"{OFFICIAL_REPOSITORY}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
DEFAULT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/structrfm_official_v0.0.8")
DEFAULT_SOURCE = Path("/home/rna_jev/third_party/structRFM")
DEFAULT_DATASET = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATASET_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
HISTORICAL_WEIGHT = Path("/mnt/cunyuliu/rna-jepa/refmodels/models/structRFM_SSP_bpRNA1m.pt")
HISTORICAL_PREDICTION = Path("/root/rna_jepa_data/eval_decision_ref/structrfm_bprna_new.json")
HISTORICAL_PREDICTION_SHA256 = "3243b141eed79a8c419bc737dece47943fd7708f2db5cd881523dcc9df1d6519"
LICENSE_SHA256 = "cb4768592947e40d6ef70a65e393eaee5cf33b139e45cfa641ff0cd04b33224b"
SOURCE_FILES = {
    "LICENSE": LICENSE_SHA256,
    "src/structRFM/model/structRFM.py": "17088a2f4662a6ba73b8e63044dcfafc5f4158315a3ef16074403471f1e4cd56",
    "src/structRFM/model/__init__.py": "d392860a86ca5988ff815449b3924367988c0b8851a8040d763e84e009566881",
    "src/structRFM/data/tokenizer.py": "8461ff17347cbb8f386266f00ae2f7845702745bb376acfb0612b414157eceff",
    "src/structRFM/data/tokenizer_mlm.json": "fe018d5c215e53d07a435dc33a6e968344da967fb36e26e6bb5951743e6d76dc",
    "tasks/seqcls_ssp/ss_pred.py": "f5c981dcfc051ad94c1698e03400dbb67d74f49e4eb6ad104de89d578144a55e",
    "tasks/seqcls_ssp/base_classes.py": "d89853503d77811eae5b58d820913cef5cf6deef5de4adfa08d72d3e3dab48d0",
    "tasks/seqcls_ssp/utils.py": "854965ba58053c72c7cfb6ac5f1acd6330df9c7ff66d4bce855c568dd43fe5b1",
    "tasks/seqcls_ssp/param_turner2004.py": "9b8e58972ed5a8fa6795a55b5a9fc01c881771baaec1bd4a7732e76dc7c2f3ac",
    "tasks/seqcls_ssp/interface.cpython-38-x86_64-linux-gnu.so": "7d332a297216ecc0d5ccbeec356c42876b543d089cba8846f478ebbaec4ea007",
}
MODEL_CONFIG = {
    "max_helix_length": 30,
    "embed_size": 64,
    "num_filters": (64, 64, 64, 64, 64, 64, 64, 64),
    "filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
    "pool_size": (1,),
    "dilation": 0,
    "num_lstm_layers": 2,
    "num_lstm_units": 32,
    "num_transformer_layers": 0,
    "num_hidden_units": (32,),
    "num_paired_filters": (64, 64, 64, 64, 64, 64, 64, 64),
    "paired_filter_size": (5, 3, 5, 3, 5, 3, 5, 3),
    "dropout_rate": 0.5,
    "fc_dropout_rate": 0.5,
    "num_att": 8,
    "pair_join": "cat",
    "no_split_lr": False,
    "n_out_paired_layers": 3,
    "n_out_unpaired_layers": 0,
    "exclude_diag": True,
    "embed_dim": 768,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def normalize_sequence(sequence):
    return sequence.upper().replace("T", "U")


def pair_set(values, length):
    pairs = set()
    partners = set()
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid pair encoding: {value}")
        i, j = value
        if type(i) is not int or type(j) is not int or not 0 <= i < j < length:
            raise ValueError(f"Invalid pair indices: {value}")
        if i in partners or j in partners:
            raise ValueError(f"Duplicate partner: {value}")
        partners.update((i, j))
        pairs.add((i, j))
    return pairs


def load_dataset(path=DEFAULT_DATASET, require_full=True):
    path = Path(path)
    if require_full and sha256(path) != DATASET_SHA256:
        raise ValueError("bpRNA-new identity hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate bpRNA-new names")
    if require_full and (len(rows) != 5388 or sum(len(row["pairs"]) for row in rows) != 135184):
        raise ValueError("bpRNA-new must contain 5388 rows and 135184 reference pairs")
    for row in rows:
        sequence = normalize_sequence(row["seq"])
        if not row["name"] or not sequence or set(sequence) - set("ACGUN"):
            raise ValueError(f"Invalid name or sequence: {row['name']}")
        pair_set(row["pairs"], len(sequence))
    return rows


def pairs_from_connects(connects, sequence):
    if len(connects) != len(sequence):
        raise ValueError("Connects length does not match sequence")
    directed = []
    for i, value in enumerate(connects):
        partner = int(value)
        if partner == 0:
            continue
        j = partner - 1
        if not 0 <= j < len(connects) or j == i:
            raise ValueError(f"Invalid one-based partner: {i + 1}->{partner}")
        if int(connects[j]) != i + 1:
            raise ValueError(f"Non-reciprocal partner: {i + 1}->{partner}")
        directed.append((min(i, j), max(i, j)))
    return sorted(pair_set(set(directed), len(sequence)))


def validate_prediction_rows(dataset_rows, prediction_rows):
    if len(prediction_rows) != len(dataset_rows):
        raise ValueError("Prediction row count does not match input")
    output = []
    for source, prediction in zip(dataset_rows, prediction_rows):
        if prediction.get("name") != source["name"]:
            raise ValueError("Prediction order or name identity mismatch")
        if normalize_sequence(prediction.get("seq", "")) != normalize_sequence(source["seq"]):
            raise ValueError(f"Prediction sequence mismatch: {source['name']}")
        pairs = pairs_from_connects(prediction["connects"], source["seq"]) if "connects" in prediction else sorted(pair_set(prediction["pred_pairs"], len(source["seq"])))
        output.append({"name": source["name"], "seq": normalize_sequence(source["seq"]), "pred_pairs": [list(pair) for pair in pairs]})
    if len({row["name"] for row in output}) != len(dataset_rows):
        raise ValueError("Duplicate prediction names")
    return output


def validate_checkpoint(path=DEFAULT_ROOT / ASSET_NAME):
    path = Path(path)
    if path.name != ASSET_NAME or not path.is_file():
        raise ValueError(f"Missing official checkpoint: {path}")
    if path.stat().st_size != ASSET_BYTES:
        raise ValueError("Official checkpoint byte count mismatch")
    actual = sha256(path)
    if actual != ASSET_SHA256:
        raise ValueError(f"Official checkpoint SHA256 mismatch: {actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual, "official_url": ASSET_URL}


def inspect_checkpoint(path=DEFAULT_ROOT / ASSET_NAME):
    identity = validate_checkpoint(path)
    import torch
    document = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(document, dict) or not {"model", "pretrained_model"}.issubset(document):
        raise ValueError("Official checkpoint must contain model and pretrained_model state dictionaries")
    states = {}
    for key in ("model", "pretrained_model"):
        state = document[key]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Empty checkpoint state: {key}")
        tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
        if len(tensors) != len(state):
            raise ValueError(f"Non-tensor state entry: {key}")
        states[key] = {"n_tensors": len(tensors), "n_parameters": sum(value.numel() for value in tensors)}
    return {**identity, "load_device": "cpu", "torch_version": torch.__version__, "state_dicts": states, "top_level_keys": sorted(document)}


def validate_source(source=DEFAULT_SOURCE):
    source = Path(source)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != INFERENCE_COMMIT:
        raise ValueError(f"Inference source commit mismatch: {commit}")
    release_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "v0.0.8^{}"], text=True).strip()
    if release_commit != RELEASE_COMMIT:
        raise ValueError(f"Release tag commit mismatch: {release_commit}")
    files = {}
    for name, expected in SOURCE_FILES.items():
        path = source / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Official source file hash mismatch: {name}")
        files[name] = actual
    return {"repository": OFFICIAL_REPOSITORY, "path": str(source), "release_tag": RELEASE_TAG, "release_commit": release_commit, "inference_commit": commit, "inference_package_version": "0.0.9", "file_sha256": files}


def historical_weight_audit():
    if not HISTORICAL_WEIGHT.is_file():
        return {"status": "MISSING_HISTORICAL_WEIGHT_BYTES", "path": str(HISTORICAL_WEIGHT), "sha256": None, "matches_official": None}
    actual = sha256(HISTORICAL_WEIGHT)
    return {"status": "BYTE_COMPARED", "path": str(HISTORICAL_WEIGHT), "sha256": actual, "matches_official": actual == ASSET_SHA256}


def build_manifest(root=DEFAULT_ROOT, dataset_path=DEFAULT_DATASET, source=DEFAULT_SOURCE, inspect_weights=True):
    root = Path(root)
    checkpoint = inspect_checkpoint(root / ASSET_NAME) if inspect_weights else validate_checkpoint(root / ASSET_NAME)
    release_path = root / "release_v0.0.8.json"
    release = json.loads(release_path.read_text())
    if release.get("tag_name") != RELEASE_TAG:
        raise ValueError("Official release metadata tag mismatch")
    asset = next(row for row in release["assets"] if row["name"] == ASSET_NAME)
    if asset["size"] != ASSET_BYTES or asset.get("digest") != f"sha256:{ASSET_SHA256}" or asset["browser_download_url"] != ASSET_URL:
        raise ValueError("Official release asset metadata mismatch")
    rows = load_dataset(dataset_path)
    historical = historical_weight_audit()
    if historical["matches_official"] is False:
        historical_status = "HISTORICAL_WEIGHT_BYTES_DIFFER_DO_NOT_REUSE_AS_OFFICIAL"
    elif historical["matches_official"] is True:
        historical_status = "HISTORICAL_WEIGHT_BYTES_MATCH_BUT_NEW_PREDICTION_STILL_REQUIRED_FOR_THIS_RUN"
    else:
        historical_status = "HISTORICAL_PREDICTION_UNLINKED_DO_NOT_USE_AS_VERIFIED_OFFICIAL"
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "baseline_id": "structrfm",
        "status": "OFFICIAL_CHECKPOINT_RECOVERED_GPU_RUNTIME_AND_PREDICTION_PENDING",
        "full_ready": False,
        "asset_ready": True,
        "ready_scope": "Official v0.0.8 release checkpoint bytes, release/source identity, model/training/single-model attributes and bpRNA-new 5388 I/O contract",
        "blocked_scope": "No GPU inference or new verified-official predictions; CPython 3.8 official DP runtime must be supplied; historical weight bytes are unavailable for retrospective linkage",
        "source": validate_source(source),
        "release": {"url": f"{OFFICIAL_REPOSITORY}/releases/tag/{RELEASE_TAG}", "api_snapshot": str(release_path), "api_snapshot_sha256": sha256(release_path), "published_at": release["published_at"]},
        "checkpoint": checkpoint,
        "license": {"code_spdx": "MIT", "source_path": str(Path(source) / "LICENSE"), "source_sha256": LICENSE_SHA256, "weight_status": "NO_SEPARATE_WEIGHT_LICENSE_FOUND; official asset distributed under the MIT-licensed project release; do not infer dataset licenses"},
        "model_identity": {"family": "structRFM", "checkpoint_version": RELEASE_TAG, "task_checkpoint": "bpRNA1m secondary-structure fine-tune", "encoder": "base: dim=768, layers=12, attention_heads=12, tokenizer_max_length=514", "head": "structRFMForSsp embeddings -> MixedFold CNN/LSTM pair scorer + Turner parameters -> interface.predict_mxfold constrained DP", "model_form": "single_checkpoint_single_model", "ensemble": False, "n_checkpoints": 1, "batch_size": 1, "dropout_disabled_by_eval": True, "mixedfold_config": {key: list(value) if isinstance(value, tuple) else value for key, value in MODEL_CONFIG.items()}},
        "training_exposure": {"status": "AUTHOR_CORPORA_IDENTIFIED_BPRNA_NEW_CLEARANCE_UNRESOLVED", "pretraining": "About 21M RNAcentral sequences with BPfold-derived structures, length <=512; official README and Zenodo DOI 10.5281/zenodo.16754363", "downstream": "bpRNA1m secondary-structure fine-tune; exact checkpoint-specific union not reconstructed here", "project_training_dataset_ids": [], "bprna_new_used_for_training_or_tuning_by_this_recovery": False, "raw_not_clean": True, "claim": "No exact/homology decontamination or bpRNA-new non-exposure claim is made"},
        "bprna_new_5388_contract": {"dataset_path": str(dataset_path), "dataset_sha256": sha256(dataset_path), "n_sequences": len(rows), "n_reference_pairs": sum(len(row["pairs"]) for row in rows), "length_min": min(len(row["seq"]) for row in rows), "length_max": max(len(row["seq"]) for row in rows), "n_over_511": sum(len(row["seq"]) > 511 for row in rows), "input_order": "frozen JSONL order and unique names", "normalization": "uppercase T->U; [CLS]+sequence; B=1; max_seq_len=512 with at most 511 residues", "output": "one JSONL row per input in identical order: {name,seq,pred_pairs}; zero-based inclusive i<j; reciprocal one-based connects accepted only at conversion boundary", "labels_excluded_from_inference_input": True, "summary_only_substitution_forbidden": True, "predictions": None, "prediction_sha256": None, "coverage": None, "metrics": None},
        "historical_artifacts": {"weight": historical, "prediction_status": historical_status, "prediction_path": str(HISTORICAL_PREDICTION), "prediction_sha256": sha256(HISTORICAL_PREDICTION), "historical_metrics_retained_as_unlinked_history_only": True, "eligible_as_new_official_prediction": False},
        "gpu_execution": {"performed": False, "launcher": "tools/run_structrfm_bprna_new_gpu_20261126.py", "runtime_requirement": "CPython 3.8 x86_64 for official interface.cpython-38-x86_64-linux-gnu.so plus compatible torch/transformers/numpy", "occupied_gpu_policy": "No automatic execution; explicit --execute required"},
        "provenance_policy": "No summary-only score, unlinked historical replay, partial download, differing-weight prediction, or reference label may be substituted for a new official-checkpoint prediction",
    }


def update_registries(repo, manifest):
    repo = Path(repo)
    baseline_path = repo / "manifests/foldnar_v1/baseline_registry.json"
    baseline_doc = json.loads(baseline_path.read_text())
    baseline = next(row for row in baseline_doc["rows"] if row["baseline_id"] == "structrfm")
    baseline["version_protocol"] = "structRFM v0.0.8 bpRNA1m SSP; single checkpoint; official inference source f0c1add; B=1; max_seq_len=512; MixedFold constrained DP"
    baseline["official_weights"] = {"status": "VERIFIED_OFFICIAL_RELEASE_BYTES", "path": manifest["checkpoint"]["path"], "bytes": ASSET_BYTES, "sha256": ASSET_SHA256, "official_url": ASSET_URL, "release_tag": RELEASE_TAG}
    baseline["training_data"] = {"status": "AUTHOR_CORPORA_IDENTIFIED_BPRNA_NEW_EXPOSURE_UNRESOLVED", "project_training_dataset_ids": [], "external_training_corpora": ["RNAcentral + BPfold-derived ~21M sequence-structure pairs (length<=512)", "bpRNA1m downstream secondary-structure fine-tune"], "raw_not_clean": True}
    baseline["license"] = {"status": "OFFICIAL_REPOSITORY_MIT_NO_SEPARATE_WEIGHT_LICENSE", "spdx": "MIT", "scope": "official source/release; no separate weight license found", "source_sha256": LICENSE_SHA256}
    baseline["inference"] = {"status": "OFFICIAL_WEIGHT_VERIFIED_GPU_PREDICTION_PENDING", "version_protocol": baseline["version_protocol"], "decoder": "MixedFold Turner + constrained DP (interface.predict_mxfold)", "threshold": None, "evaluation_scope": "bpRNA-new raw 5388; no new prediction yet", "model_form": "single_checkpoint_not_ensemble", "ensemble": False}
    baseline["comparability"]["status"] = "HISTORICAL_RAW_REPLAY_UNLINKED; new official-checkpoint raw predictions pending; raw is not clean"
    baseline["provenance_status"] = "PARTIAL_OFFICIAL_WEIGHT_VERIFIED_RUNTIME_AND_PREDICTION_PENDING"
    baseline["status"] = "OFFICIAL_WEIGHT_RECOVERED_GPU_PREDICTION_PENDING"
    baseline["inclusion_reason"] = "required historical 25 coverage; official v0.0.8 checkpoint recovered, new official 5388 prediction pending"
    baseline["prediction_registry_scope"] = "HISTORICAL_UNLINKED_NOT_VERIFIED_OFFICIAL"
    baseline["n_frozen_prediction_rows"] = 0
    baseline["n_blocked_prediction_rows"] = 1
    baseline["bprna_new_recovery"] = {"registry": "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json", "asset_ids": ["structrfm_author_bprna1m_single_raw"], "status": "HISTORICAL_UNLINKED_OFFICIAL_REPLAY_PENDING", "raw_clean_sota_claim": False}
    baseline["structrfm_official_recovery"] = "manifests/foldnar_v2/structrfm_official_recovery_20261126.json"
    write_json(baseline_path, baseline_doc)

    prediction_path = repo / "manifests/foldnar_v1/prediction_registry.json"
    prediction_doc = json.loads(prediction_path.read_text())
    prediction = next(row for row in prediction_doc["rows"] if row["prediction_id"] == "structrfm/bprna_new")
    prediction["status"] = "BLOCKED"
    prediction["e0_scope"] = "HISTORICAL_UNLINKED_DO_NOT_USE_AS_VERIFIED_OFFICIAL"
    prediction["training_identity"] = "Historical log names structRFM_SSP_bpRNA1m.pt, but historical weight bytes are absent; official v0.0.8 checkpoint recovered separately"
    prediction["official_checkpoint_link"] = {"status": "UNVERIFIED_HISTORICAL_WEIGHT_BYTES", "historical_weight_path": str(HISTORICAL_WEIGHT), "historical_weight_sha256": None, "new_official_checkpoint_sha256": ASSET_SHA256, "eligible_as_verified_official": False}
    prediction["comparability"] = "Historical project replay retained for trace only; blocked from verified-official comparisons; raw is not clean"
    write_json(prediction_path, prediction_doc)

    recovered_path = repo / "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
    recovered_doc = json.loads(recovered_path.read_text())
    recovered = next(row for row in recovered_doc["rows"] if row["asset_id"] == "structrfm_author_bprna1m_single_raw")
    recovered["status"] = "HISTORICAL_UNLINKED_OFFICIAL_WEIGHT_BYTES_UNAVAILABLE"
    recovered["training_identity"] = prediction["training_identity"]
    recovered["sota_use"] = "historical unlinked project replay only; not a verified official v0.0.8 result and not clean SOTA"
    recovered["official_checkpoint_link"] = prediction["official_checkpoint_link"]
    recovered_doc["remaining_blocked"] = ["structRFM historical SSP weight is absent; old 5388 predictions are trace-only, while the recovered official checkpoint still needs a new GPU replay" if "structRFM" in row else row for row in recovered_doc["remaining_blocked"]]
    write_json(recovered_path, recovered_doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_structrfm_recovery"))
    parser.add_argument("--skip-checkpoint-load", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.dataset, args.source, not args.skip_checkpoint_load)
    write_json(args.root / "reports/structrfm_official_provenance_20261126.json", manifest)
    write_json(args.repo / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json", manifest)
    update_registries(args.repo, manifest)
    print(json.dumps({"status": manifest["status"], "full_ready": False, "checkpoint": manifest["checkpoint"], "historical_weight": manifest["historical_artifacts"]["weight"]}, indent=2))


if __name__ == "__main__":
    main()
