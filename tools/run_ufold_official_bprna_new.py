#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ASSET_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/ufold-official-recovery-20261127")
OFFICIAL_SOURCE = ASSET_ROOT / "source/UFold-75bd9acc83826059682dfca9d3659df66b132cd1"
OFFICIAL_URL = "https://github.com/uci-cbcl/UFold"
OFFICIAL_COMMIT = "75bd9acc83826059682dfca9d3659df66b132cd1"
SOURCE_ARCHIVE_SHA256 = "a233d3577cbb213712c06f8f0827b18624e7df99ad093c6bfd70782b8e86b76b"
LICENSE_SHA256 = "5c93b3a2d31b6e7b964633ff5e81bd9e31ff9ee845996ab3b0d3d71b0d368f2e"
CONFIG_SHA256 = "1cfd3d7310f549b43e7defe890f19e69b09b001caefd61bab0ce2f9a9b3fa88d"
AUTHOR_WEIGHTS_URL = "https://drive.google.com/drive/folders/1Sq7MVgFOshGPlumRE_hpNXadvhJKaryi?usp=sharing"
EXPECTED_CHECKPOINT = "ufold_train_alldata.pt"
EXPECTED_CHECKPOINT_SHA256 = None
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
N_SEQUENCES = 5388
MODEL_INPUT_LENGTH = 600
MODEL_PARAMETER_COUNT = 8641377
MODEL_STATE_DICT_BYTES = 34593076
CANONICAL = {"AU", "UA", "CG", "GC", "GU", "UG"}
DEFAULT_TRAIN_FILES = [
    "RNAStralign", "ArchiveII", "TR0_with_data_augmentation", "TS0",
    "PDB_train", "bpnew", "TS1", "TS2", "TS3",
]


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pair_set(values, length, sequence=None):
    pairs = set()
    partners = {}
    for value in values:
        require(len(value) == 2, f"Invalid pair: {value}")
        i, j = map(int, value)
        require(0 <= i < j < length, f"Out-of-range or reversed pair: {(i, j, length)}")
        require(j - i > 3, f"Minimum-loop violation: {(i, j)}")
        if sequence is not None:
            require(sequence[i] + sequence[j] in CANONICAL,
                    f"Non-canonical pair: {(i, j, sequence[i], sequence[j])}")
        require(i not in partners and j not in partners, f"Partner conflict: {(i, j)}")
        partners[i], partners[j] = j, i
        pairs.add((i, j))
    return pairs


def load_bprna_new(path=DATA):
    path = Path(path)
    require(path.is_file(), f"Missing dataset: {path}")
    require(sha256(path) == DATA_SHA256, "bpRNA-new SHA256 does not match the frozen input contract")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == N_SEQUENCES, f"Expected {N_SEQUENCES} rows, found {len(rows)}")
    require(all(count == 1 for count in Counter(row["name"] for row in rows).values()),
            "Sequence names must be unique")
    for row in rows:
        sequence = row["seq"]
        require(0 < len(sequence) <= MODEL_INPUT_LENGTH,
                f"Official short-model length violation: {row['name']}")
        require(set(sequence) <= set("ACGU"), f"Non-ACGU sequence: {row['name']}")
        pair_set(row["pairs"], len(sequence), sequence)
    return rows


def validate_official_source(source=OFFICIAL_SOURCE):
    source = Path(source).resolve()
    require(source.name == f"UFold-{OFFICIAL_COMMIT}",
            "Official source directory must carry the full commit")
    require(sha256(source / "LICENSE") == LICENSE_SHA256, "Official MIT license SHA256 mismatch")
    require(sha256(source / "ufold/config.json") == CONFIG_SHA256,
            "Official config SHA256 mismatch")
    archive = source.parent / f"UFold-{OFFICIAL_COMMIT}.tar.gz"
    require(archive.is_file(), f"Official source archive is unavailable: {archive}")
    require(sha256(archive) == SOURCE_ARCHIVE_SHA256, "Official source archive SHA256 mismatch")
    return source


def load_official_modules(source=OFFICIAL_SOURCE):
    source = validate_official_source(source)
    loaded = sys.modules.get("ufold")
    if loaded is not None and hasattr(loaded, "__file__") and loaded.__file__ is not None:
        loaded_path = Path(loaded.__file__).resolve()
        require(source in loaded_path.parents, f"ufold already loaded from another source: {loaded_path}")
    sys.path.insert(0, str(source))
    network = importlib.import_module("Network")
    data_generator = importlib.import_module("ufold.data_generator")
    postprocess = importlib.import_module("ufold.postprocess")
    utils = importlib.import_module("ufold.utils")
    for module in (data_generator, postprocess, utils):
        require(source in Path(module.__file__).resolve().parents,
                f"Official import escaped the source: {module.__file__}")
    require(Path(network.__file__).resolve() == source / "Network.py",
            "Network import escaped the official source")
    return network, data_generator, postprocess, utils


def pairs_from_partner_indices(partner_indices, sequence):
    partners = {}
    for i, value in enumerate(partner_indices):
        partner = int(value) - 1
        if partner >= 0:
            require(partner < len(sequence), f"CT partner out of range at {i}: {partner}")
            require(partner != i, f"CT self-pair at {i}")
            partners[i] = partner
    pairs = []
    for i, j in partners.items():
        require(partners.get(j) == i, f"CT pair is not mutual: {(i, j)}")
        if i < j:
            pairs.append([i, j])
    return sorted(pair_set(pairs, len(sequence), sequence))


def validate_checkpoint_identity(checkpoint, expected_sha256=EXPECTED_CHECKPOINT_SHA256):
    checkpoint = Path(checkpoint)
    require(checkpoint.name == EXPECTED_CHECKPOINT,
            f"Offline all-data checkpoint must be named {EXPECTED_CHECKPOINT}")
    require(checkpoint.is_file(), f"Author checkpoint is unavailable: {checkpoint}")
    require(expected_sha256 is not None,
            "Author checkpoint SHA256 is not certified; refusing to select or infer from an unauthenticated weight")
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(char in "0123456789abcdef" for char in expected_sha256),
            "Checkpoint SHA256 must be 64 lowercase hexadecimal characters")
    actual = sha256(checkpoint)
    require(actual == expected_sha256, f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def build_official_input(sequence, data_generator, numpy):
    length = len(sequence)
    bases = "AUCG"
    data_seq = numpy.zeros((MODEL_INPUT_LENGTH, 4))
    data_seq[numpy.arange(length), [bases.index(base) for base in sequence]] = 1
    padded_length = data_generator.get_cut_len(length, 80)
    pair_features = numpy.zeros((16, padded_length, padded_length))
    if padded_length >= 500:
        adjusted = numpy.zeros((padded_length, 4))
        adjusted[:length] = data_seq[:length]
        data_seq = adjusted
    for channel, (i, j) in enumerate(data_generator.perm):
        pair_features[channel, :length, :length] = numpy.matmul(
            data_seq[:length, i].reshape(-1, 1), data_seq[:length, j].reshape(1, -1))
    coupling = numpy.zeros((1, padded_length, padded_length))
    coupling[0, :length, :length] = data_generator.creatmat(data_seq[:length])
    features = numpy.concatenate((pair_features, coupling), axis=0)
    require(features.shape == (17, padded_length, padded_length), "Official feature shape mismatch")
    return features, data_seq[:padded_length]


def load_single_model(network, checkpoint, checkpoint_sha256, device, torch):
    actual = validate_checkpoint_identity(checkpoint, checkpoint_sha256)
    model = network.U_Net(img_ch=17)
    require(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETER_COUNT,
            "Official U_Net parameter count changed")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    require(isinstance(state, dict), "Checkpoint must contain a state dictionary")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, actual


def training_exposure():
    return {
        "scope": "bpRNA-new is external Test for protocol audit only, never model selection",
        "official_default_train_files": DEFAULT_TRAIN_FILES,
        "bpnew_in_official_default_train_files": True,
        "selected_offline_checkpoint": EXPECTED_CHECKPOINT,
        "official_offline_entry": "ufold_predict.py",
        "assessment": "DIRECT_BPNEW_EXPOSURE_DECLARED_BY_OFFICIAL_ALL_DATA_TRAINING_DEFAULT",
        "per_weight_training_ledger": None,
        "caveat": "The official repository provides no per-weight training manifest; the all-data label and default training list are recorded, not promoted to an independently verified lineage.",
    }


def contract_report(rows):
    lengths = [len(row["seq"]) for row in rows]
    return {
        "status": "CONTRACT_PASS_NO_INFERENCE",
        "official_url": OFFICIAL_URL,
        "official_commit": OFFICIAL_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": {"spdx": "MIT", "sha256": LICENSE_SHA256},
        "author_weights_url": AUTHOR_WEIGHTS_URL,
        "expected_checkpoint": EXPECTED_CHECKPOINT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dataset": {
            "path": str(DATA),
            "sha256": DATA_SHA256,
            "n_sequences": len(rows),
            "unique_names": len({row["name"] for row in rows}),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "n_reference_pairs": sum(len(row["pairs"]) for row in rows),
            "pair_convention": "zero-based inclusive i<j; canonical AU/CG/GU; j-i>3; one partner per base",
        },
        "model": {
            "identity": "UFold official offline all-data U_Net",
            "input_channels": 17,
            "input_storage_length": MODEL_INPUT_LENGTH,
            "batch_size": 1,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "state_dict_bytes_fp32": MODEL_STATE_DICT_BYTES,
            "postprocess": "postprocess_new(pred, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5); threshold >0.5",
            "noncanonical": False,
            "ensemble": False,
        },
        "training_exposure": training_exposure(),
        "predictions": None,
    }


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_inference(args):
    import numpy as np
    import torch

    require(args.device == "cuda",
            "The production replay is intentionally CUDA-only; use --contract-only for CPU checks")
    require(torch.cuda.is_available(), "CUDA is not available")
    rows = load_bprna_new(args.data)
    network, data_generator, postprocess, utils = load_official_modules(args.official_source)
    device = torch.device("cuda")
    model, checkpoint_actual = load_single_model(
        network, args.checkpoint, args.checkpoint_sha256, device, torch)
    utils.seed_torch(0)
    output = args.output
    require(not output.exists(), f"Refusing to overwrite existing predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    require(not partial.exists(), f"Refusing to overwrite partial predictions: {partial}")
    with partial.open("w") as handle:
        for row in rows:
            sequence = row["seq"]
            features, seq_ori = build_official_input(sequence, data_generator, np)
            seq_batch = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            seq_ori_batch = torch.tensor(seq_ori, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pred_contacts = model(seq_batch)
                postprocessed = postprocess.postprocess_new(
                    pred_contacts, seq_ori_batch, 0.01, 0.1, 100, 1.6, True, 1.5)
                binary = (postprocessed > 0.5).float()
            ct = utils.contact2ct(binary[0].cpu().numpy(), seq_ori, len(sequence))
            pairs = pairs_from_partner_indices(ct["pair_index"].tolist(), sequence)
            handle.write(canonical_json({"name": row["name"], "pairs": pairs}) + "\n")
            handle.flush()
    require(sum(1 for line in partial.open() if line.strip()) == N_SEQUENCES,
            "Prediction row count differs from the frozen input")
    partial.replace(output)
    manifest = {
        "status": "COMPLETE_RAW_PREDICTIONS_PROTOCOL_AUDIT_ONLY",
        "prediction_class": "PROJECT_REPLAY_WITH_OFFICIAL_SOURCE_AND_AUTHOR_LINKED_WEIGHTS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_url": OFFICIAL_URL,
        "official_commit": OFFICIAL_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": {"spdx": "MIT", "sha256": LICENSE_SHA256},
        "author_weights_url": AUTHOR_WEIGHTS_URL,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_actual,
        "dataset_sha256": DATA_SHA256,
        "n_sequences": N_SEQUENCES,
        "output": str(output),
        "output_sha256": sha256(output),
        "seed": 0,
        "training_exposure": training_exposure(),
        "model_selection_permitted": False,
    }
    write_json(args.manifest, manifest)


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen UFold official bpRNA-new5388 replay contract")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--official-source", type=Path, default=OFFICIAL_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=ASSET_ROOT / "weights" / EXPECTED_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=ASSET_ROOT / "predictions/bprna_new5388.jsonl")
    parser.add_argument("--manifest", type=Path, default=ASSET_ROOT / "predictions/bprna_new5388.manifest.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.contract_only:
        rows = load_bprna_new(args.data)
        validate_official_source(args.official_source)
        print(json.dumps(contract_report(rows), indent=2, ensure_ascii=False))
        return
    run_inference(args)


if __name__ == "__main__":
    main()
