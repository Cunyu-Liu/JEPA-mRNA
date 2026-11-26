#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ASSET_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/e2efold-official-recovery-20261126")
OFFICIAL_SOURCE = ASSET_ROOT / "source/e2efold-2aab9e78326ddaa27d14c49eb34001c0003a1960"
OFFICIAL_URL = "https://github.com/ml4bio/e2efold"
OFFICIAL_COMMIT = "2aab9e78326ddaa27d14c49eb34001c0003a1960"
SOURCE_ARCHIVE_SHA256 = "4a599a1a74eae1766de9782ca3eb7ca8017ac322b996cadba0188b9594ab2803"
LICENSE_SHA256 = "a10fe3436b8aaf66478bdfd46c45fdfe39674a0db2a0a9aa1b37cf443a53cd12"
CONFIG_SHA256 = "10f00718840eb53efaf9b89cb2d78364d654c7d05c60b5d50e1006ffaf3b291a"
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
N_SEQUENCES = 5388
MODEL_LENGTH = 600
EXPECTED_CHECKPOINT = "e2e_att_simple_fix_mixed_s20_d10_rnastralign_all_600_f1_position_matrix.pt"
CANONICAL = {"AU", "UA", "CG", "GC", "GU", "UG"}


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
        require(0 < len(sequence) <= MODEL_LENGTH, f"Short-model length violation: {row['name']}")
        require(set(sequence) <= set("ACGU"), f"Non-ACGU sequence: {row['name']}")
        pair_set(row["pairs"], len(sequence), sequence)
    return rows


def load_model_spec(source=OFFICIAL_SOURCE):
    source = Path(source)
    config_path = source / "e2efold_productive/config.json"
    require(sha256(config_path) == CONFIG_SHA256, "Official short-model config SHA256 mismatch")
    config = json.loads(config_path.read_text())
    expected = {
        "u_net_d": 10,
        "pp_steps": 20,
        "pp_loss": "f1",
        "pp_model": "mixed",
        "rho_per_position": "matrix",
        "data_type": "rnastralign_all_600",
        "model_type": "att_simple_fix",
        "k": 1,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"Official config changed at {key}: {config.get(key)}")
    checkpoint = "e2e_{}_{}_s{}_d{}_{}_{}_position_{}.pt".format(
        config["model_type"], config["pp_model"], config["pp_steps"], config["u_net_d"],
        config["data_type"], config["pp_loss"], config["rho_per_position"])
    require(checkpoint == EXPECTED_CHECKPOINT, "Unexpected official checkpoint identity")
    return config


def load_official_modules(source=OFFICIAL_SOURCE):
    source = Path(source).resolve()
    require(source.name == f"e2efold-{OFFICIAL_COMMIT}",
            "Official source directory must carry the full commit")
    require(sha256(source / "LICENSE") == LICENSE_SHA256, "Official MIT license SHA256 mismatch")
    load_model_spec(source)
    loaded = sys.modules.get("e2efold")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        require(source in loaded_path.parents, f"e2efold already loaded from another source: {loaded_path}")
    else:
        sys.path.insert(0, str(source))
        importlib.import_module("e2efold")
    models = importlib.import_module("e2efold.models")
    utils = importlib.import_module("e2efold.common.utils")
    require(source in Path(models.__file__).resolve().parents, "Model import escaped the official source")
    require(source in Path(utils.__file__).resolve().parents, "Utility import escaped the official source")
    return models, utils


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


def ct_to_pairs(ct, sequence):
    require(len(ct) == len(sequence), "CT length does not match sequence length")
    return pairs_from_partner_indices(ct["pair_index"].tolist(), sequence)


def validate_checkpoint_identity(checkpoint, expected_sha256):
    checkpoint = Path(checkpoint)
    require(checkpoint.name == EXPECTED_CHECKPOINT,
            f"Single-model checkpoint must be named {EXPECTED_CHECKPOINT}")
    require(checkpoint.is_file(), f"Official checkpoint is unavailable: {checkpoint}")
    require(expected_sha256 is not None, "--checkpoint-sha256 is required; unknown hashes are not accepted")
    actual = sha256(checkpoint)
    require(actual == expected_sha256, f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def build_single_model(models, device):
    contact_net = models.ContactAttention_simple_fix_PE(d=10, L=MODEL_LENGTH, device=device).to(device)
    lag_pp_net = models.Lag_PP_mixed(20, 1, "matrix").to(device)
    model = models.RNA_SS_e2e(contact_net, lag_pp_net).to(device)
    return model


def load_single_model(models, checkpoint, checkpoint_sha256, device, torch):
    validate_checkpoint_identity(checkpoint, checkpoint_sha256)
    model = build_single_model(models, device)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def contract_report(rows, source=OFFICIAL_SOURCE):
    lengths = [len(row["seq"]) for row in rows]
    return {
        "status": "CONTRACT_PASS_NO_INFERENCE",
        "official_url": OFFICIAL_URL,
        "official_commit": OFFICIAL_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": {"spdx": "MIT", "sha256": LICENSE_SHA256},
        "expected_checkpoint": EXPECTED_CHECKPOINT,
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
            "identity": "E2Efold official short RNAStrAlign end-to-end model",
            "model_type": "ContactAttention_simple_fix_PE + Lag_PP_mixed",
            "model_length": MODEL_LENGTH,
            "d": 10,
            "pp_steps": 20,
            "k": 1,
            "rho_per_position": "matrix",
            "threshold": 0.5,
            "batch_size": 1,
            "ensemble": False,
        },
        "predictions": None,
    }


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_inference(args):
    import torch

    rows = load_bprna_new(args.data)
    models, utils = load_official_modules(args.official_source)
    require(args.device == "cuda", "The production replay is intentionally CUDA-only; use --contract-only for CPU checks")
    require(torch.cuda.is_available(), "CUDA is not available")
    device = torch.device("cuda")
    model = load_single_model(models, args.checkpoint, args.checkpoint_sha256, device, torch)
    utils.seed_torch(0)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w") as handle:
        for row in rows:
            sequence = row["seq"]
            encoded = utils.padding(utils.seq_encoding(sequence), MODEL_LENGTH)
            seq_batch = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
            seq_lens = torch.tensor([len(sequence)], dtype=torch.int32)
            pe_batch = utils.get_pe(seq_lens, MODEL_LENGTH).float().to(device)
            state_pad = torch.zeros(1, 2, 2, device=device)
            with torch.no_grad():
                pred_contacts = model.model_att(pe_batch, seq_batch, state_pad)
                a_pred_list = model.model_pp(pred_contacts, seq_batch)
            final_pred = (a_pred_list[-1].cpu() > 0.5).float()
            ct = utils.contact2ct(final_pred[0].numpy(), encoded, len(sequence))
            pairs = ct_to_pairs(ct, sequence)
            handle.write(canonical_json({"name": row["name"], "pairs": pairs}) + "\n")
            handle.flush()
    require(sum(1 for line in partial.open() if line.strip()) == N_SEQUENCES,
            "Prediction row count differs from the frozen input")
    partial.replace(output)
    manifest = {
        "status": "COMPLETE_RAW_PREDICTIONS",
        "prediction_class": "PROJECT_REPLAY_WITH_OFFICIAL_SOURCE_AND_AUTHOR_WEIGHTS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_url": OFFICIAL_URL,
        "official_commit": OFFICIAL_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": {"spdx": "MIT", "sha256": LICENSE_SHA256},
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "dataset_sha256": DATA_SHA256,
        "n_sequences": N_SEQUENCES,
        "output": str(output),
        "output_sha256": sha256(output),
        "seed": 0,
        "device": "cuda",
        "batch_size": 1,
        "model_length": MODEL_LENGTH,
        "threshold": 0.5,
        "ensemble": False,
        "label_use": "Reference pairs are contract-validated but never passed to the model or decoder",
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Strict single-checkpoint official E2Efold bpRNA-new replay")
    parser.add_argument("--official-source", type=Path, default=OFFICIAL_SOURCE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--checkpoint", type=Path,
                        default=ASSET_ROOT / "weights" / EXPECTED_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path,
                        default=ASSET_ROOT / "predictions/e2efold_bprna_new.jsonl")
    parser.add_argument("--manifest", type=Path,
                        default=ASSET_ROOT / "reports/e2efold_bprna_new_run_manifest.json")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--contract-only", action="store_true",
                        help="CPU-only source/data/config checks; never loads weights or runs inference")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_bprna_new(args.data)
    load_official_modules(args.official_source)
    if args.contract_only:
        print(json.dumps(contract_report(rows, args.official_source), indent=2))
        return
    run_inference(args)


if __name__ == "__main__":
    main()
