#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DEFAULT_DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
DEFAULT_OFFICIAL = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/ernierna_official_recovery_20261126/ERNIE-RNA")
DEFAULT_PROVENANCE = REPO / "manifests/foldnar_v1/ernierna_official_asset_provenance_20261126.json"
DEFAULT_OUTPUT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/ernierna_official_recovery_20261126/bprna_new_batch")
OFFICIAL_COMMIT = "43bc06de1088ed03ffd7de918ad4b2c2a3346a43"
FINETUNE = "finetune"
ZEROSHOT = "zeroshot"
SUFFIXES = {FINETUNE: "_finetune_prediction.ct", ZEROSHOT: "_zeroshot_prediction.ct"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(temporary, path)


def read_provenance(path):
    document = json.loads(Path(path).read_text())
    assert document["official_source"]["commit"] == OFFICIAL_COMMIT
    return document


def audit_assets(official_root, provenance):
    official_root = Path(official_root)
    assets = {}
    for relative, expected in provenance["official_source"]["required_file_sha256"].items():
        path = official_root / relative
        actual = sha256(path) if path.is_file() else None
        assets[relative] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "ready": actual == expected}
    for item in provenance["weights"]:
        path = official_root.parent / item["local_relative_path"]
        actual = sha256(path) if path.is_file() else None
        expected = item.get("sha256")
        assets[item["asset_id"]] = {
            "path": str(path), "expected_sha256": expected, "actual_sha256": actual,
            "ready": bool(expected) and actual == expected,
            "official_url": item["official_url"],
        }
    source_text = (official_root / "predict_ss_rna.py").read_text()
    mapping_ok = "'bpRNA-new': 'ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt'" in source_text
    utils_text = (official_root / "src/utils.py").read_text()
    single_ok = "rna_models[0]" in utils_text and "mlm_pretrained_model_path.split(os.pathsep)" in utils_text
    return {"ready": all(row["ready"] for row in assets.values()) and mapping_ok and single_ok,
            "assets": assets, "bprna_new_checkpoint_mapping_verified": mapping_ok,
            "single_checkpoint_load_verified": single_ok}


def read_dataset(path, expected_sha256):
    path = Path(path)
    actual = sha256(path)
    assert actual == expected_sha256, (actual, expected_sha256)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names)), "duplicate dataset names"
    records = []
    for index, row in enumerate(rows):
        name = row["name"]
        assert name and not name.isspace()
        original = row["seq"]
        sequence = original.upper().replace("T", "U")
        assert sequence and re.fullmatch(r"[A-Z]+", sequence), (name, sequence)
        records.append({
            "index": index, "input_id": f"bprna_new_{index:06d}", "name": name,
            "length": len(sequence), "sequence": sequence,
            "source_sequence_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "normalized_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        })
    return records, actual


def public_record(record):
    return {key: value for key, value in record.items() if key != "sequence"}


def fasta_text(records):
    return "".join(f">{row['input_id']}\n{row['sequence']}\n" for row in records)


def prepare(args):
    provenance = read_provenance(args.provenance)
    records, data_sha = read_dataset(args.data, args.data_sha256)
    asset_audit = audit_assets(args.official_root, provenance)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fasta_path = args.output_root / "bprna_new.official_input.fasta"
    fasta_path.write_text(fasta_text(records))
    contract = {
        "schema_version": 1,
        "contract_id": "ernierna_author_frozen_bprna1m/bprna_new/raw_v1",
        "dataset": {"path": str(args.data), "sha256": data_sha, "n_sequences": len(records),
                    "label_use": "IDs and sequences only for this batch contract; no tuning or external Test access"},
        "official_source": provenance["official_source"],
        "model_identity": provenance["model_identity"],
        "input_fasta": {"path": str(fasta_path), "sha256": sha256(fasta_path)},
        "records": [public_record(row) for row in records],
        "asset_audit": asset_audit,
        "eligible_to_launch": asset_audit["ready"],
        "execution_status": "NOT_RUN_NO_INFERENCE_AUTHORIZED",
        "future_command_not_executed": (
            f"cd {args.official_root} && python predict_ss_rna.py --dataset_name bpRNA-new "
            f"--seqs_path {fasta_path} --save_path {args.output_root / 'official_ct'} --device 0"
        ),
        "output_contract": {
            "directory": str(args.output_root / "official_ct"),
            "required_ct_per_sequence": 2,
            "finetune_suffix": SUFFIXES[FINETUNE], "zeroshot_suffix": SUFFIXES[ZEROSHOT],
            "target_prediction": FINETUNE,
            "pair_convention": "zero-based inclusive sorted [i,j], i<j; no crossing or short-span filtering",
            "summary_or_demo_substitution_allowed": False,
        },
    }
    write_json(args.output_root / "batch_contract.json", contract)
    print(json.dumps({"contract": str(args.output_root / "batch_contract.json"), "n_sequences": len(records),
                      "eligible_to_launch": asset_audit["ready"], "inference_executed": False}, indent=2))


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    assert contract["schema_version"] == 1
    assert contract["output_contract"]["summary_or_demo_substitution_allowed"] is False
    names = [row["name"] for row in contract["records"]]
    ids = [row["input_id"] for row in contract["records"]]
    assert len(names) == len(set(names)) and len(ids) == len(set(ids))
    assert ids == [f"bprna_new_{index:06d}" for index in range(len(ids))]
    fasta = Path(contract["input_fasta"]["path"])
    assert sha256(fasta) == contract["input_fasta"]["sha256"]
    by_id = {row["input_id"]: row for row in contract["records"]}
    current_id = None
    sequences = {}
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            current_id = line[1:]
            assert current_id in by_id and current_id not in sequences
            sequences[current_id] = ""
        else:
            assert current_id is not None
            sequences[current_id] += line
    assert list(sequences) == ids
    for row in contract["records"]:
        sequence = sequences[row["input_id"]]
        assert len(sequence) == row["length"]
        assert hashlib.sha256(sequence.encode()).hexdigest() == row["normalized_sequence_sha256"]
    return contract, sequences


def parse_ct(path, expected_title, sequence):
    path = Path(path)
    lines = path.read_text().splitlines()
    assert len(lines) == len(sequence) + 1, (path, "line count", len(lines), len(sequence) + 1)
    header = lines[0].split(None, 1)
    assert len(header) == 2 and int(header[0]) == len(sequence), (path, "header length")
    assert header[1].strip() == expected_title, (path, "header ID", header[1], expected_title)
    partners = {}
    for offset, line in enumerate(lines[1:]):
        fields = line.split()
        assert len(fields) == 6, (path, offset + 2, "CT columns")
        index, base, previous, following, partner, natural = fields
        position = offset + 1
        assert int(index) == position and int(natural) == position
        assert base == sequence[offset], (path, position, "sequence mismatch", base, sequence[offset])
        assert int(previous) == (position - 1 if position > 1 else 0)
        assert int(following) == (position + 1 if position < len(sequence) else 0)
        partner = int(partner)
        assert 0 <= partner <= len(sequence) and partner != position
        partners[position] = partner
    for position, partner in partners.items():
        if partner:
            assert partners[partner] == position, (path, "asymmetric pair", position, partner)
    return sorted((left - 1, right - 1) for left, right in partners.items() if right and left < right)


def export_predictions(args):
    contract, sequences = load_contract(args.contract)
    provenance = read_provenance(args.provenance)
    assets = audit_assets(args.official_root, provenance)
    assert assets["ready"], "official code/weight identity is not fully verified"
    ct_root = Path(contract["output_contract"]["directory"])
    expected_paths = {row["input_id"] + suffix for row in contract["records"] for suffix in SUFFIXES.values()}
    actual_paths = {path.name for path in ct_root.glob("*.ct")} if ct_root.is_dir() else set()
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    assert not missing and not extra, {"missing_ct": missing[:20], "extra_ct": extra[:20],
                                      "missing_count": len(missing), "extra_count": len(extra)}
    output_rows = {FINETUNE: [], ZEROSHOT: []}
    source_ct = []
    for row in contract["records"]:
        for kind, suffix in SUFFIXES.items():
            title = row["input_id"] + suffix[:-3]
            path = ct_root / (title + ".ct")
            pairs = parse_ct(path, title, sequences[row["input_id"]])
            output_rows[kind].append({"name": row["name"], "seq": sequences[row["input_id"]], "pairs": pairs})
            source_ct.append({"name": row["name"], "kind": kind, "path": str(path), "sha256": sha256(path),
                              "n_pairs": len(pairs)})
    paths = {kind: args.output_root / f"{kind}_predictions.jsonl" for kind in SUFFIXES}
    for kind, path in paths.items():
        write_jsonl(path, output_rows[kind])
    n = len(contract["records"])
    manifest = {
        "schema_version": 1, "status": "PASS_FULL_BPRNA_NEW_PREDICTIONS",
        "contract_id": contract["contract_id"], "dataset_sha256": contract["dataset"]["sha256"],
        "n_dataset_sequences": n, "n_finetune_predictions": n, "n_zeroshot_predictions": n,
        "coverage": 1.0 if n else None, "missing_names": [], "extra_names": [],
        "prediction_identity": {
            kind: {"path": str(paths[kind]), "sha256": sha256(paths[kind]),
                   "content_sha256": hashlib.sha256(canonical([[row["name"], row["pairs"]] for row in rows]).encode()).hexdigest()}
            for kind, rows in output_rows.items()
        },
        "target_prediction": FINETUNE, "model_identity": provenance["model_identity"],
        "official_weight_sha256": {row["asset_id"]: row["sha256"] for row in provenance["weights"]},
        "source_ct": source_ct,
    }
    write_json(args.output_root / "export_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "coverage": 1.0, "n_sequences": n,
                      "target_prediction": FINETUNE, "inference_executed_by_this_tool": False}, indent=2))


def audit(args):
    contract, _ = load_contract(args.contract)
    provenance = read_provenance(args.provenance)
    assets = audit_assets(args.official_root, provenance)
    export_path = args.output_root / "export_manifest.json"
    full = False
    if export_path.is_file():
        export = json.loads(export_path.read_text())
        n = len(contract["records"])
        expected_names = [row["name"] for row in contract["records"]]
        assert export["status"] == "PASS_FULL_BPRNA_NEW_PREDICTIONS" and export["coverage"] == 1.0
        for kind in SUFFIXES:
            identity = export["prediction_identity"][kind]
            path = Path(identity["path"])
            assert sha256(path) == identity["sha256"]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == n and [row["name"] for row in rows] == expected_names
            content = hashlib.sha256(canonical([[row["name"], row["pairs"]] for row in rows]).encode()).hexdigest()
            assert content == identity["content_sha256"]
        full = True
    result = {
        "status": "FULL_PREDICTIONS_READY" if full else ("ASSETS_READY_PREDICTIONS_PENDING" if assets["ready"] else "BLOCKED_OFFICIAL_ASSETS"),
        "n_dataset_sequences": len(contract["records"]), "asset_ready": assets["ready"],
        "full_prediction_ready": full, "asset_audit": assets,
        "inference_executed_by_this_tool": False,
    }
    write_json(args.output_root / "readiness_audit.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "export", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
        sub.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
        if command == "prepare":
            sub.add_argument("--data", type=Path, default=DEFAULT_DATA)
            sub.add_argument("--data-sha256", default=DEFAULT_DATA_SHA256)
        else:
            sub.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT / "batch_contract.json")
        sub.set_defaults(function={"prepare": prepare, "export": export_predictions, "audit": audit}[command])
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
