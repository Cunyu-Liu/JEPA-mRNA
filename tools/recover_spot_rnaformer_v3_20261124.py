#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
N_SEQUENCES = 5388
BASE = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
ARMS = {
    "spot_rna_registry_raw": {
        "path": BASE / "eval_decision/round39_clean_benchmark/spotrna/raw/bprna_new.json",
        "sha256": "613ba7a4c8ceae0a8197657a6b96e8a01c3c54cd923cfa81974b9ddbb16af6cc",
        "schema": "spotrna",
        "role": "PRIMARY_REGISTRY_FROZEN",
    },
    "spot_rna_postprocessed": {
        "path": BASE / "eval_decision/round39_clean_benchmark/spotrna/raw_postprocessed/bprna_new.json",
        "sha256": "c76f473574fa0032213f49bc8e03e70406bf7cb2720d4a78a57d149b71ff5ec8",
        "schema": "spotrna",
        "role": "SEPARATE_POSTPROCESSED_PROTOCOL",
    },
    "rnaformer_bprna": {
        "path": BASE / "eval_decision/round39_clean_benchmark/rnaformer/raw/bprna_new.json",
        "sha256": "1559e6c4fbfcbb5e36debfe9c41b9ed0af6575a01100beac4b2c07780397c856",
        "schema": "rnaformer",
        "role": "PRIMARY_REGISTRY_FROZEN",
    },
}
ASSETS = {
    "spot_rna_model_safetensors": (
        BASE / "weights/round39/multimolecule/spotrna/model.safetensors",
        "b7056fcad23a00d5bbaa21f91b54da779e87fa5e2185fb3ccc19c5440cadf0c5",
    ),
    "spot_rna_config": (
        BASE / "weights/round39/multimolecule/spotrna/config.json",
        "c260db054bad4f3954e736ab31c74a5270ab045302e6c5e2ca9e8f225f2b45f9",
    ),
    "rnaformer_bprna_state_dict": (
        BASE / "weights/baselines/rnaformer/RNAformer_32M_state_dict_bprna.pth",
        "b2775613e9addafe30d17f7152dc985e8381826842d5d5056030006adfad6769",
    ),
    "rnaformer_bprna_config": (
        BASE / "weights/baselines/rnaformer/RNAformer_32M_config_bprna.yml",
        "2af84e2caed3a784c5e45101be0a04c37c94f4605d3d3e3ed82a18c4d2948054",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def pair_set(values, length=None):
    pairs = set()
    partners = {}
    for value in values:
        assert len(value) == 2, value
        i, j = map(int, value)
        assert 0 <= i < j, value
        if length is not None:
            assert j < length, (value, length)
        assert i not in partners and j not in partners, value
        partners[i], partners[j] = j, i
        pairs.add((i, j))
    return pairs


def strict_metrics(rows, prediction_by_name):
    tp = fp = fn = 0
    per_sequence = []
    for row in rows:
        name = row["name"]
        assert name in prediction_by_name, f"MISSING_PREDICTION: {name}"
        truth = pair_set(row["pairs"], len(row["seq"]))
        predicted = pair_set(prediction_by_name[name]["pairs"], len(row["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        denominator = 2 * a + b + c
        f1 = 2 * a / denominator if denominator else 1.0
        tp, fp, fn = tp + a, fp + b, fn + c
        per_sequence.append({
            "name": name,
            "length": len(row["seq"]),
            "n_truth_pairs": len(truth),
            "n_predicted_pairs": len(predicted),
            "tp": a,
            "fp": b,
            "fn": c,
            "precision": a / (a + b) if a + b else None,
            "recall": a / (a + c) if a + c else None,
            "f1": f1,
        })
    n = len(rows)
    denominator = 2 * tp + fp + fn
    return {
        "n_dataset_sequences": n,
        "n_with_prediction": n,
        "coverage": 1.0 if n else None,
        "missing_names": [],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / denominator if denominator else None,
        "macro_f1": math.fsum(row["f1"] for row in per_sequence) / n if n else None,
        "n_zero_truth_sequences": sum(row["n_truth_pairs"] == 0 for row in per_sequence),
        "n_empty_predictions": sum(row["n_predicted_pairs"] == 0 for row in per_sequence),
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }, per_sequence


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(rows) == N_SEQUENCES
    assert len(Counter(row["name"] for row in rows)) == N_SEQUENCES
    for row in rows:
        pair_set(row["pairs"], len(row["seq"]))
    return rows


def extract_predictions(document, schema):
    if schema == "spotrna":
        source = document["per_sequence"]
        return {row["name"]: {"pairs": row["pred_pairs"]} for row in source}, source
    source = document["predictions"]
    return {row["name"]: {"pairs": row["pairs"]} for row in source}, source


def verify_recorded(document, schema, metrics, per_sequence, identity_hash):
    if schema == "spotrna":
        for actual, recorded in zip(per_sequence, document["per_sequence"]):
            assert actual["name"] == recorded["name"]
            assert (actual["tp"], actual["fp"], actual["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"])
            assert_close(actual["f1"], recorded["f1"])
        repeats = document["repeats"]
        assert len(repeats) == 3
        for repeat in repeats:
            recorded = repeat["metrics"]
            assert repeat["prediction_sha256"] == identity_hash
            assert (recorded["tp"], recorded["fp"], recorded["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
            assert_close(recorded["micro_f1"], metrics["micro_f1"])
            assert_close(recorded["macro_f1"], metrics["macro_f1"])
        return len(repeats)
    repeats = document["runs"]
    assert len(repeats) == 3
    recorded_identity_hash = hashlib.sha256(json.dumps(document["predictions"], separators=(",", ":")).encode()).hexdigest()
    for repeat in repeats:
        assert repeat["predictions_sha256"] == recorded_identity_hash
        assert (repeat["tp"], repeat["fp"], repeat["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
        assert_close(repeat["strict_micro_f1"], metrics["micro_f1"])
        recorded_macro = math.fsum(
            0.0 if row["n_truth_pairs"] == 0 and row["n_predicted_pairs"] == 0 else row["f1"]
            for row in per_sequence
        ) / len(per_sequence)
        assert_close(repeat["strict_macro_f1"], recorded_macro)
    return len(repeats)


def audit_arm(name, spec, rows, output_root):
    path = spec["path"]
    assert sha256(path) == spec["sha256"]
    document = json.loads(path.read_text())
    prediction_by_name, source = extract_predictions(document, spec["schema"])
    names = [row["name"] for row in source]
    assert len(names) == len(set(names)) == N_SEQUENCES
    assert names == [row["name"] for row in rows]
    assert set(prediction_by_name) == {row["name"] for row in rows}
    metrics, per_sequence = strict_metrics(rows, prediction_by_name)
    identity = [[row["name"], prediction_by_name[row["name"]]["pairs"]] for row in rows]
    identity_hash = hashlib.sha256(canonical(identity).encode()).hexdigest()
    repeats = verify_recorded(document, spec["schema"], metrics, per_sequence, identity_hash)
    if spec["schema"] == "rnaformer":
        identity_hash = hashlib.sha256(json.dumps(document["predictions"], separators=(",", ":")).encode()).hexdigest()
    write_jsonl(output_root / "predictions" / f"{name}_bprna_new_v3.jsonl",
                [{"name": prediction_name, "pairs": pairs} for prediction_name, pairs in identity])
    write_jsonl(output_root / "reports" / f"{name}_bprna_new_per_sequence_v3.jsonl", per_sequence)
    protocol = document.get("protocol", {
        key: value for key, value in document.items() if key not in {"per_sequence", "repeats", "predictions", "runs"}
    })
    if spec["schema"] == "spotrna":
        recorded_macro = document["repeats"][0]["metrics"]["macro_f1"]
    else:
        recorded_macro = document["runs"][0]["strict_macro_f1"]
    return {
        "status": "PASS_INDEPENDENT_RECOMPUTATION",
        "role": spec["role"],
        "artifact": str(path),
        "artifact_sha256": spec["sha256"],
        "prediction_identity_sha256": identity_hash,
        "unique_names": len(set(names)),
        "order_matches_frozen_data": True,
        "recorded_repeats_verified": repeats,
        "recorded_macro_f1": recorded_macro,
        "recorded_zero_pair_f1_convention": "F1=1 when both are empty" if spec["schema"] == "spotrna" else "F1=0 when both are empty; unified recomputation uses F1=1",
        "threshold": document.get("threshold"),
        "metrics": metrics,
        "recorded_protocol": protocol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl"))
    args = parser.parse_args()
    assert sha256(args.data) == DATA_SHA256
    rows = load_rows(args.data)
    assets = {}
    for name, (path, expected) in ASSETS.items():
        actual = sha256(path)
        assert actual == expected, (name, actual, expected)
        assets[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual, "status": "PASS_HASH"}
    results = {name: audit_arm(name, spec, rows, args.output_root) for name, spec in ARMS.items()}
    output = {
        "schema_version": 1,
        "status": "PASS_FULL_COVERAGE_INDEPENDENT_RECOMPUTATION",
        "dataset": {
            "name": "bprna_new",
            "path": str(args.data),
            "sha256": DATA_SHA256,
            "n_sequences": len(rows),
            "unique_names": len({row["name"] for row in rows}),
            "pair_convention": "zero-based inclusive; exact pairs; no crossing, canonicality, or minimum-span filtering by this scorer",
        },
        "computation": "CPU-only JSONL parsing and exact set scoring; no model inference and no project evaluator import",
        "assets": assets,
        "arms": results,
    }
    write_json(args.output_root / "reports/bprna_new_metrics_v3.json", output)
    print(json.dumps({
        name: {key: result["metrics"][key] for key in ("n_with_prediction", "coverage", "tp", "fp", "fn", "micro_f1", "macro_f1")}
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
