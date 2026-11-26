#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SPLITS = {"chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"}
EXPECTED_ARMS = {"r2d", "xens2", "ens3"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(record: dict) -> dict:
    path = Path(record["path"])
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if actual != record:
        raise AssertionError(f"file identity mismatch: {path}")
    return actual


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_source(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [row.get("id") for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in ids):
        raise AssertionError(f"missing frozen source id: {path}")
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate frozen source id: {path}")
    return rows


def pair_set(pairs: list) -> set[tuple[int, int]]:
    result = {tuple(pair) for pair in pairs}
    if len(result) != len(pairs) or any(len(pair) != 2 for pair in result):
        raise AssertionError("duplicate or invalid pair")
    return result


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def without_raw_names(raw: dict, repaired: dict) -> tuple[dict, dict]:
    left = copy.deepcopy(raw)
    right = copy.deepcopy(repaired)
    for row in left["per_sequence"]:
        row["name"] = None
    for row in right["per_sequence"]:
        row["name"] = None
    return left, right


def verify_prediction(row: dict, source: list[dict], frozen_rows: dict[str, dict]) -> dict:
    raw_path = Path(row["raw_prediction"]["path"])
    output_path = Path(row["repaired_prediction"]["path"])
    check_file(row["raw_prediction"])
    check_file(row["repaired_prediction"])
    frozen = frozen_rows.get(str(raw_path))
    if frozen is None or frozen.get("status") != "FROZEN":
        raise AssertionError(f"raw artifact lacks frozen registry identity: {raw_path}")
    if frozen["sha256"] != row["raw_prediction"]["sha256"]:
        raise AssertionError(f"raw registry hash mismatch: {raw_path}")
    if row["frozen_registry_prediction_id"] != frozen["prediction_id"]:
        raise AssertionError(f"raw registry prediction id mismatch: {raw_path}")
    raw = load_json(raw_path)
    repaired = load_json(output_path)
    raw_rows = raw["per_sequence"]
    output_rows = repaired["per_sequence"]
    if len(source) != len(raw_rows) or len(source) != len(output_rows):
        raise AssertionError(f"sequence count mismatch: {output_path}")
    if raw.get("n_sequences", len(raw_rows)) != len(source):
        raise AssertionError(f"raw n_sequences mismatch: {raw_path}")
    if repaired.get("n_sequences", len(output_rows)) != len(source):
        raise AssertionError(f"output n_sequences mismatch: {output_path}")
    raw_without_names, repaired_without_names = without_raw_names(raw, repaired)
    if raw_without_names != repaired_without_names:
        raise AssertionError(f"non-name content changed: {output_path}")
    tp_total = fp_total = fn_total = 0
    f1_values = []
    for index, (source_row, raw_row, output_row) in enumerate(
            zip(source, raw_rows, output_rows)):
        if raw_row.get("name") != "" or output_row.get("name") != source_row["id"]:
            raise AssertionError(f"ID provenance mismatch at row {index}: {output_path}")
        gt = pair_set(source_row["pairs"])
        pred = pair_set(output_row["pred_pairs"])
        n_gt = output_row.get("n_gt_pairs", output_row.get("n_gt"))
        n_pred = output_row.get("n_pred_pairs", output_row.get("n_pred"))
        if n_gt != len(gt) or n_pred != len(pred):
            raise AssertionError(f"pair count mismatch at row {index}: {output_path}")
        if "length" in output_row and output_row["length"] != len(source_row["seq"]):
            raise AssertionError(f"length mismatch at row {index}: {output_path}")
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        f1 = 1.0 if not pred and not gt else 2 * tp / (len(pred) + len(gt))
        if not close(output_row["f1"], f1):
            raise AssertionError(f"F1 mismatch at row {index}: {output_path}")
        f1_values.append(f1)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro = repaired["pair_level"]["micro"]
    if (micro["tp"], micro["fp"], micro["fn"]) != (tp_total, fp_total, fn_total):
        raise AssertionError(f"micro counts mismatch: {output_path}")
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    macro_f1 = math.fsum(f1_values) / len(f1_values)
    for key, expected in (("precision", precision), ("recall", recall), ("f1", micro_f1)):
        if not close(micro[key], expected):
            raise AssertionError(f"micro {key} mismatch: {output_path}")
    if not close(repaired["pair_level"]["macro"]["f1"], macro_f1):
        raise AssertionError(f"macro F1 mismatch: {output_path}")
    expected_metrics = {
        "n_sequences": len(source), "micro_f1": micro_f1, "macro_f1": macro_f1,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }
    if row["metrics"] != expected_metrics:
        raise AssertionError(f"manifest metrics mismatch: {output_path}")
    if row["changed_fields"] != ["per_sequence[].name"] or row["metrics_unchanged"] is not True:
        raise AssertionError(f"invalid repair scope: {output_path}")
    return {
        "prediction_id": row["prediction_id"],
        "n_sequences": len(source),
        "raw_empty_names": len(source),
        "verified_ids_in_source_order": len(source),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "raw_sha256": row["raw_prediction"]["sha256"],
        "repaired_sha256": row["repaired_prediction"]["sha256"],
        "status": "PASS",
    }


def verify_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    if manifest["status"] != "PASS" or manifest["schema_version"] != 1:
        raise AssertionError("invalid manifest status or schema")
    if len(manifest["rows"]) != 12:
        raise AssertionError("expected exactly 12 repaired predictions")
    identities = {(row["arm"], row["split"]) for row in manifest["rows"]}
    if identities != {(arm, split) for arm in EXPECTED_ARMS for split in EXPECTED_SPLITS}:
        raise AssertionError("prediction scope mismatch")
    registry_record = manifest["frozen_registry"]
    check_file(registry_record)
    frozen_rows = {row["path"]: row for row in load_json(Path(registry_record["path"]))["rows"]
                   if row.get("path")}
    tool = manifest["reexport_tool"]
    tool_path = Path(__file__).resolve().parents[1] / tool["path"]
    if sha256_file(tool_path) != tool["sha256"]:
        raise AssertionError("reexport tool hash mismatch")
    source_cache = {}
    for split, record in manifest["source_files"].items():
        check_file(record)
        source_cache[split] = load_source(Path(record["path"]))
    results = [verify_prediction(row, source_cache[row["split"]], frozen_rows)
               for row in manifest["rows"]]
    scope = manifest["scope"]
    if scope["n_prediction_rows_total"] != sum(row["n_sequences"] for row in results):
        raise AssertionError("manifest total row count mismatch")
    if scope["n_sequences_per_arm"] != sum(row["n_sequences"] for row in results
                                          if row["prediction_id"].split("/")[1] == "r2d"):
        raise AssertionError("manifest per-arm row count mismatch")
    return {
        "schema_version": 1,
        "status": "PASS",
        "verification": "independent ordered-ID, raw-hash, pair-count, per-row F1 and aggregate metric recomputation",
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "verifier": {
            "path": "tools/verify_chanrg_prediction_id_provenance.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "n_prediction_files": len(results),
        "n_prediction_rows_total": sum(row["n_sequences"] for row in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify_manifest(args.manifest)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "n_prediction_files": report["n_prediction_files"],
                      "n_prediction_rows_total": report["n_prediction_rows_total"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
