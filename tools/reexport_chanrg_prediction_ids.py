#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SPLITS = ("chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf")
ARM_SPECS = {
    "r2d": ("rinalmo_2d_s0", "{split}/r2d/result.json"),
    "xens2": ("xens2", "{split}/xens2/result.json"),
    "ens3": ("chanrg_ens3", "chanrg_ens3_gt/{split}/result.json"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_source(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [row.get("id") for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in ids):
        raise AssertionError(f"empty source or missing frozen id: {path}")
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate frozen source ids: {path}")
    for row in rows:
        if not isinstance(row.get("seq"), str) or not isinstance(row.get("pairs"), list):
            raise AssertionError(f"invalid frozen source schema: {path}")
    return rows


def count_pairs(pairs: list) -> set[tuple[int, int]]:
    result = {tuple(pair) for pair in pairs}
    if len(result) != len(pairs) or any(len(pair) != 2 for pair in result):
        raise AssertionError("duplicate or invalid pair")
    return result


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def validate_raw(source: list[dict], raw: dict, *, require_empty_names: bool = True) -> dict:
    rows = raw.get("per_sequence")
    if not isinstance(rows, list) or len(rows) != len(source):
        raise AssertionError("raw/source sequence count mismatch")
    if raw.get("n_sequences", len(rows)) != len(rows) or raw.get("n_skipped", 0) not in (0, None):
        raise AssertionError("raw export is incomplete or skipped source rows")
    tp_total = fp_total = fn_total = 0
    f1_values = []
    for index, (source_row, raw_row) in enumerate(zip(source, rows)):
        expected_name = "" if require_empty_names else source_row["id"]
        if raw_row.get("name") != expected_name:
            raise AssertionError(f"unexpected raw name at row {index}")
        gt = count_pairs(source_row["pairs"])
        pred = count_pairs(raw_row["pred_pairs"])
        n_gt = raw_row.get("n_gt_pairs", raw_row.get("n_gt"))
        n_pred = raw_row.get("n_pred_pairs", raw_row.get("n_pred"))
        if n_gt != len(gt) or n_pred != len(pred):
            raise AssertionError(f"raw pair count mismatch at row {index}")
        if "length" in raw_row and raw_row["length"] != len(source_row["seq"]):
            raise AssertionError(f"raw/source length mismatch at row {index}")
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        expected_f1 = 1.0 if not pred and not gt else 2 * tp / (len(pred) + len(gt))
        if not close(raw_row["f1"], expected_f1):
            raise AssertionError(f"raw/source ordered F1 mismatch at row {index}")
        tp_total += tp
        fp_total += fp
        fn_total += fn
        f1_values.append(expected_f1)
    micro = raw["pair_level"]["micro"]
    if (tp_total, fp_total, fn_total) != (micro["tp"], micro["fp"], micro["fn"]):
        raise AssertionError("raw micro pair counts do not recompute")
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    macro_f1 = math.fsum(f1_values) / len(f1_values)
    for key, expected in (("precision", precision), ("recall", recall), ("f1", micro_f1)):
        if not close(micro[key], expected):
            raise AssertionError(f"raw micro {key} does not recompute")
    if not close(raw["pair_level"]["macro"]["f1"], macro_f1):
        raise AssertionError("raw macro F1 does not recompute")
    return {
        "n_sequences": len(rows),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def registry_row(registry_path: Path | None, raw_path: Path) -> dict | None:
    if registry_path is None:
        return None
    matches = [row for row in load_json(registry_path)["rows"]
               if Path(row.get("path") or "") == raw_path]
    if len(matches) != 1:
        raise AssertionError(f"expected one frozen registry row for {raw_path}")
    row = matches[0]
    if row.get("status") != "FROZEN" or row.get("sha256") != sha256_file(raw_path):
        raise AssertionError(f"raw artifact does not match frozen registry: {raw_path}")
    return row


def reexport_prediction(source_path: Path, raw_path: Path, output_path: Path,
                        *, arm: str, split: str,
                        frozen_registry: Path | None = None) -> dict:
    source_path, raw_path, output_path = Path(source_path), Path(raw_path), Path(output_path)
    source = load_source(source_path)
    raw = load_json(raw_path)
    metrics = validate_raw(source, raw)
    frozen_row = registry_row(frozen_registry, raw_path)
    repaired = copy.deepcopy(raw)
    for source_row, output_row in zip(source, repaired["per_sequence"]):
        output_row["name"] = source_row["id"]
    stripped = copy.deepcopy(repaired)
    for original_row, output_row in zip(raw["per_sequence"], stripped["per_sequence"]):
        output_row["name"] = original_row["name"]
    if stripped != raw:
        raise AssertionError("a field other than per_sequence.name changed")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    row = {
        "prediction_id": f"chanrg_id_repaired/{arm}/{split}",
        "arm": arm,
        "model": ARM_SPECS[arm][0],
        "split": split,
        "status": "ID_PROVENANCE_REPAIRED",
        "source_data": file_record(source_path),
        "raw_prediction": file_record(raw_path),
        "repaired_prediction": file_record(output_path),
        "raw_empty_names": len(source),
        "repaired_names": len(source),
        "changed_fields": ["per_sequence[].name"],
        "metrics": metrics,
        "metrics_unchanged": True,
        "frozen_registry_prediction_id": frozen_row["prediction_id"] if frozen_row else None,
        "frozen_registry_sha256_match": frozen_row is not None,
    }
    if row["metrics"] != validate_raw(source, load_json(output_path), require_empty_names=False):
        raise AssertionError("re-exported metrics changed")
    return row


def export_all(source_root: Path, raw_root: Path, output_root: Path,
               frozen_registry: Path | None = None) -> dict:
    source_root, raw_root, output_root = Path(source_root), Path(raw_root), Path(output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    rows = []
    source_files = {}
    for split in SPLITS:
        source_path = source_root / f"{split}.jsonl"
        source_files[split] = file_record(source_path)
        for arm, (_model, raw_template) in ARM_SPECS.items():
            raw_path = raw_root / raw_template.format(split=split)
            output_path = output_root / "predictions" / arm / f"{split}.json"
            rows.append(reexport_prediction(
                source_path, raw_path, output_path, arm=arm, split=split,
                frozen_registry=frozen_registry))
    manifest = {
        "schema_version": 1,
        "manifest_id": "chanrg_prediction_id_provenance_20261126",
        "status": "PASS",
        "scope": {
            "splits": list(SPLITS),
            "arms": list(ARM_SPECS),
            "n_prediction_files": len(rows),
            "n_sequences_per_arm": sum(row["metrics"]["n_sequences"] for row in rows
                                       if row["arm"] == "r2d"),
            "n_prediction_rows_total": sum(row["metrics"]["n_sequences"] for row in rows),
        },
        "join_contract": {
            "source_id_field": "id",
            "raw_name_field": "per_sequence[].name",
            "order": "frozen JSONL record order equals raw per_sequence export order",
            "required_raw_name": "",
            "per_row_evidence": [
                "raw pred_pairs and frozen pairs recompute the stored F1",
                "raw predicted/reference pair counts match",
                "stored length matches frozen sequence length when present",
            ],
            "aggregate_evidence": "micro TP/FP/FN, precision, recall, F1 and macro F1 recompute",
            "export_implementations": {
                "r2d": "eval/ss/evaluate_decision.py",
                "xens2": "tools/xens2_eval.py",
                "ens3": "tools/eval_multi_head_ensemble.py -> eval/ss/evaluate_decision.py",
            },
        },
        "usage_policy": {
            "inference": "NOT_RUN",
            "model_selection": "NOT_RUN",
            "labels": "verification only; predictions and metrics are unchanged",
            "original_predictions": "read-only; original registry is not modified",
            "historical_results_scientific_gates_gec_seed0": "UNCHANGED",
        },
        "source_files": source_files,
        "frozen_registry": file_record(frozen_registry) if frozen_registry else None,
        "reexport_tool": {
            "path": "tools/reexport_chanrg_prediction_ids.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/root/rna_jepa_data/eval_decision"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path)
    args = parser.parse_args()
    manifest = export_all(args.source_root, args.raw_root, args.output_root,
                          args.frozen_registry)
    print(json.dumps({"status": manifest["status"], **manifest["scope"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
