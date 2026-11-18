#!/usr/bin/env python3
"""No-training label/mask ceiling and TR1c pair-environment coverage audit."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
MAIN_SPLITS = ("archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seq_of(row: dict) -> str:
    return str(row.get("seq", row.get("sequence", ""))).upper().replace("T", "U")


def name_of(row: dict, line_number: int) -> str:
    return str(row.get("name", row.get("id", f"row:{line_number}")))


def normalize_pairs(raw, length: int):
    valid, seen = set(), set()
    duplicate = invalid = 0
    for pair in raw or []:
        try:
            if len(pair) != 2:
                raise ValueError
            i, j = sorted((int(pair[0]), int(pair[1])))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not (0 <= i < j < length):
            invalid += 1
            continue
        if (i, j) in seen:
            duplicate += 1
            continue
        seen.add((i, j))
        valid.add((i, j))
    return valid, duplicate, invalid


def prf(pred: set, truth: set) -> dict:
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0),
            "recall": tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0}


class RangeTree:
    def __init__(self, values):
        self.n = len(values)
        self.inf = float("inf")
        size = 1
        while size < max(1, self.n):
            size *= 2
        self.size = size
        self.mn = [self.inf] * (2 * size)
        self.mx = [-self.inf] * (2 * size)
        for k, value in enumerate(values):
            self.mn[size + k] = self.mx[size + k] = value
        for k in range(size - 1, 0, -1):
            self.mn[k] = min(self.mn[2 * k], self.mn[2 * k + 1])
            self.mx[k] = max(self.mx[2 * k], self.mx[2 * k + 1])

    def query(self, left, right, maximum=False):
        if left >= right or self.n == 0:
            return -self.inf if maximum else self.inf
        left += self.size
        right += self.size
        value = -self.inf if maximum else self.inf
        while left < right:
            if left & 1:
                value = max(value, self.mx[left]) if maximum else min(value, self.mn[left])
                left += 1
            if right & 1:
                right -= 1
                value = max(value, self.mx[right]) if maximum else min(value, self.mn[right])
            left //= 2
            right //= 2
        return value


def pair_geometry(seq: str, pairs: set):
    """Return crossing, terminal-hairpin and local register features per pair."""
    ordered = sorted(pairs)
    starts = [p[0] for p in ordered]
    tree = RangeTree([p[1] for p in ordered])
    crossing_possible = False
    terminal = {}
    for idx, (i, j) in enumerate(ordered):
        lo = bisect.bisect_right(starts, i)
        hi = bisect.bisect_left(starts, j)
        if tree.query(lo, hi, maximum=True) > j:
            crossing_possible = True
        terminal[(i, j)] = not (tree.query(lo, hi, maximum=False) < j)
    crossing = {p: False for p in ordered}
    if crossing_possible:
        for a, (i, j) in enumerate(ordered):
            for k in range(a + 1, len(ordered)):
                x, y = ordered[k]
                if x >= j:
                    break
                if x > i and y > j:
                    crossing[(i, j)] = crossing[(x, y)] = True
    partner = Counter(x for p in ordered for x in p)
    conflict = {p: partner[p[0]] > 1 or partner[p[1]] > 1 for p in ordered}
    features = {}
    for i, j in ordered:
        left = 0
        while (i - left - 1, j + left + 1) in pairs:
            left += 1
        right = 0
        while (i + right + 1, j - right - 1) in pairs:
            right += 1
        pair_type = seq[i] + seq[j] if 0 <= i < j < len(seq) else "INVALID"
        legal = (seq[i], seq[j]) in PAIRS and j - i > 3
        features[(i, j)] = {
            "pair_i": i, "pair_j": j, "pair_type": pair_type,
            "span": j - i, "canonical": (seq[i], seq[j]) in PAIRS,
            "min_loop": j - i > 3, "legal_mask": legal,
            "nested": not crossing[(i, j)], "crossing": crossing[(i, j)],
            "partner_conflict": conflict[(i, j)],
            "terminal_hairpin": terminal[(i, j)],
            "hairpin_loop_length": j - i - 1 if terminal[(i, j)] else None,
            "helix_length": left + right + 1, "register": min(left, right),
            "inner_step": (i + 1, j - 1) in pairs,
            "outer_step": (i - 1, j + 1) in pairs,
        }
    return features


def span_bin(span):
    for label, lo, hi in (("0_3", 0, 4), ("4_9", 4, 10), ("10_24", 10, 25), ("25_49", 25, 50), ("50_99", 50, 100), ("100_199", 100, 200), ("200_plus", 200, None)):
        if span >= lo and (hi is None or span < hi):
            return label
    raise ValueError(span)


def helix_bin(n):
    return "1" if n == 1 else "2" if n == 2 else "3_4" if n <= 4 else "5_7" if n <= 7 else "8_12" if n <= 12 else "13_plus"


def register_bin(n):
    return str(n) if n <= 2 else "3_plus"


def length_bin(n):
    return "0_64" if n <= 64 else "65_128" if n <= 128 else "129_256" if n <= 256 else "257_512" if n <= 512 else "513_1024" if n <= 1024 else "1025_plus"


def gc_bin(x):
    return "0_035" if x < .35 else "035_045" if x < .45 else "045_055" if x < .55 else "055_065" if x < .65 else "065_100"


def loop_bin(f):
    if not f["terminal_hairpin"]:
        return "NONTERMINAL"
    n = f["hairpin_loop_length"]
    return "0_2" if n < 3 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11_20" if n <= 20 else "21_plus"


def keys_for(f):
    k1 = f["pair_type"]
    k2 = f"{k1}|span={span_bin(f['span'])}"
    local = (f"{k2}|helix={helix_bin(f['helix_length'])}|reg={register_bin(f['register'])}"
             f"|in={int(f['inner_step'])}|out={int(f['outer_step'])}|loop={loop_bin(f)}")
    exact_loop = f["hairpin_loop_length"] if f["terminal_hairpin"] else "NONTERMINAL"
    exact = (f"{f['pair_type']}|span={f['span']}|helix={f['helix_length']}|reg={f['register']}"
             f"|in={int(f['inner_step'])}|out={int(f['outer_step'])}|loop={exact_loop}")
    return k1, k2, local, exact


def dataset_rows(registry, dataset_id):
    with open(registry[dataset_id]["path"], encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def mask_audit(registry, out: Path, only=None, limit=0):
    per_sequence = []
    summaries = {}
    for dataset_id, spec in registry.items():
        if only and dataset_id not in only:
            continue
        if spec.get("status") != "FROZEN" or not spec.get("exists"):
            continue
        path = Path(spec["path"])
        if not path.is_file():
            continue
        counts = Counter()
        macro_recall, macro_compatible = [], []
        n_lines = 0
        for line_number, row in dataset_rows(registry, dataset_id):
            n_lines += 1
            if limit and n_lines > limit:
                break
            seq = seq_of(row)
            pairs, duplicate, invalid = normalize_pairs(row.get("pairs"), len(seq))
            if not pairs and "structure" not in row and "dbn" not in row:
                counts["out_of_scope_no_structure_labels"] += 1
                continue
            features = pair_geometry(seq, pairs)
            n = len(pairs)
            legal = sum(f["legal_mask"] for f in features.values())
            compatible = sum(f["legal_mask"] and f["nested"] and not f["partner_conflict"] for f in features.values())
            non_acgu = sum(c not in "ACGU" for c in seq)
            record = {
                "dataset_id": dataset_id, "name": name_of(row, line_number),
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(), "length": len(seq),
                "n_pairs_raw": len(row.get("pairs") or []), "n_pairs": n,
                "n_duplicate_pairs": duplicate, "n_invalid_pairs": invalid,
                "n_pairs_field": row.get("n_pairs"), "n_crossing_field": row.get("n_crossing"),
                "n_dropped_crossing": int(row.get("n_dropped_crossing") or 0),
                "n_dropped_illegal": int(row.get("n_dropped_illegal") or 0),
                "is_pseudoknot_field": row.get("is_pseudoknot"),
                "n_canonical": sum(f["canonical"] for f in features.values()),
                "n_nested": sum(f["nested"] for f in features.values()),
                "n_min_loop": sum(f["min_loop"] for f in features.values()),
                "n_legal_mask": legal, "n_crossing_pairs": sum(f["crossing"] for f in features.values()),
                "n_partner_conflict_pairs": sum(f["partner_conflict"] for f in features.values()),
                "n_noncanonical_pairs": sum(not f["canonical"] for f in features.values()),
                "n_fully_compatible_pairs": compatible, "n_non_acgu_residues": non_acgu,
                "mask_theoretical_max_pair_recall": legal / n if n else None,
                "fully_compatible_pair_fraction": compatible / n if n else None,
                "missing_residue_deletion_status": "NOT_OBSERVABLE_FROM_FROZEN_JSONL" if spec.get("modified_parent_mapping") == "NOT_AVAILABLE_IN_JSONL" else "REQUIRES_PARENT_MAPPING_AUDIT",
            }
            dropped = record["n_dropped_crossing"] + record["n_dropped_illegal"]
            record["retained_pair_yield_vs_recorded_pre_clean"] = n / (n + dropped) if n + dropped else None
            per_sequence.append(record)
            for key, value in record.items():
                if key.startswith("n_") and isinstance(value, (int, float)) and value is not None:
                    counts[key] += value
            counts["n_sequences"] += 1
            if n:
                macro_recall.append(legal / n)
                macro_compatible.append(compatible / n)
        n_pairs = counts["n_pairs"]
        dropped = counts["n_dropped_crossing"] + counts["n_dropped_illegal"]
        summaries[dataset_id] = {
            **dict(counts), "label_source": spec.get("label_source"), "role": spec.get("role"),
            "path": spec["path"], "sha256": spec.get("sha256"),
            "mask_theoretical_max_pair_recall_micro": counts["n_legal_mask"] / n_pairs if n_pairs else None,
            "mask_theoretical_max_pair_recall_macro": float(np.mean(macro_recall)) if macro_recall else None,
            "fully_compatible_pair_fraction_micro": counts["n_fully_compatible_pairs"] / n_pairs if n_pairs else None,
            "fully_compatible_pair_fraction_macro": float(np.mean(macro_compatible)) if macro_compatible else None,
            "retained_pair_yield_vs_recorded_pre_clean": n_pairs / (n_pairs + dropped) if n_pairs + dropped else None,
            "historical_cleaning_fields_available": any(k in counts for k in ("n_dropped_crossing", "n_dropped_illegal")),
            "limit_per_dataset": limit or None,
        }
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_sequence).to_parquet(out / "label_mask_per_sequence.parquet", index=False)
    report = {"schema_version": 1, "created_utc": utcnow(), "scope": "NO_TRAINING_LABEL_AND_MASK_AUDIT",
              "denominator": "all unique valid current GT pairs; illegal GT remain in recall denominator",
              "mask_definition": "canonical AU/UA/GC/CG/GU/UG and span>3",
              "compatible_fraction_is_not_nested_oracle": True,
              "datasets": summaries}
    (out / "label_mask_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def load_predictions(path: Path):
    obj = json.loads(path.read_text())
    rows = obj.get("per_sequence", obj.get("predictions"))
    if not isinstance(rows, list):
        raise ValueError(f"no per-sequence predictions: {path}")
    out = {}
    for index, row in enumerate(rows):
        name = str(row.get("name", row.get("id", f"row:{index}")))
        pairs, _, _ = normalize_pairs(row.get("pred_pairs", row.get("pairs", [])), 10**12)
        out.setdefault(name, []).append({"pairs": pairs, "length": row.get("length"),
                                        "n_gt": row.get("n_gt", row.get("n_gt_pairs")), "f1": row.get("f1"), "index": index})
    return out


def verified_positional(truth_rows, predictions):
    if set(predictions) != {""}:
        return None
    pred = predictions[""]
    names = [name_of(row, i + 1) for i, row in enumerate(truth_rows)]
    if len(set(names)) != len(names) or len(pred) != len(truth_rows):
        raise ValueError("blank-name prediction alignment failed record/unique-name check")
    aligned = {}
    for row, name, p in zip(truth_rows, names, pred):
        seq = seq_of(row)
        truth, _, _ = normalize_pairs(row.get("pairs"), len(seq))
        if p["length"] is not None and int(p["length"]) != len(seq):
            raise ValueError("blank-name length mismatch")
        if p["n_gt"] is not None and int(p["n_gt"]) != len(truth):
            raise ValueError("blank-name n_gt mismatch")
        if p["f1"] is not None and abs(float(p["f1"]) - prf(p["pairs"], truth)["f1"]) > 1e-8:
            raise ValueError("blank-name F1 mismatch")
        aligned[name] = p
    return aligned


def train_index(registry):
    counts = [Counter(), Counter(), Counter(), Counter()]
    sequence_hashes = set()
    for _, row in dataset_rows(registry, "bprna_tr1c"):
        seq = seq_of(row)
        sequence_hashes.add(hashlib.sha256(seq.encode()).hexdigest())
        pairs, _, _ = normalize_pairs(row.get("pairs"), len(seq))
        for f in pair_geometry(seq, pairs).values():
            if not f["legal_mask"]:
                continue
            for counter, key in zip(counts, keys_for(f)):
                counter[key] += 1
    return counts, sequence_hashes


def coverage_audit(registry, predictions, out: Path, only=None, limit=0):
    train_counts, train_sequences = train_index(registry)
    e0_path = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/foldnar_v1/e0_expanded/error_decomposition.parquet")
    e0 = pd.read_parquet(e0_path, columns=["model", "split", "repeat", "name", "pair_tp", "pair_fn"])
    e0 = e0[(e0.model == "xens2") & (e0["repeat"].isna() | (e0["repeat"] == 0))]
    e0 = {(r.split, r.name): (int(r.pair_tp), int(r.pair_fn)) for r in e0.itertuples()}
    pair_rows, bucket_rows, join_rows = [], [], []
    summaries = {}
    targets = ["bprna_vl0", *MAIN_SPLITS]
    for dataset_id in targets:
        if only and dataset_id not in only:
            continue
        spec = registry[dataset_id]
        all_truth = list(dataset_rows(registry, dataset_id))
        if limit:
            all_truth = all_truth[:limit]
        pred_spec = predictions.get(f"xens2/{dataset_id}") or predictions.get(f"xens2/{dataset_id}/repeat0")
        pred = load_predictions(Path(pred_spec["path"])) if pred_spec else {}
        evaluated = bool(pred)
        positional = verified_positional([r for _, r in all_truth], pred) if pred else None
        names = [name_of(row, line) for line, row in all_truth]
        if positional is None and any(value > 1 for value in Counter(names).values()):
            raise ValueError(f"duplicate truth name requires explicit join audit: {dataset_id}")
        joined = positional if positional is not None else pred
        join_method = "positional_verified" if positional is not None else "exact_name" if pred else "NO_FROZEN_XENS2_PREDICTION"
        join_rows.append({"dataset_id": dataset_id, "join_method": join_method,
                          "prediction_id": (pred_spec or {}).get("prediction_id"),
                          "prediction_path": (pred_spec or {}).get("path"),
                          "prediction_sha256": (pred_spec or {}).get("sha256"),
                          "n_truth": len(all_truth), "n_prediction_records": sum(map(len, pred.values()))})
        counts = Counter()
        local_buckets = Counter()
        for (line_number, row), name in zip(all_truth, names):
            seq = seq_of(row)
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()
            truth, _, _ = normalize_pairs(row.get("pairs"), len(seq))
            p = joined.get(name)
            if p is None and positional is None and len(joined.get(name, [])) == 1:
                p = joined[name][0]
            elif isinstance(p, list):
                if len(p) != 1:
                    raise ValueError(f"ambiguous prediction: {dataset_id}/{name}")
                p = p[0]
            pred_pairs = p["pairs"] if p else set()
            metrics = prf(pred_pairs, truth) if p else None
            if p and (dataset_id, name) in e0:
                expected = e0[(dataset_id, name)]
                if (metrics["tp"], metrics["fn"]) != expected:
                    raise ValueError(f"E0 pair mismatch {dataset_id}/{name}: {metrics} != {expected}")
            features = pair_geometry(seq, truth)
            family = row.get("family") or row.get("rfam_id") or row.get("rfam")
            clan = row.get("clan")
            gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else None
            for pair, f in features.items():
                k1, k2, k3, k4 = keys_for(f)
                legal = f["legal_mask"]
                status = "NOT_EVALUATED" if not p else "FN" if pair not in pred_pairs else "TP"
                train_n = [counter[key] if legal else None for counter, key in zip(train_counts, (k1, k2, k3, k4))]
                out_row = {
                    "dataset_id": dataset_id, "name": name, "sequence_sha256": seq_hash,
                    **f, "length": len(seq), "length_bin": length_bin(len(seq)),
                    "gc": gc, "gc_bin": gc_bin(gc) if gc is not None else None,
                    "family": family, "clan": clan, "source": row.get("source"),
                    "label_source": spec.get("label_source"),
                    "experimental_method": row.get("experimental_method", row.get("method")),
                    "train_count_pair_type": train_n[0], "train_count_pair_span": train_n[1], "train_count_local": train_n[2],
                    "train_count_local_exact": train_n[3],
                    "sequence_exact_in_tr1c": seq_hash in train_sequences, "xens2_status": status,
                }
                pair_rows.append(out_row)
                counts["n_pairs"] += 1
                counts["n_legal"] += int(legal)
                counts["n_fn"] += int(status == "FN")
                counts["n_tp"] += int(status == "TP")
                counts["n_legal_fn"] += int(legal and status == "FN")
                counts["n_illegal_fn"] += int(not legal and status == "FN")
                dimensions = {
                    "pair_type": f["pair_type"], "span": span_bin(f["span"]),
                    "helix": helix_bin(f["helix_length"]), "register": register_bin(f["register"]),
                    "loop": loop_bin(f), "length": length_bin(len(seq)),
                    "gc": gc_bin(gc) if gc is not None else "UNKNOWN", "family": family or "UNKNOWN",
                    "clan": clan or "UNKNOWN", "source": row.get("source") or "UNKNOWN",
                    "label_source": spec.get("label_source") or "UNKNOWN",
                }
                if legal:
                    for level, n in zip(("pair_type", "pair_span", "local", "local_exact"), train_n):
                        dimensions[f"train_{level}"] = "0" if n == 0 else "1_9" if n < 10 else "10_99" if n < 100 else "100_plus"
                for dimension, bucket in dimensions.items():
                    key = (dimension, str(bucket))
                    local_buckets[key] += 1
                    if status == "FN":
                        local_buckets[(dimension, str(bucket), "fn")] += 1
                    if legal:
                        local_buckets[(dimension, str(bucket), "legal")] += 1
                        if status == "FN":
                            local_buckets[(dimension, str(bucket), "legal_fn")] += 1
        for dimension, bucket in sorted({(d, b) for d, b, *_ in local_buckets}):
            n = local_buckets[(dimension, bucket)]
            fn = local_buckets[(dimension, bucket, "fn")]
            legal_n = local_buckets[(dimension, bucket, "legal")]
            legal_fn = local_buckets[(dimension, bucket, "legal_fn")]
            bucket_rows.append({"dataset_id": dataset_id, "dimension": dimension, "bucket": bucket,
                                "n_pairs": n, "n_fn": fn if evaluated else None,
                                "fn_rate": fn / n if evaluated and n else None,
                                "n_legal_pairs": legal_n, "n_legal_fn": legal_fn if evaluated else None,
                                "legal_fn_rate": legal_fn / legal_n if evaluated and legal_n else None})
        n = counts["n_pairs"]
        summaries[dataset_id] = {**dict(counts), "join_method": join_method,
                                 "pair_recall": counts["n_tp"] / (counts["n_tp"] + counts["n_fn"]) if evaluated and counts["n_tp"] + counts["n_fn"] else None,
                                 "legal_pair_recall": (counts["n_tp"] / (counts["n_tp"] + counts["n_legal_fn"])) if evaluated and counts["n_tp"] + counts["n_legal_fn"] else None,
                                 "limit_per_dataset": limit or None}
        if not evaluated:
            for key in ("n_fn", "n_tp", "n_legal_fn", "n_illegal_fn"):
                summaries[dataset_id][key] = None
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_parquet(out / "pair_environment_coverage.parquet", index=False)
    pd.DataFrame(bucket_rows).to_parquet(out / "coverage_buckets.parquet", index=False)
    pd.DataFrame(join_rows).to_parquet(out / "coverage_joins.parquet", index=False)
    report = {"schema_version": 1, "created_utc": utcnow(), "scope": "NO_TRAINING_TR1C_PAIR_ENVIRONMENT_COVERAGE",
              "training_reference": "bprna_tr1c", "train_count_space": "legal GT pair occurrences only",
              "local_key": "pair_type|span|helix_length|register|inner_step|outer_step|terminal_loop",
              "local_exact_key": "pair_type|exact_span|helix_length|register|inner_step|outer_step|terminal_loop|exact_loop_length",
              "datasets": summaries}
    (out / "coverage_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="mask,coverage")
    ap.add_argument("--datasets", default="", help="optional smoke-test subset; official run leaves empty")
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ART)
    args = ap.parse_args()
    registry = {r["dataset_id"]: r for r in json.loads((REPO / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]}
    predictions = {r["prediction_id"]: r for r in json.loads((REPO / "manifests/foldnar_v1/prediction_registry.json").read_text())["rows"]}
    only = set(args.datasets.split(",")) if args.datasets else None
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    tasks = set(args.tasks.split(","))
    if "mask" in tasks:
        results["mask"] = mask_audit(registry, args.out, only=only, limit=args.limit_per_dataset)
    if "coverage" in tasks:
        results["coverage"] = coverage_audit(registry, predictions, args.out, only=only, limit=args.limit_per_dataset)
    manifest = {"schema_version": 1, "created_utc": utcnow(), "official": not args.limit_per_dataset and not only,
                "tasks": sorted(tasks), "only": sorted(only) if only else None,
                "limit_per_dataset": args.limit_per_dataset or None,
                "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__).resolve()),
                "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                "data_registry_sha256": sha256(REPO / "manifests/foldnar_v1/data_registry.json"),
                "prediction_registry_sha256": sha256(REPO / "manifests/foldnar_v1/prediction_registry.json"),
                "result_datasets": {k: list(v["datasets"]) for k, v in results.items()}}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
