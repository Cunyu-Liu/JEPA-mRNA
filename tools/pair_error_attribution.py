#!/usr/bin/env python3
"""Attribute C2 VL0 false-negative pairs with direct, per-pair evidence.

The formal categories are deliberately conservative. LABEL and MASK describe
input/task exclusions; DECODER requires direct score/posterior support for a
pair that the selected decoder omitted; DATA_SHIFT requires zero direct TR1c
support for the pair's structural environment. Remaining scoring failures are
RAW_SCORE, not unsubstantiated REPRESENTATION or OPTIMIZATION claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_decode_common import decode_mea  # noqa: E402
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402

ARMS = ("C2",)
CATEGORIES = ("LABEL", "MASK", "RAW_SCORE", "DECODER", "OPTIMIZATION",
              "DATA_SHIFT", "REPRESENTATION", "UNKNOWN")
CANONICAL = {"AU", "UA", "GC", "CG", "GU", "UG"}
SPAN_BINS = ((4, 10), (11, 20), (21, 50), (51, 100), (101, 200))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalize_sequence(value: str) -> str:
    return str(value).upper().replace("T", "U")


def span_bin(span: int) -> str:
    for lower, upper in SPAN_BINS:
        if lower <= span <= upper:
            return f"{lower}-{upper}"
    if span >= 201:
        return "201+"
    return "LT4"


def pair_environments(seq: str, pairs: list[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    """Describe each pair by its enclosed loop and exact-stem register."""
    pair_set = {tuple(pair) for pair in pairs}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    children: dict[tuple[int, int], list[tuple[int, int]]] = {pair: [] for pair in pairs}
    for pair in pairs:
        i, j = pair
        enclosing = [q for q in pairs if q[0] < i and q[1] > j]
        closest = min(enclosing, key=lambda q: q[1] - q[0]) if enclosing else None
        parent[pair] = closest
        if closest is not None:
            children[closest].append(pair)
    output = {}
    for i, j in pairs:
        pair = (i, j)
        direct = children[pair]
        if not direct:
            loop_type, junction_order = "HAIRPIN", 0
        elif len(direct) == 1:
            ci, cj = direct[0]
            left_gap, right_gap = ci - i - 1, j - cj - 1
            junction_order = 1
            if left_gap == 0 and right_gap == 0:
                loop_type = "STACK"
            elif left_gap == 0 or right_gap == 0:
                loop_type = "BULGE"
            else:
                loop_type = "INTERNAL"
        else:
            loop_type, junction_order = "MULTILOOP", len(direct) + 1
        outer = pair
        while (outer[0] - 1, outer[1] + 1) in pair_set:
            outer = (outer[0] - 1, outer[1] + 1)
        stem = []
        cursor = outer
        while cursor in pair_set:
            stem.append(cursor)
            cursor = (cursor[0] + 1, cursor[1] - 1)
        register = stem.index(pair)
        if len(stem) == 1:
            stem_position = "SINGLE"
        elif register == 0:
            stem_position = "OUTER"
        elif register == len(stem) - 1:
            stem_position = "INNER"
        else:
            stem_position = "MIDDLE"
        pair_type = seq[i] + seq[j] if 0 <= i < len(seq) and 0 <= j < len(seq) else "INVALID"
        output[pair] = {
            "pair_type": pair_type,
            "span": j - i,
            "span_bin": span_bin(j - i),
            "enclosed_loop_type": loop_type,
            "junction_order": junction_order,
            "stem_length": len(stem),
            "stem_register": register,
            "stem_position": stem_position,
            "parent_pair": list(parent[pair]) if parent[pair] is not None else None,
            "n_direct_children": len(direct),
        }
    return output


def environment_key(environment: dict, level: str = "environment") -> tuple:
    if level == "base":
        return environment["pair_type"], environment["span_bin"]
    if level == "exact":
        return (environment["pair_type"], environment["span_bin"],
                environment["enclosed_loop_type"], environment["stem_position"])
    return (environment["pair_type"], environment["span_bin"],
            environment["enclosed_loop_type"])


def qa_record(record: dict) -> dict:
    seq = normalize_sequence(record["seq"])
    pairs = [(int(i), int(j)) for i, j in record.get("pairs", [])]
    issues = []
    invalid_characters = sorted(set(seq) - set("ACGU"))
    if invalid_characters:
        issues.append("INVALID_SEQUENCE_CHARACTERS")
    if len(record.get("structure", seq)) != len(seq):
        issues.append("STRUCTURE_LENGTH_MISMATCH")
    if int(record.get("n_pairs", len(pairs))) != len(pairs):
        issues.append("N_PAIRS_COUNT_MISMATCH")
    partner_owner = defaultdict(list)
    pair_label_issues = defaultdict(list)
    pair_mask_issues = defaultdict(list)
    for index, (i, j) in enumerate(pairs):
        if i < 0 or j < 0 or i >= len(seq) or j >= len(seq) or i >= j:
            pair_label_issues[(i, j)].append("MALFORMED_OR_OUT_OF_RANGE")
            continue
        partner_owner[i].append(j)
        partner_owner[j].append(i)
        if seq[i] + seq[j] not in CANONICAL:
            pair_mask_issues[(i, j)].append("NONCANONICAL")
        if j - i <= 3:
            pair_mask_issues[(i, j)].append("MIN_LOOP")
    for pair in pairs:
        i, j = pair
        if len(partner_owner[i]) > 1 or len(partner_owner[j]) > 1:
            pair_label_issues[pair].append("PARTNER_CONFLICT")
    pair_set = set(pairs)
    for i, j in pairs:
        for k, l in pairs:
            if i < k < j < l:
                pair_mask_issues[(i, j)].append("CROSSING")
    return {
        "name": str(record.get("name", "")),
        "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
        "length": len(seq),
        "gc_fraction": (seq.count("G") + seq.count("C")) / max(1, len(seq)),
        "invalid_characters": invalid_characters,
        "record_issues": issues,
        "pair_label_issues": pair_label_issues,
        "pair_mask_issues": pair_mask_issues,
        "n_pairs": len(pairs),
        "n_dropped_illegal": int(record.get("n_dropped_illegal", 0)),
        "n_dropped_crossing": int(record.get("n_dropped_crossing", 0)),
        "n_crossing_remaining": sum(1 for pair in pairs if "CROSSING" in pair_mask_issues[pair]) // 2,
        "source": str(record.get("source", "")),
        "label_source": "DATABASE",
        "experimental_provenance": "UNCHECKED_FROM_AGGREGATED_BPRNA_JSONL",
        "modified_parent_provenance": "UNCHECKED_NO_MMCIF_MAPPING_IN_JSONL",
        "normalized_sequence": seq,
        "normalized_pairs": pairs,
        "pair_set": pair_set,
    }


def raw_partner_ranks(scores: np.ndarray, mask: np.ndarray, i: int, j: int) -> tuple[int | None, int | None]:
    value = float(scores[i, j])
    if not math.isfinite(value):
        return None, None
    rank_i = 1 + int(np.sum(scores[i, mask[i]] > value))
    rank_j = 1 + int(np.sum(scores[j, mask[j]] > value))
    return rank_i, rank_j


def classify_fn(evidence: dict) -> str:
    if evidence["label_issues"]:
        return "LABEL"
    if not evidence["model_mask"] or evidence["mask_issues"]:
        return "MASK"
    if not evidence["finite_score"] or not evidence["finite_posterior"]:
        return "UNKNOWN"
    if evidence["raw_mutual_best"] or evidence["posterior_probability"] >= 0.5:
        return "DECODER"
    if evidence["training_environment_count"] == 0:
        return "DATA_SHIFT"
    return "RAW_SCORE"


def build_training_coverage(records: list[dict]) -> tuple[Counter, Counter, Counter, set[str], dict]:
    exact, environment, base = Counter(), Counter(), Counter()
    sequences = set()
    source_counts = Counter()
    total_pairs = 0
    for record in records:
        seq = normalize_sequence(record["seq"])
        sequences.add(hashlib.sha256(seq.encode()).hexdigest())
        source_counts[str(record.get("source", "")).split(":", 1)[0]] += 1
        pairs = [(int(i), int(j)) for i, j in record.get("pairs", [])]
        total_pairs += len(pairs)
        for pair, features in pair_environments(seq, pairs).items():
            exact[environment_key(features, "exact")] += 1
            environment[environment_key(features, "environment")] += 1
            base[environment_key(features, "base")] += 1
    return exact, environment, base, sequences, {
        "n_sequences": len(records), "n_pairs": total_pairs,
        "source_counts": dict(source_counts),
        "n_exact_environments": len(exact), "n_environments": len(environment),
    }


def selected_fn_lookup(stage: Path) -> dict[str, int]:
    rows = read_jsonl(stage / "decode_vl0/vl0_selected_per_sequence.jsonl")
    return {row["sequence_sha256"]: int(row["pair"]["fn"])
            for row in rows if row["arm"] == "C2"}


def analyze_seed(seed: int, artifact_root: Path, training_records: list[dict], coverage: tuple,
                 vl0_by_hash: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    exact_coverage, env_coverage, base_coverage, training_sequences, _ = coverage
    stage = artifact_root / f"seed{seed}_backbone_lr_1e-6"
    export_dir = stage / "export_train_off/bprna_vl0"
    config_path = stage / "decode_vl0/frozen_decode_config.json"
    verification_path = stage / "optimizer_scientific_verification.json"
    export_metadata = json.loads((stage / "export_train_off/export_metadata.json").read_text())
    config = json.loads(config_path.read_text())["config"]["C2"]
    verification = json.loads(verification_path.read_text())
    expected_fn = selected_fn_lookup(stage)
    fn_rows, predictions = [], []
    actual_fn = Counter()
    truth_environment_counts = Counter()
    for path in sorted(export_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            seq = normalize_sequence(str(data["seq"]))
            sequence_hash = hashlib.sha256(seq.encode()).hexdigest()
            if sequence_hash != path.stem:
                raise ValueError(f"sequence hash mismatch: {path}")
            truth = {(int(i), int(j)) for i, j in np.asarray(data["gt_pairs"]).reshape(-1, 2)}
            scores = np.asarray(data["plana_off"], dtype=np.float64)
            name = str(data["name"])
        qa = qa_record(vl0_by_hash[sequence_hash])
        environments = pair_environments(seq, sorted(truth))
        for environment in environments.values():
            truth_environment_counts[environment_key(environment, "environment")] += 1
        mask = valid_pair_mask(seq)
        temperature = float(config["temperature"])
        log_z, posterior = inside_outside(scores / temperature, mask)
        predicted = set(decode_mea(posterior, mask, float(config["gamma"])))
        if not np.isfinite(posterior).all() or not math.isfinite(float(log_z)):
            raise ValueError(f"non-finite posterior for seed={seed} sequence={sequence_hash}")
        predictions.append({
            "seed": seed, "arm": "C2", "name": name, "sequence_sha256": sequence_hash,
            "raw_npz": str(path), "raw_npz_sha256": sha256_file(path),
            "temperature": temperature, "gamma": float(config["gamma"]), "log_z": float(log_z),
            "predicted_pairs": [list(pair) for pair in sorted(predicted)],
        })
        false_negatives = sorted(truth - predicted)
        actual_fn[sequence_hash] = len(false_negatives)
        for i, j in false_negatives:
            pair = (i, j)
            environment = environments[pair]
            rank_i, rank_j = raw_partner_ranks(scores, mask, i, j)
            partner_i = next((b if a == i else a for a, b in predicted if a == i or b == i), None)
            partner_j = next((b if a == j else a for a, b in predicted if a == j or b == j), None)
            score = float(scores[i, j])
            post = float(posterior[i, j])
            env_count = int(env_coverage[environment_key(environment, "environment")])
            evidence = {
                "label_issues": sorted(set(qa["record_issues"] + qa["pair_label_issues"][pair])),
                "mask_issues": sorted(qa["pair_mask_issues"][pair]),
                "model_mask": bool(mask[i, j]),
                "finite_score": math.isfinite(score),
                "finite_posterior": math.isfinite(post),
                "raw_score": score if math.isfinite(score) else None,
                "raw_sigmoid": (1.0 / (1.0 + math.exp(-score))) if math.isfinite(score) else None,
                "raw_rank_for_i": rank_i,
                "raw_rank_for_j": rank_j,
                "raw_mutual_best": rank_i == 1 and rank_j == 1,
                "posterior_probability": post,
                "posterior_support_threshold": 0.5,
                "predicted_partner_for_i": partner_i,
                "predicted_partner_for_j": partner_j,
                "score_of_predicted_partner_for_i": float(scores[i, partner_i]) if partner_i is not None else None,
                "score_of_predicted_partner_for_j": float(scores[j, partner_j]) if partner_j is not None else None,
                "training_exact_count": int(exact_coverage[environment_key(environment, "exact")]),
                "training_environment_count": env_count,
                "training_base_count": int(base_coverage[environment_key(environment, "base")]),
                "exact_training_sequence_match": sequence_hash in training_sequences,
                "stable_low_lr_run": bool(verification["stability_pass"]),
                "run_max_loss": max(verification["runs"][arm]["max_loss"] for arm in ("on", "off")),
                "run_max_gnorm": max(verification["runs"][arm]["max_gnorm"] for arm in ("on", "off")),
                "causal_attribution_note": "OPTIMIZATION/REPRESENTATION require a separate counterfactual; not inferred from an FN alone",
            }
            category = classify_fn(evidence)
            fn_rows.append({
                "seed": seed, "arm": "C2", "dataset_id": "bprna_vl0", "name": name,
                "sequence_sha256": sequence_hash, "pair": [i, j],
                "sequence_length": len(seq), "sequence_gc_fraction": qa["gc_fraction"],
                "label_source": qa["label_source"], "source": qa["source"],
                "environment": environment, "attribution": category, "evidence": evidence,
            })
    mismatches = {key: {"expected": expected_fn.get(key), "actual": actual_fn.get(key, 0)}
                  for key in set(expected_fn) | set(actual_fn)
                  if expected_fn.get(key) != actual_fn.get(key, 0)}
    if mismatches:
        raise ValueError(f"C2 FN replay mismatch for seed={seed}: {mismatches}")
    diagnostics = {
        "seed": seed, "n_sequences": len(predictions), "n_fn": len(fn_rows),
        "expected_fn_total": sum(expected_fn.values()),
        "independent_fn_replay": "PASS",
        "decode_config": config,
        "frozen_decode_config_sha256": sha256_file(config_path),
        "optimizer_verification_sha256": sha256_file(verification_path),
        "export_metadata_sha256": sha256_file(stage / "export_train_off/export_metadata.json"),
        "checkpoint_sha256": export_metadata["checkpoint"]["sha256"],
        "export_git_head": export_metadata["git_head"],
        "truth_environment_counts": {"|".join(map(str, key)): value
                                     for key, value in sorted(truth_environment_counts.items())},
        "attribution_counts": dict(Counter(row["attribution"] for row in fn_rows)),
        "training_stability_pass": verification["stability_pass"],
        "old_external_panels_used": False,
    }
    return fn_rows, predictions, diagnostics


def summarize(fn_rows: list[dict], seed_diagnostics: list[dict]) -> dict:
    counts = Counter(row["attribution"] for row in fn_rows)
    eval_truth_counts = Counter()
    for diagnostics in seed_diagnostics:
        for key, value in diagnostics["truth_environment_counts"].items():
            pair_type, span, loop_type = key.split("|")
            eval_truth_counts[(pair_type, span, loop_type)] += value
    grouped = defaultdict(list)
    for row in fn_rows:
        grouped[environment_key(row["environment"], "environment")].append(row)
    high_fn = []
    for key, rows in grouped.items():
        support = max(row["evidence"]["training_environment_count"] for row in rows)
        eval_count = int(eval_truth_counts[key])
        rate = len(rows) / max(1, eval_count)
        if support < 20 or eval_count < 5 or rate < 0.2:
            continue
        high_fn.append({
            "environment": list(key), "n_fn": len(rows), "n_eval_truth_pairs": eval_count,
            "fn_rate_within_eval_environment": rate, "training_environment_count": support,
            "attribution_counts": dict(Counter(row["attribution"] for row in rows)),
            "seed_coverage": sorted({row["seed"] for row in rows}),
            "interpretation": "Stable low-LR runs exclude current numerical divergence; LABEL counts and DECODER evidence are reported separately, while RAW_SCORE does not by itself distinguish representation from optimization objective.",
        })
    high_fn.sort(key=lambda row: (-row["n_fn"], row["environment"]))
    pair_groups = defaultdict(list)
    for row in fn_rows:
        pair_groups[(row["sequence_sha256"], tuple(row["pair"]))].append(row)
    recurrent = []
    for (sequence_hash, pair), rows in pair_groups.items():
        if len(rows) < 2:
            continue
        recurrent.append({
            "sequence_sha256": sequence_hash, "pair": list(pair), "seeds": sorted(row["seed"] for row in rows),
            "attributions": dict(Counter(row["attribution"] for row in rows)),
            "raw_score_recurrent": all(row["attribution"] == "RAW_SCORE" for row in rows),
            "causal_status": "REPRESENTATION_VS_OPTIMIZATION_UNRESOLVED" if all(row["attribution"] == "RAW_SCORE" for row in rows) else "SEE_DIRECT_STAGE_ATTRIBUTION",
        })
    dominant = counts.most_common(1)[0][0] if counts else None
    return {
        "n_fn": len(fn_rows),
        "attribution_counts": dict(sorted(counts.items())),
        "attribution_fractions": {key: value / max(1, len(fn_rows)) for key, value in sorted(counts.items())},
        "dominant_attribution": dominant,
        "dominant_direct_evidence": "Per-pair hierarchy: LABEL/MASK -> direct DECODER support -> zero-support DATA_SHIFT -> residual RAW_SCORE.",
        "high_support_high_fn_environments": high_fn,
        "cross_seed_recurrent_fn": recurrent,
        "unassigned_fn": sum(1 for row in fn_rows if row["attribution"] not in CATEGORIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"))
    parser.add_argument("--vl0-data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite artifact directory: {args.out}")
    if args.final and sorted(args.seeds) != [0, 1, 2]:
        raise ValueError("final attribution requires seeds 0 1 2")
    training_records = read_jsonl(args.training_data)
    vl0_records = read_jsonl(args.vl0_data)
    vl0_by_hash = {hashlib.sha256(normalize_sequence(row["seq"]).encode()).hexdigest(): row for row in vl0_records}
    coverage = build_training_coverage(training_records)
    all_rows, all_predictions, seed_diagnostics = [], [], []
    for seed in args.seeds:
        rows, predictions, diagnostics = analyze_seed(seed, args.artifact_root, training_records, coverage, vl0_by_hash)
        all_rows.extend(rows)
        all_predictions.extend(predictions)
        seed_diagnostics.append(diagnostics)
    args.out.mkdir(parents=True)
    rows_path = args.out / "pair_error_attribution_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    predictions_path = args.out / "c2_replayed_predictions.jsonl"
    with predictions_path.open("w") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    vl0_qa = [qa_record(row) for row in vl0_records]
    total_truth = sum(row["n_pairs"] for row in vl0_qa)
    total_mask_excluded = sum(1 for row in vl0_qa for pair in row["normalized_pairs"] if row["pair_mask_issues"][pair])
    data_qa = {
        "n_sequences": len(vl0_qa), "n_truth_pairs": total_truth,
        "n_dropped_illegal_before_model": sum(row["n_dropped_illegal"] for row in vl0_qa),
        "n_dropped_crossing_before_model": sum(row["n_dropped_crossing"] for row in vl0_qa),
        "n_crossing_remaining": sum(row["n_crossing_remaining"] for row in vl0_qa),
        "n_truth_pairs_with_mask_issue": total_mask_excluded,
        "theoretical_max_recall_after_pair_mask": (total_truth - total_mask_excluded) / max(1, total_truth),
        "record_issue_counts": dict(Counter(issue for row in vl0_qa for issue in row["record_issues"])),
        "pair_label_issue_counts": dict(Counter(issue for row in vl0_qa for pair in row["normalized_pairs"] for issue in row["pair_label_issues"][pair])),
        "pair_mask_issue_counts": dict(Counter(issue for row in vl0_qa for pair in row["normalized_pairs"] for issue in row["pair_mask_issues"][pair])),
        "label_source": "DATABASE; underlying experimental/comparative provenance remains UNCHECKED per record",
        "modified_parent_qa": "UNCHECKED: aggregated JSONL has no mmCIF modified-parent mapping",
        "sequence_normalization": "uppercase T->U; 0-based pairs; symmetric matrices constructed by decoder",
    }
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_THREE_SEED_MODEL_DEV" if args.final else "PARTIAL_DEVELOPMENT_DEBUG",
        "scope": "C2 bprna_vl0 model-dev FN only; no external test, production, or SOTA claim",
        "categories": list(CATEGORIES),
        "attribution_policy": {
            "LABEL": "Malformed pairs, partner conflicts, or record-level label inconsistency",
            "MASK": "Truth pair excluded by canonical/min-loop/nested task mask",
            "DECODER": "Raw mutual-best partners or posterior >=0.5, but selected MEA omitted the pair",
            "DATA_SHIFT": "No TR1c pair in the same pair-type x span-bin x enclosed-loop environment",
            "RAW_SCORE": "Residual score/ranking failure with nonzero direct training-environment support",
            "OPTIMIZATION": "Not assigned without a direct per-pair optimization counterfactual",
            "REPRESENTATION": "Not assigned without a direct representation counterfactual",
            "UNKNOWN": "Missing or non-finite direct evidence",
        },
        "source_files": {
            "training_data": str(args.training_data), "training_data_sha256": sha256_file(args.training_data),
            "vl0_data": str(args.vl0_data), "vl0_data_sha256": sha256_file(args.vl0_data),
            "code": str(Path(__file__).resolve()), "code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "training_coverage": coverage[-1],
        "vl0_data_qa": data_qa,
        "seed_diagnostics": seed_diagnostics,
        "summary": summarize(all_rows, seed_diagnostics),
        "rows_jsonl": str(rows_path), "rows_sha256": sha256_file(rows_path),
        "predictions_jsonl": str(predictions_path), "predictions_sha256": sha256_file(predictions_path),
        "fn_records": all_rows,
        "old_external_panels_used": False,
        "production_repair": False,
        "sota_claim": False,
    }
    (args.out / "pair_error_attribution.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.out / "data_label_qa.json").write_text(json.dumps(data_qa, indent=2) + "\n")
    concise_summary = {key: output["summary"][key] for key in
                       ("n_fn", "attribution_counts", "attribution_fractions", "dominant_attribution", "unassigned_fn")}
    concise_summary["n_high_support_high_fn_environments"] = len(output["summary"]["high_support_high_fn_environments"])
    concise_summary["n_cross_seed_recurrent_fn"] = len(output["summary"]["cross_seed_recurrent_fn"])
    print(json.dumps({"status": output["status"], "out": str(args.out), "summary": concise_summary,
                      "vl0_data_qa": data_qa}, indent=2))


if __name__ == "__main__":
    main()
