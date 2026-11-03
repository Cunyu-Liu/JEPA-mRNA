#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256

SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
SPECS = {
    "eternafold": ("eternafold/clean_train_union", "eternafold_train_union", None),
    "bpfold": ("bpfold/clean", "bpfold_train_union", None),
    "mxfold2": ("mxfold2/clean_trainsetab", "mxfold2_trainsetab", None),
    "rnaformer_interfam": ("rnaformer/interfam/clean", "rnaformer_interfam_train", None),
    "rnaformer_bprna": ("rnaformer/clean_bprna", "rnaformer_bprna_train", None),
    "rnaformer_intrafam": ("rnaformer/intrafam/clean", "rnaformer_intrafam_train", None),
    "rnaformer_biophysical": ("rnaformer/biophysical/clean", "rnaformer_biophysical_train", None),
    "rinalmo_ft": ("rinalmo_ft/clean_tr0", "tr0", None),
    "spotrna": ("spotrna/clean_tr0_tr1", "spotrna_tr0_tr1", None),
    "kirigami_fast": ("kirigami_fast/clean_tr0", "tr0", None),
    "nucleicbert_ft": ("nucleicbert_ft/clean_tr1c", "tr1c", "input_max_length_with_special_tokens"),
    "nucleicbert_frozen": ("nucleicbert_frozen/clean", "nucleicbert_rnastralign_tr0", 598),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_pairs(pairs, limit=None):
    output = set()
    for pair in pairs:
        assert len(pair) == 2
        i, j = sorted((int(pair[0]), int(pair[1])))
        assert i < j and i >= 0
        if limit is None or j < limit:
            output.add((i, j))
    return output


def native_limit(result, spec_limit):
    if isinstance(spec_limit, int):
        assert result["sequence_core_max_length"] == spec_limit
        return spec_limit
    if spec_limit is None:
        return None
    return int(result[spec_limit]) - 2


def run_metrics(run):
    metrics = run.get("metrics", run)
    return metrics["tp"], metrics["fp"], metrics["fn"], metrics.get("strict_micro_f1", metrics.get("micro_f1"))


def score_entries(entries, names=None):
    selected = entries if names is None else {name: entries[name] for name in names}
    tp = fp = fn = 0
    for entry in selected.values():
        tp += len(entry["prediction"] & entry["truth"])
        fp += len(entry["prediction"] - entry["truth"])
        fn += len(entry["truth"] - entry["prediction"])
    return {"n": len(selected), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def matched_group(system_entries):
    names = set.intersection(*(set(entries) for entries in system_entries.values()))
    if not names:
        return {system: {"n": 0, "status": "EMPTY_INTERSECTION"} for system in system_entries}
    for name in sorted(names):
        reference = next(iter(system_entries.values()))[name]
        for entries in system_entries.values():
            entry = entries[name]
            assert entry["sequence"] == reference["sequence"], f"sequence mismatch: {name}"
            assert entry["truth"] == reference["truth"], f"truth mismatch: {name}"
    return {system: score_entries(entries, names) for system, entries in system_entries.items()}


def matched_pairs(system_entries):
    output = {}
    systems = sorted(system_entries)
    for index, first in enumerate(systems):
        for second in systems[index + 1:]:
            matched = matched_group({first: system_entries[first], second: system_entries[second]})
            n = matched[first]["n"]
            output[f"{first}|{second}"] = {
                "status": "PASS_MATCHED" if n else "EMPTY_INTERSECTION", "n": n,
                "systems": matched if n else {system: {"status": values["status"]}
                                               for system, values in matched.items()}}
    return output


def load_system_split(root, system, split):
    directory, corpus, limit_spec = SPECS[system]
    result_path = root / directory / f"{split}.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    corpus_root = root / "clean_splits" / corpus
    flat_layout = corpus in {"tr0", "tr1c"}
    data_path = corpus_root / (f"{split}.jsonl" if flat_layout else f"splits/{split}.jsonl")
    audit_path = corpus_root / (f"{split}.audit.json" if flat_layout else f"audits/{split}.audit.json")
    audit = json.loads(audit_path.read_text())
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[split]
    assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[split]
    assert audit["clean_sha256"] == sha256(data_path) and audit["clean_n"] == len(rows)
    assert audit["raw_n"] == len(rows) + audit["removed_exact"] + audit["removed_homology_only"]
    protocol = result.get("protocol") if isinstance(result.get("protocol"), dict) else {}
    assert result.get("data_sha256", protocol.get("data_sha256")) == audit["clean_sha256"]
    if not rows:
        assert result["status"] == "EMPTY_AFTER_DELEAK"
        return {"scope": "EMPTY_AFTER_DELEAK", "entries": {}, "n": 0,
                "result_sha256": sha256(result_path), "clean_sha256": audit["clean_sha256"],
                "audit_sha256": sha256(audit_path)}
    assert result["status"] == "PASS"
    runs = result.get("runs", result.get("repeats"))
    assert len(runs) == 3
    hashes = [run.get("predictions_sha256", run.get("prediction_sha256")) for run in runs]
    assert all(hashes) and len(set(hashes)) == 1
    limit = native_limit(result, limit_spec)
    predictions = result.get("predictions", result.get("per_sequence"))
    assert len(predictions) == len(rows)
    entries = {}
    for row, prediction in zip(rows, predictions):
        name = row["name"]
        assert name == prediction["name"] and name not in entries
        sequence = row["seq"].upper().replace("T", "U")
        truth = canonical_pairs(row.get("pairs", row.get("gt_pairs", [])), limit)
        predicted = canonical_pairs(prediction.get("pairs", prediction.get("pred_pairs", [])), limit)
        raw_predicted = canonical_pairs(prediction.get("pairs", prediction.get("pred_pairs", [])))
        assert predicted == raw_predicted, f"prediction outside native scope: {system}/{name}"
        assert all(j < len(sequence) for _, j in truth | predicted)
        entries[name] = {"sequence": sequence, "truth": truth, "prediction": predicted}
    native = score_entries(entries)
    for run in runs:
        tp, fp, fn, f1 = run_metrics(run)
        assert (tp, fp, fn) == (native["tp"], native["fp"], native["fn"]), (system, split)
        assert abs(f1 - native["f1"]) < 1e-12
    truncated = limit is not None and any(len(entry["sequence"]) > limit for entry in entries.values())
    return {"scope": f"first_{limit}_nt" if truncated else "full_length",
            "native_limit": limit, "entries": entries, "n": len(entries),
            "n_truncated": sum(len(entry["sequence"]) > limit for entry in entries.values()) if limit else 0,
            "result_sha256": sha256(result_path), "clean_sha256": audit["clean_sha256"],
            "audit_sha256": sha256(audit_path), "native": native}


def aggregate_split(root, split):
    loaded = {system: load_system_split(root, system, split) for system in SPECS}
    available = {system: value for system, value in loaded.items() if value is not None}
    scopes = sorted({value["scope"] for value in available.values()})
    groups = {}
    for scope in scopes:
        if scope == "EMPTY_AFTER_DELEAK":
            continue
        members = {system: value["entries"] for system, value in available.items() if value["scope"] == scope}
        matched = matched_group(members)
        n = next(iter(matched.values()))["n"]
        systems = {}
        for system, values in matched.items():
            if not n:
                systems[system] = values
                continue
            metadata = available[system]
            systems[system] = {**values, "f1_values": [values["f1"]] * 3,
                               "f1_mean": statistics.mean([values["f1"]] * 3),
                               "f1_sd": statistics.stdev([values["f1"]] * 3),
                               "result_sha256": metadata["result_sha256"],
                               "clean_sha256": metadata["clean_sha256"],
                               "audit_sha256": metadata["audit_sha256"],
                               "native_n": metadata["n"], "n_truncated": metadata["n_truncated"]}
        groups[scope] = {"status": "PASS_MATCHED" if n and len(systems) >= 2 else
                         "EMPTY_INTERSECTION" if not n else "INSUFFICIENT_COMPARATORS",
                         "n": n, "systems": systems, "pairwise": matched_pairs(members)}
    return {"available_systems": sorted(available), "missing_systems": sorted(set(SPECS) - set(available)),
            "complete_system_count": len(available), "groups": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"schema_version": 1,
              "protocol": "per-system own-training-union clean; exact name+sequence+evaluated-GT matched intersection; native truncation scopes separated; deterministic inference repeats, not training seeds",
              "ranking_eligibility": "sensitivity only until every eligible system and split is complete; unequal native scopes are never ranked together",
              "splits": {split: aggregate_split(args.root, split) for split in SPLITS}}
    complete = all(value["complete_system_count"] == len(SPECS) for value in output["splits"].values())
    output["status"] = "PASS_COMPLETE_MATCHED_SENSITIVITY" if complete else "PARTIAL_MATCHED_SENSITIVITY_NOT_LEADERBOARD"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out), "sha256": sha256(args.out),
                      "systems_per_split": {split: value["complete_system_count"]
                                            for split, value in output["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
