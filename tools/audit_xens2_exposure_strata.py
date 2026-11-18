import gzip
import hashlib
import json
from pathlib import Path
from statistics import fmean

ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
AUDIT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(sequence):
    return sequence.strip().upper().replace("T", "U")


def metrics(rows):
    tp = sum(round(row["f1"] * (row["n_pred"] + row["n_gt"]) / 2) for row in rows)
    n_pred = sum(row["n_pred"] for row in rows)
    n_gt = sum(row["n_gt"] for row in rows)
    fp = n_pred - tp
    fn = n_gt - tp
    return {"n": len(rows), "macro_f1": fmean(row["f1"] for row in rows) if rows else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "n_pred": n_pred, "n_gt": n_gt}


def history_differences(rows, older):
    return sum({key: value for key, value in row.items() if key not in ("query_id", "homology_detected")} != older[row["name"]] for row in rows)


def group_metrics(rows, predicate):
    return {"detected": metrics([row for row in rows if predicate(row["name"])]),
            "not_detected": metrics([row for row in rows if not predicate(row["name"])])}


def main():
    gate = json.loads((ART / "manifests/p0/gate_decision.json").read_text())
    prediction_path = ART / "artifacts/p0/xens2_bprna_new_replay/bprna_new.json"
    prediction = json.loads(prediction_path.read_text())
    rows = prediction["per_sequence"]
    if prediction["n_sequences"] != len(rows) or prediction["n_skipped"] != 0:
        raise AssertionError("incomplete prediction set")
    checkpoint_hashes = {"plana_s0_sha256": "plana_ckpt", "plana_s1_sha256": "plana_ckpt2", "r2d_s0_sha256": "r2d_ckpt"}
    for key, path_key in checkpoint_hashes.items():
        if sha256(prediction[path_key]) != gate["original_xens2"][key]:
            raise AssertionError(f"checkpoint identity mismatch: {path_key}")
    if sha256(DATA) != gate["original_xens2"]["bprna_new_sha256"]:
        raise AssertionError("bpRNA-new data identity changed")
    names = [row["name"] for row in rows]
    dataset = [json.loads(line) for line in DATA.open() if line.strip()]
    if names != [row["name"] for row in dataset] or len(set(names)) != len(names):
        raise AssertionError("query name/order mismatch")
    sequence_hashes = {row["name"]: hashlib.sha256(normalize(row.get("seq", row.get("sequence"))).encode()).hexdigest() for row in dataset}
    query_by_hash = {}
    query_source_sets = {}
    with gzip.open(AUDIT / "homology_round56/query_index.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            query_by_hash[row["sequence_sha256"]] = row["query_id"]
            query_source_sets[row["query_id"]] = row["source_sets"]
    homology = {}
    with gzip.open(AUDIT / "homology_round56/query_homology_union.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            homology[row["query_id"]] = row
    for name, sequence_hash in sequence_hashes.items():
        query_id = query_by_hash.get(sequence_hash)
        if query_id not in homology:
            raise AssertionError(f"missing query identity: {name}")
        if "bprna_new" not in query_source_sets[query_id]:
            raise AssertionError(f"query set mismatch: {name}")
        rows_by_name = next(row for row in rows if row["name"] == name)
        rows_by_name["query_id"] = query_id
        rows_by_name["homology_detected"] = bool(homology[query_id]["detected_sources"])
    initial = json.loads((AUDIT / "exact_overlap.json").read_text())
    ensembl = json.loads((AUDIT / "ensembl_exact_overlap.json").read_text())
    exact_names = set()
    for source, report in (("rnacentral_22_active", initial["sources"]["rnacentral_22_active"]),
                           ("rfam_14_9", initial["sources"]["rfam_14_9"]),
                           ("ensembl_109_and_bacteria_56", ensembl["sources"]["ensembl_109_and_bacteria_56"])):
        expected = report["queries"]["bprna_new"]["exact_overlap_unique"]
        hits = set(report["queries"]["bprna_new"]["overlap_names"])
        if len(hits) != expected:
            raise AssertionError(f"exact identity mismatch: {source}")
        exact_names.update(hits)
    if len(exact_names) != 5381:
        raise AssertionError("exact union count mismatch")
    full = metrics(rows)
    stored = prediction["pair_level"]
    if abs(full["macro_f1"] - stored["macro"]["f1"]) > 1e-12:
        raise AssertionError("macro-F1 independent recomputation failed")
    for key in ("f1", "precision", "recall", "tp", "fp", "fn"):
        if abs(full["micro_f1" if key == "f1" else key] - stored["micro"][key]) > 1e-12:
            raise AssertionError(f"micro metric mismatch: {key}")
    exact = group_metrics(rows, lambda name: name in exact_names)
    homology_by_name = {row["name"]: row["homology_detected"] for row in rows}
    homologous = group_metrics(rows, lambda name: homology_by_name[name])
    alternative_path = Path("/root/rna_jepa_data/eval_decision_ref/xens2_bprna_new_w0.7/result.json")
    alternative = json.loads(alternative_path.read_text())
    older = {row["name"]: row for row in alternative["per_sequence"]}
    differences = history_differences(rows, older)
    report = {"status": "PASS_FROZEN_EXPOSURE_STRATIFIED_WITH_NO_HOMOLOGY_UNKNOWN_COMPARISON",
              "scientific_scope": "read-only exposure stratification of verified original xens2 predictions; no unseen-family or clean accuracy claim and no new training",
              "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
              "data": str(DATA), "data_sha256": sha256(DATA),
              "checkpoint_sha256": {key: gate["original_xens2"][key] for key in checkpoint_hashes},
              "weight": prediction["w_plana"], "plana_s1_weight": prediction["plana_s1_weight"],
              "n_sequences": len(rows), "n_skipped": prediction["n_skipped"],
              "independent_full_metrics": full,
              "exact_overlap_strata": exact,
              "homology_strata": homologous,
              "primary_comparison": {"available": False,
                                     "reason": "all 5388 bpRNA-new queries have a detected >=80% homology match in at least one known source; no homology-not-detected comparison arm",
                                     "exact_not_detected_n": exact["not_detected"]["n"],
                                     "exact_not_detected_interpretation": "unknown, not clean; no significance test"},
              "statistical_test": None,
              "alternative_history_not_used": {"path": str(alternative_path), "sha256": sha256(alternative_path),
                                               "macro_f1": alternative["pair_level"]["macro"]["f1"],
                                               "full_record_differences_from_p0": differences,
                                               "reason": "older result is not mixed with the P0-verified historical/replay identity"},
              "limitations": ["known-source homology is not complete historical pretraining exposure",
                              "nt snapshot is not pinned", "not-detected is unknown, not clean",
                              "no group is selected by outcome and no threshold is changed"]}
    output = ART / "artifacts/v2/exposure_stratified/xens2_bprna_new.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "full": full, "exact": exact,
                      "homology": homologous, "alternative_differences": differences,
                      "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
