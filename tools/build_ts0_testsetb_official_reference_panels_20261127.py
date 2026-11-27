#!/usr/bin/env python3
"""Build TS0/TestSetB official-full and projected-subset frozen-prediction panels."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
OUT = ROOT / "artifacts/ts0_testsetb_official_reference_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-ts0-testsetb-official-reference-panels.md"
PRIOR = ROOT / "artifacts/archive_label_lineage_audit_20261127/summary.json"
J = Path("/root/rna_jepa_data/ss_data/jsonl")
BD = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
MX_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/mxfold2")
SEED = 20261127
BOOTSTRAP = 10000
SCHEMA = "ts0_testsetb.official_reference_panels.v1"
DETERMINISTIC = {"contrafold", "eternafold", "mxfold2", "rnastructure_fold", "rnastructure_maxexpect"}
SPECS = {
    "bprna_ts0": {
        "project_n": 1288, "official_n": 1305, "project_sha": "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882",
        "reference_archive": BD, "reference_prefix": "BPfold_data/bpRNA/TS0/", "reference_sha": "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403",
        "mirror_archive": MX_ROOT / "bpRNA.tar.gz", "mirror_prefix": "data/bpRNA_dataset-canonicals/TS0/",
        "mirror_sha": "69dc58fe6cfdf1ec178b1732b7da35b828dbbc7608e89d9debdde9d538937a55",
        "full_id": "bprna_ts0_official_reference_1305", "project_panel_id": "bprna_ts0_projected_reference_1288",
        "full_file": "bprna_ts0_official_1305.jsonl.gz", "project_file": "bprna_ts0_projected_1288.jsonl.gz",
        "metrics_file": "bprna_ts0_system_metrics.json", "expected_available": 21,
        "id_transform": lambda name: name,
    },
    "testsetb": {
        "project_n": 428, "official_n": 430, "project_sha": "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06",
        "reference_archive": MX_ROOT / "Rivas.tar.gz", "reference_prefix": "data/TestSetB/",
        "reference_sha": "4ec63e183ef21cdc7dc76561da5ad1543bdd53d1ec09c14d692cc0cd8da86db2",
        "mirror_archive": None, "mirror_prefix": None, "mirror_sha": None,
        "full_id": "testsetb_official_reference_430", "project_panel_id": "testsetb_projected_reference_428",
        "full_file": "testsetb_official_430.jsonl.gz", "project_file": "testsetb_projected_428.jsonl.gz",
        "metrics_file": "testsetb_system_metrics.json", "expected_available": 20,
        "id_transform": lambda name: "testsetb_" + name,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical(row) + "\n")
    os.replace(temporary, path)


def pair_set(values, length):
    result, occupied = set(), set()
    for value in values:
        i, j = map(int, value)
        if not 0 <= i < j < length or i in occupied or j in occupied:
            raise ValueError(f"invalid/conflicting pair {(i, j)} for length {length}")
        occupied.update((i, j))
        result.add((i, j))
    return result


def parse_bpseq(payload):
    bases, partners = [], []
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or int(fields[0]) != len(bases) + 1:
            raise ValueError("invalid BPSEQ")
        bases.append(fields[1].upper().replace("T", "U"))
        partners.append(int(fields[2]))
    result = set()
    for i, partner in enumerate(partners):
        if partner:
            if not 1 <= partner <= len(bases) or partners[partner - 1] != i + 1:
                raise ValueError("BPSEQ partner mismatch")
            if partner != i + 1:
                result.add(tuple(sorted((i, partner - 1))))
    return "".join(bases), pair_set(result, len(bases))


def tar_records(path, prefix):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".bpseq"):
                continue
            payload = archive.extractfile(member).read()
            sequence, pairs = parse_bpseq(payload)
            name = Path(member.name).name
            if name in result:
                raise ValueError(f"duplicate BPSEQ {name}")
            result[name] = {"seq": sequence, "pairs": pairs, "member": member.name,
                            "member_sha256": hashlib.sha256(payload).hexdigest()}
    return result


def build_panels(dataset):
    spec = SPECS[dataset]
    project_path = J / f"{dataset}.jsonl"
    if sha256(project_path) != spec["project_sha"] or sha256(spec["reference_archive"]) != spec["reference_sha"]:
        raise ValueError(f"input identity changed {dataset}")
    official = tar_records(spec["reference_archive"], spec["reference_prefix"])
    mirror = tar_records(spec["mirror_archive"], spec["mirror_prefix"]) if spec["mirror_archive"] else None
    if mirror and sha256(spec["mirror_archive"]) != spec["mirror_sha"]:
        raise ValueError("mirror SHA changed")
    if len(official) != spec["official_n"] or (mirror and official.keys() != mirror.keys()):
        raise ValueError(f"official population mismatch {dataset}")
    full = []
    for source_id in sorted(official):
        ref = official[source_id]
        if mirror:
            other = mirror[source_id]
            if (ref["seq"], ref["pairs"]) != (other["seq"], other["pairs"]):
                raise ValueError(f"reference mirror mismatch {source_id}")
        project_id = spec["id_transform"](source_id)
        core = {"name": project_id, "source_id": source_id, "seq": ref["seq"],
                "pairs": [list(pair) for pair in sorted(ref["pairs"])],
                "sequence_sha256": hashlib.sha256(ref["seq"].encode()).hexdigest(),
                "pair_set_sha256": value_sha(sorted(ref["pairs"]))}
        row = {**core, "record_sha256": value_sha(core), "reference_member": ref["member"],
               "reference_member_sha256": ref["member_sha256"],
               "reference_cross_package_status": "EXACT" if mirror else "UNCHECKED_SINGLE_PACKAGE"}
        if mirror:
            other = mirror[source_id]
            row.update({"mirror_member": other["member"], "mirror_member_sha256": other["member_sha256"]})
        else:
            row.update({"mirror_member": None, "mirror_member_sha256": None})
        full.append(row)
    project = [json.loads(line) for line in project_path.read_text().splitlines() if line.strip()]
    if len(project) != spec["project_n"] or len({row["name"] for row in project}) != spec["project_n"]:
        raise ValueError(f"project population mismatch {dataset}")
    by_name = {row["name"]: row for row in full}
    projected = []
    for project_row in project:
        name = project_row["name"]
        if name not in by_name or by_name[name]["seq"] != project_row["seq"].upper().replace("T", "U"):
            raise ValueError(f"project sequence mismatch {name}")
        projected.append(by_name[name])
    write_gzip_jsonl(SOURCE / spec["full_file"], full)
    write_gzip_jsonl(SOURCE / spec["project_file"], projected)
    return full, projected


def model_form(prediction_id, baseline_id):
    if baseline_id == "xens2":
        return "three_checkpoint_ensemble"
    if baseline_id == "bpfold":
        return "six_checkpoint_ensemble"
    if baseline_id in DETERMINISTIC:
        return "deterministic_single_predictor"
    return "single_model"


def inventory(dataset):
    registry = json.loads((ROOT / "manifests/foldnar_v1/prediction_registry.json").read_text())
    entries = []
    for row in sorted((item for item in registry["rows"] if item.get("dataset_id") == dataset), key=lambda item: item["prediction_id"]):
        available = bool(row.get("exists") and row.get("status") == "FROZEN" and row.get("prediction_schema") in {"per_sequence", "predictions"})
        entries.append({"prediction_id": row["prediction_id"], "baseline_id": row.get("baseline_id", row.get("model")),
                        "source_group": "REGISTERED", "registry_status": row.get("status"),
                        "path": row.get("path"), "sha256": row.get("sha256"), "data_path": row.get("data_path"),
                        "available": available, "model_form": model_form(row["prediction_id"], row.get("baseline_id", row.get("model"))),
                        "prediction_origin": "project_bpfold_replay" if row.get("baseline_id") == "bpfold" else "registered_frozen_prediction"})
    for seed, digest in ((1, {"bprna_ts0": "5f670b2bd64a1091bb00cbd220ecb3228310add49b413a240cc5bcd093c5792f", "testsetb": "fe3b7e261b1f38a6d61762f88f878c5f6e6bd2454733c1889aa29fe8d30fd893"}[dataset]),
                         (2, {"bprna_ts0": "77214cb915ce99d478f621e8161f01b158a68fe753d83879df84033a1a79b773", "testsetb": "588b99cebc99c5c7293d9768179074a29d748d4fa8f493df8878935d3a458f7c"}[dataset])):
        path = f"/root/rna_jepa_data/eval_decision/ow_rinalmo_r2dtr1c_b4_s{seed}_step20000/{dataset}/result.json"
        entries.append({"prediction_id": f"rinalmo_2d_s{seed}/{dataset}", "baseline_id": "r2d",
                        "source_group": "INDEPENDENT_AUDIT_ASSET", "registry_status": "UNREGISTERED_INDEPENDENT_AUDIT_ASSET",
                        "path": path, "sha256": digest, "data_path": str(J / f"{dataset}.jsonl"),
                        "available": True, "model_form": "single_model", "prediction_origin": f"additional_frozen_r2d_seed{seed}"})
    if len(entries) != SPECS[dataset]["expected_available"] + 14 or sum(row["available"] for row in entries) != SPECS[dataset]["expected_available"]:
        raise ValueError(f"prediction inventory changed {dataset}")
    return entries


def load_predictions(entry):
    document = json.loads(Path(entry["path"]).read_text())
    if document.get("data") and document["data"] != entry["data_path"]:
        raise ValueError(f"prediction data path mismatch {entry['prediction_id']}")
    dataset = entry["prediction_id"].rsplit("/", 1)[-1]
    if document.get("data_sha256") and document["data_sha256"] != SPECS[dataset]["project_sha"]:
        raise ValueError(f"prediction data SHA mismatch {entry['prediction_id']}")
    if entry["prediction_id"].startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    elif "per_sequence" in document:
        source = document["per_sequence"]
    elif "predictions" in document:
        source = document["predictions"]
    else:
        raise ValueError("NO_RECORD_LEVEL_PREDICTIONS")
    result = {}
    for row in source:
        name = row["name"]
        pairs = row.get("pred_pairs", row.get("pairs"))
        if name in result or pairs is None:
            raise ValueError(f"duplicate/missing prediction {name}")
        result[name] = {"seq": row.get("seq"), "pairs": pairs}
    return result


def prf(truth, predicted):
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    if not truth and not predicted:
        return tp, fp, fn, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1


def score_present(panel, predictions):
    totals = [0, 0, 0]
    values, records, missing = [], {}, []
    for label in panel:
        name = label["name"]
        if name not in predictions:
            missing.append(name)
            continue
        prediction = predictions[name]
        if prediction.get("seq") is not None and prediction["seq"].upper().replace("T", "U") != label["seq"]:
            raise ValueError(f"prediction sequence mismatch {name}")
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(prediction["pairs"], len(label["seq"]))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        a, b, c, p, r, f = prf(truth, predicted)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        values.append((p, r, f))
        records[name] = {"tp": a, "fp": b, "fn": c, "precision": p, "recall": r, "f1": f,
                         "truth_empty": not truth, "prediction_empty": not predicted,
                         "empty_empty": not truth and not predicted}
    n = len(values)
    tp, fp, fn = totals
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
               "truth_empty_records": sum(row["truth_empty"] for row in records.values()),
               "empty_empty_records": sum(row["empty_empty"] for row in records.values())}
    extra = sorted(set(predictions) - {row["name"] for row in panel})
    return metrics, records, missing, extra


def training_exposure(entry, dataset, baselines, prior):
    baseline = baselines.get(entry["baseline_id"], {})
    training = baseline.get("training_data", {})
    project_ids = training.get("project_training_dataset_ids", [])
    corpora = project_ids or training.get("external_training_corpora", [])
    result = {"baseline_id": entry["baseline_id"], "training_provenance_status": training.get("status", "UNCHECKED"),
              "training_corpora": corpora, "route": None, "exact_sequence_overlap": None,
              "homology_only_overlap": None, "clean_n": None, "clean_sha256": None,
              "evidence_status": "UNCHECKED_EXTERNAL_TRAINING_OVERLAP",
              "full_project_panel_exposure_status": "UNCHECKED"}
    route = "bprna_tr1c" if "bprna_tr1c" in project_ids else "bprna_tr0" if entry["baseline_id"] == "rinalmo_ft" else None
    if route:
        evidence = prior["training_overlap"][dataset][route]
        result.update({"route": route, "exact_sequence_overlap": evidence["exact_sequence_overlap_records"],
                       "homology_only_overlap": evidence["removed_homology_only"], "clean_n": evidence["clean_n"],
                       "clean_sha256": evidence["clean_sha256"], "evidence_status": "FROZEN_EXACT_AND_MMSEQS_AUDIT_RECONSTRUCTED"})
        if evidence["exact_sequence_overlap_records"]:
            result["full_project_panel_exposure_status"] = "EXACT_TRAIN_OVERLAP_PRESENT_NOT_INDEPENDENT"
        elif evidence["removed_homology_only"]:
            result["full_project_panel_exposure_status"] = "HOMOLOGY_OVERLAP_PRESENT_UNDER_FROZEN_SCREEN"
        return result
    if training.get("status") == "NOT_APPLICABLE_NOT_SEQUENCE_FITTED":
        result.update({"route": "NOT_SEQUENCE_FITTED", "evidence_status": "NOT_APPLICABLE_NOT_SEQUENCE_FITTED",
                       "full_project_panel_exposure_status": "NOT_SEQUENCE_FITTED"})
        return result
    if entry["baseline_id"] == "mxfold2":
        summary = json.loads((ROOT / "source_data/appendix_leakage/round39_mxfold2_clean_trainsetab_summary.json").read_text())
        audit_path = Path(summary["splits"][dataset]["clean_audit"])
        audit = json.loads(audit_path.read_text())
        result.update({"route": "mxfold2_trainsetab", "exact_sequence_overlap": audit["removed_exact"],
                       "homology_only_overlap": audit["removed_homology_only"], "clean_n": audit["clean_n"],
                       "clean_sha256": audit["clean_sha256"], "evidence_status": "OFFICIAL_TRAINSETAB_AUDIT_ARTIFACT; FULL_EXPOSURE_NOT_PROVEN",
                       "full_project_panel_exposure_status": "AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED"})
        return result
    if entry["baseline_id"] == "nucleicbert_frozen_official":
        summary = json.loads((ROOT / "source_data/appendix_leakage/round39_nucleicbert_frozen_clean_audit_summary.json").read_text())
        evidence = summary["per_split"][dataset]
        result.update({"route": "rnastralign_tr0_style", "exact_sequence_overlap": evidence["removed_exact"],
                       "homology_only_overlap": evidence["removed_homology_only"], "clean_n": evidence["clean_n"],
                       "clean_sha256": evidence["clean_sha256"], "evidence_status": "AUTHOR_TR0_STYLE_AUDIT_ARTIFACT; FULL_EXPOSURE_NOT_PROVEN",
                       "full_project_panel_exposure_status": "AUDITED_SUBSET_AVAILABLE_BUT_FULL_EXTERNAL_EXPOSURE_UNRESOLVED"})
    return result


def bootstrap_interval(deltas):
    rng = np.random.default_rng(SEED)
    array = np.asarray(deltas)
    values = []
    for start in range(0, BOOTSTRAP, 200):
        size = min(200, BOOTSTRAP - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        values.extend(array[indices].mean(axis=1).tolist())
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def panel_identity(panel, path, dataset_id, spec, selection):
    return {"dataset_id": dataset_id, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size, "n_sequences": len(panel),
            "id_order_sha256": value_sha([row["name"] for row in panel]),
            "empty_reference_records": sum(not row["pairs"] for row in panel),
            "selection_rule": selection,
            "reference_cross_package_status": "EXACT" if spec["mirror_archive"] else "UNCHECKED_SINGLE_PACKAGE",
            "source_archives": {"reference": {"path": str(spec["reference_archive"]), "sha256": spec["reference_sha"]},
                                "mirror": {"path": str(spec["mirror_archive"]) if spec["mirror_archive"] else None,
                                           "sha256": spec["mirror_sha"]}},
            "pair_protocol": "unfiltered official BPSEQ within the stated ID panel; no scorer-added projection"}


def public_strongest(row, panel_id):
    result = row["panels"][panel_id]
    return {"prediction_id": row["prediction_id"], "model_form": row["model_form"],
            "metrics": result["metrics"], "delta_vs_xens2": result["delta_vs_xens2"],
            "training_exposure": row["training_exposure"],
            "ranking_scope": "DESCRIPTIVE_FROZEN_PROJECTED_SUBSET_AUDIT_ONLY_NOT_MODEL_SELECTION"}


def evaluate_dataset(dataset, full, projected, prior):
    spec = SPECS[dataset]
    full_id, project_id = spec["full_id"], spec["project_panel_id"]
    baselines = {row["baseline_id"]: row for row in json.loads((ROOT / "manifests/foldnar_v1/baseline_registry.json").read_text())["rows"]}
    rows, record_rows, project_f1 = [], [], {}
    for entry in inventory(dataset):
        row = dict(entry)
        row["training_exposure"] = training_exposure(entry, dataset, baselines, prior)
        row["panels"] = {}
        if not entry["available"]:
            status = "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED" if entry["registry_status"] == "SUMMARY_ONLY" else "BLOCKED_REGISTERED_PREDICTION_MISSING"
            for panel_id, panel in ((full_id, full), (project_id, projected)):
                row["panels"][panel_id] = {"status": status, "n_panel_sequences": len(panel), "n_with_prediction": 0,
                                           "coverage": 0.0, "metrics": None, "delta_vs_xens2": None,
                                           "paired_macro_f1_delta_ci95_vs_xens2": None}
            rows.append(row)
            continue
        if not Path(entry["path"]).is_file() or sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"prediction SHA mismatch {entry['prediction_id']}")
        predictions = load_predictions(entry)
        row["n_prediction_records"] = len(predictions)
        row["sequence_join"] = "frozen source data path/SHA plus per-ID reference join; direct seq checked when present"
        for panel_id, panel in ((full_id, full), (project_id, projected)):
            metrics, records, missing, extra = score_present(panel, predictions)
            row["panels"][panel_id] = {
                "status": "PASS_FULL_COVERAGE" if not missing else "PARTIAL_COVERAGE_NOT_RANKED",
                "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing), "missing_names": missing,
                "extra_count": len(extra), "extra_names": extra, "metrics": metrics,
                "delta_vs_xens2": None, "paired_macro_f1_delta_ci95_vs_xens2": None,
                "missing_predictions_imputed_as_zero": False,
            }
            for name, values in records.items():
                record_rows.append({"dataset": dataset, "panel_id": panel_id,
                                    "prediction_id": entry["prediction_id"], "name": name, **values})
            if panel_id == project_id and not missing:
                project_f1[entry["prediction_id"]] = {name: values["f1"] for name, values in records.items()}
        rows.append(row)
    xens2_id = f"xens2/{dataset}"
    if len(project_f1) != spec["expected_available"] or xens2_id not in project_f1:
        raise ValueError(f"project full-coverage population changed {dataset}")
    names = [row["name"] for row in projected]
    xens2_values = np.asarray([project_f1[xens2_id][name] for name in names])
    reference = next(row for row in rows if row["prediction_id"] == xens2_id)["panels"][project_id]["metrics"]
    for row in rows:
        result = row["panels"][project_id]
        if result["status"] != "PASS_FULL_COVERAGE":
            continue
        metrics = result["metrics"]
        result["delta_vs_xens2"] = {key: metrics[key] - reference[key] for key in (
            "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
        values = np.asarray([project_f1[row["prediction_id"]][name] for name in names])
        deltas = values - xens2_values
        result["paired_macro_f1_delta_ci95_vs_xens2"] = {
            "observed_delta": float(deltas.mean()), "ci95": bootstrap_interval(deltas),
            "method": "paired sequence-level percentile bootstrap", "n_bootstrap": BOOTSTRAP,
            "seed": SEED, "unit": "sequence", "applicability": f"full common {dataset} projected subset coverage only"}
    eligible = [row for row in rows if row["panels"][project_id]["status"] == "PASS_FULL_COVERAGE"]
    strongest = {
        "strongest_overall": public_strongest(max(eligible, key=lambda row: row["panels"][project_id]["metrics"]["micro_f1"]), project_id),
        "strongest_ensemble": public_strongest(max((row for row in eligible if "ensemble" in row["model_form"]), key=lambda row: row["panels"][project_id]["metrics"]["micro_f1"]), project_id),
        "strongest_single_predictor": public_strongest(max((row for row in eligible if "ensemble" not in row["model_form"]), key=lambda row: row["panels"][project_id]["metrics"]["micro_f1"]), project_id),
    }
    if any(row["panels"][full_id]["status"] == "PASS_FULL_COVERAGE" for row in rows):
        raise ValueError(f"unexpected official-full prediction coverage {dataset}")
    return rows, record_rows, strongest


def coverage_gaps(dataset, full, rows):
    spec = SPECS[dataset]
    present_names = set()
    for row in rows:
        if row["available"]:
            present_names = set(row["panels"][spec["full_id"]]["missing_names"])
            break
    official_only = [row for row in full if row["name"] in present_names]
    return {"dataset": dataset, "official_n": len(full), "project_n": spec["project_n"],
            "official_missing_from_project_count": len(official_only),
            "official_missing_from_project": [{"name": row["name"], "source_id": row["source_id"],
                    "reason": "PROJECT_REJECTED_BAD_ALPHABET", "non_acgu_characters": sorted(set(row["seq"]) - set("ACGU")),
                    "sequence_sha256": row["sequence_sha256"], "pair_set_sha256": row["pair_set_sha256"],
                    "reference_member": row["reference_member"], "reference_member_sha256": row["reference_member_sha256"]}
                   for row in official_only],
            "per_prediction_official_coverage": [{"prediction_id": row["prediction_id"],
                    "status": row["panels"][spec["full_id"]]["status"],
                    "n_with_prediction": row["panels"][spec["full_id"]]["n_with_prediction"],
                    "coverage": row["panels"][spec["full_id"]]["coverage"],
                    "missing_count": row["panels"][spec["full_id"]].get("missing_count"),
                    "missing_names": row["panels"][spec["full_id"]].get("missing_names")} for row in rows]}


def system_document(dataset, full, projected, rows, strongest, prior):
    spec = SPECS[dataset]
    full_id, project_id = spec["full_id"], spec["project_panel_id"]
    return {
        "schema_version": SCHEMA, "dataset": dataset, "audit_date": "2026-11-27",
        "scope": "CPU-only frozen-prediction rescoring on official-full and projected-reference panels; no inference, training or model selection",
        "label_panels": {
            full_id: panel_identity(full, SOURCE / spec["full_file"], full_id, spec, "all official BPSEQ records"),
            project_id: panel_identity(projected, SOURCE / spec["project_file"], project_id, spec, "official reference restricted to frozen project IDs; historical project labels were projected"),
        },
        "metric_contract": {
            "pair_matching": "strict zero-based inclusive exact pairs; no additional crossing/canonical/min-loop filtering",
            "micro": "pool TP/FP/FN; zero-denominator precision or recall=0",
            "macro": "unweighted mean per-sequence P/R/F1; P=R=F1=1 only when truth and prediction are both empty",
            "ranking": "micro-F1, full coverage within the stated panel only; descriptive audit, not model selection",
            "missing_predictions": "present-only coverage accounting; never impute zero; partial coverage not ranked",
            "paired_ci": f"10000 paired sequence-level percentile bootstraps versus xens2 on projected {spec['project_n']}; seed 20261127",
        },
        "coverage_accounting": {
            "inventory_entries": len(rows), "scored_identities": sum(row["available"] for row in rows),
            "blocked_identities": sum(not row["available"] for row in rows),
            "projected_full_coverage": sum(row["panels"][project_id]["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "official_full_coverage": sum(row["panels"][full_id]["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "official_partial_not_ranked": sum(row["panels"][full_id]["status"] == "PARTIAL_COVERAGE_NOT_RANKED" for row in rows),
            "official_records_missing_from_project": spec["official_n"] - spec["project_n"],
        },
        **strongest,
        "official_full_summary": {"strongest_overall": "UNCHECKED_NO_FULL_COVERAGE_PREDICTION",
                                  "strongest_ensemble": "UNCHECKED_NO_FULL_COVERAGE_PREDICTION",
                                  "strongest_single_predictor": "UNCHECKED_NO_FULL_COVERAGE_PREDICTION"},
        "xens2_reference": next(row for row in rows if row["prediction_id"] == f"xens2/{dataset}"),
        "training_exposure_summary": prior["training_overlap"][dataset],
        "legacy_project_label_disposition": "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION; projected-reference-protocol-only; not unfiltered official protocol",
        "rows": rows, "external_test_model_selection": False, "new_training_or_inference": False, "gpu_used": False,
    }


def render_report(documents, gaps):
    lines = ["# bpRNA TS0 / TestSetB 官方全集与 projected-reference 双面板", "",
             "仅对冻结预测做 CPU 复算，不做训练、新推理、阈值调整或模型选择。项目子集结果是 `DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION` 的显式 projected-reference 协议，不与未过滤官方全集混排。缺预测不补 0，部分覆盖不排名；共享 registry 未修改。", "",
             "严格 zero-based inclusive 配对，不追加过滤；macro 是逐序列 P/R/F1 非加权平均，仅 empty/empty 时 P/R/F1=1。TS0 官方唯一 truth-empty 记录不在项目 1288 中，当前所有预测均缺失该记录，未被补 0 或计入 macro；TestSetB 无 truth-empty。", ""]
    for dataset, document in documents.items():
        spec = SPECS[dataset]
        project_id, full_id = spec["project_panel_id"], spec["full_id"]
        lines += [f"## {dataset}：projected {spec['project_n']} 子集结果", "",
                  "| Prediction identity | Model form | n | micro F1 | macro F1 | Δmicro vs xens2 | Δmacro vs xens2 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        scored = sorted((row for row in document["rows"] if row["panels"][project_id]["status"] == "PASS_FULL_COVERAGE"),
                        key=lambda row: row["panels"][project_id]["metrics"]["micro_f1"], reverse=True)
        for row in scored:
            result = row["panels"][project_id]
            metric, delta = result["metrics"], result["delta_vs_xens2"]
            lines.append(f"| `{row['prediction_id']}` | {row['model_form']} | {spec['project_n']} | {metric['micro_f1']:.10f} | {metric['macro_f1']:.10f} | {delta['micro_f1']:+.10f} | {delta['macro_f1']:+.10f} |")
        lines += ["", "### 最强单预测器与集成（描述性，不用于选模）", ""]
        for title, key in (("最强集成/总体", "strongest_ensemble"), ("最强单预测器", "strongest_single_predictor")):
            row = document[key]
            metric, delta = row["metrics"], row["delta_vs_xens2"]
            ci = next(item for item in document["rows"] if item["prediction_id"] == row["prediction_id"])["panels"][project_id]["paired_macro_f1_delta_ci95_vs_xens2"]["ci95"]
            exposure = row["training_exposure"]
            lines.append(f"- **{title}**：`{row['prediction_id']}`，micro-F1={metric['micro_f1']:.16f}，macro-F1={metric['macro_f1']:.16f}；Δmicro={delta['micro_f1']:+.16f}，Δmacro={delta['macro_f1']:+.16f}，配对 macro Δ95% CI [{ci[0]:+.16f}, {ci[1]:+.16f}]。训练暴露栏：`{exposure['full_project_panel_exposure_status']}`（route={exposure['route'] or 'UNCHECKED'}）。")
        lines += ["", f"### 官方全集 {spec['official_n']} 覆盖缺口", "",
                  f"官方比项目子集多 {spec['official_n'] - spec['project_n']} 条；所有可用预测均只有 {spec['project_n']}/{spec['official_n']}，coverage={spec['project_n'] / spec['official_n']:.10f}，状态为 `PARTIAL_COVERAGE_NOT_RANKED`。全集没有全覆盖预测，最强单/集成均为 `UNCHECKED_NO_FULL_COVERAGE_PREDICTION`。", "",
                  "官方独有 ID（均为项目 bad-alphabet 拒绝，异常字符在括号内）：" + "、".join(
                      f"`{row['name']}` ({''.join(row['non_acgu_characters'])})" for row in gaps[dataset]["official_missing_from_project"]) + "。", "",
                  "逐预测 missing ID 全表见 `official_coverage_gaps.json`；summary-only/blocked 的 metrics=null，不补 0。", "",
                  "### 训练暴露分栏", "",
                  "以下为数据集级路线事实；TR1 homology 未检查。模型相关暴露另按 prediction identity 列出，不能用标签独立替代训练独立。", "",
                  "| 路线 | 精确序列重叠 | homology-only | clean n | clean SHA 重建 |",
                  "|---|---:|---:|---:|---|"]
        for route, item in document["training_exposure_summary"].items():
            lines.append(f"| {route} | {item['exact_sequence_overlap_records']} | {item.get('removed_homology_only', 'UNCHECKED')} | {item.get('clean_n', 'UNCHECKED')} | {item.get('clean_subset_reconstruction', 'UNCHECKED')} |")
        lines += ["", "| Prediction identity | 训练语料/路线 | 精确重叠 | homology-only | clean n | 全项目面板暴露判定 |",
                  "|---|---|---:|---:|---:|---|"]
        for row in scored:
            exposure = row["training_exposure"]
            fmt = lambda value: "UNCHECKED" if value is None else value
            corpora = exposure["route"] or "+".join(exposure["training_corpora"]) or "UNCHECKED"
            lines.append(f"| `{row['prediction_id']}` | {corpora} | {fmt(exposure['exact_sequence_overlap'])} | {fmt(exposure['homology_only_overlap'])} | {fmt(exposure['clean_n'])} | `{exposure['full_project_panel_exposure_status']}` |")
        lines += ["", "外部模型的未核实训练语料保持 `UNCHECKED`；MXfold2/NucleicBERT 的审计子集证据不升级为完整训练暴露排除证明。逐模型 provenance 原文和 clean SHA 见 `system_metrics.json`。", ""]
    return "\n".join(lines)


def main():
    registry_paths = [ROOT / f"manifests/foldnar_v1/{name}.json" for name in ("data_registry", "prediction_registry", "baseline_registry")]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in registry_paths}
    prior = json.loads(PRIOR.read_text())
    documents, gaps, all_records = {}, {}, []
    for dataset in SPECS:
        full, projected = build_panels(dataset)
        rows, record_rows, strongest = evaluate_dataset(dataset, full, projected, prior)
        document = system_document(dataset, full, projected, rows, strongest, prior)
        documents[dataset] = document
        all_records.extend(record_rows)
        gaps[dataset] = coverage_gaps(dataset, full, rows)
        write_json(OUT / SPECS[dataset]["metrics_file"], document)
    write_gzip_jsonl(OUT / "per_record_metrics.jsonl.gz", all_records)
    write_json(OUT / "official_coverage_gaps.json", {"schema_version": SCHEMA + ".coverage_gaps", "datasets": gaps})
    recommendation = {
        "schema_version": SCHEMA + ".registry_recommendation", "recommendation_only": True,
        "apply_by_unified_process_only": True, "shared_registry_modified": False,
        "datasets": {dataset: {
            "legacy_project_labels": {"label_source": "DERIVED_FROM_OFFICIAL_REFERENCE_PROJECTION",
                "disposition": "projected-reference-protocol-only", "unfiltered_official_protocol_eligible": False},
            "suggested_data_rows": list(documents[dataset]["label_panels"].values()),
            "suggested_prediction_overlay": {"path": f"artifacts/ts0_testsetb_official_reference_20261127/{SPECS[dataset]['metrics_file']}",
                "sha256": sha256(OUT / SPECS[dataset]["metrics_file"]), "rows": documents[dataset]["rows"]},
            "training_exposure_overlay": documents[dataset]["training_exposure_summary"],
            "official_full_summary": documents[dataset]["official_full_summary"],
        } for dataset in SPECS},
        "coverage_gaps_path": "artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json",
    }
    write_json(OUT / "registry_recommendation.json", recommendation)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(documents, gaps), encoding="utf-8")
    source_manifest = {
        "schema_version": SCHEMA + ".source_manifest",
        "datasets": {dataset: {"project": {"path": str(J / f"{dataset}.jsonl"), "sha256": SPECS[dataset]["project_sha"]},
            "reference": {"path": str(SPECS[dataset]["reference_archive"]), "sha256": SPECS[dataset]["reference_sha"]},
            "mirror": {"path": str(SPECS[dataset]["mirror_archive"]) if SPECS[dataset]["mirror_archive"] else None,
                       "sha256": SPECS[dataset]["mirror_sha"]},
            "predictions": [{"prediction_id": row["prediction_id"], "path": row["path"], "sha256": row["sha256"],
                             "available": row["available"]} for row in documents[dataset]["rows"]]} for dataset in SPECS},
        "shared_registry_sha256_before_after": before, "shared_registry_modified": False,
        "new_training_or_inference": False, "gpu_used": False, "external_test_model_selection": False,
    }
    write_json(SOURCE / "source_manifest.json", source_manifest)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in registry_paths}
    if before != after:
        raise ValueError("shared registry modified")
    output_files = [SOURCE / spec[name] for spec in SPECS.values() for name in ("full_file", "project_file")]
    output_files += [SOURCE / "source_manifest.json", OUT / "per_record_metrics.jsonl.gz",
                     OUT / "official_coverage_gaps.json", OUT / "registry_recommendation.json", REPORT]
    output_files += [OUT / spec["metrics_file"] for spec in SPECS.values()]
    write_json(OUT / "build_receipt.json", {"schema_version": SCHEMA + ".build_receipt", "status": "PASS",
        "per_record_metric_rows": len(all_records), "prediction_identities": sum(len(doc["rows"]) for doc in documents.values()),
        "outputs": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files}})
    print(json.dumps({"status": "PASS", "per_record_metric_rows": len(all_records),
                      "datasets": {dataset: {"coverage": doc["coverage_accounting"],
                      "strongest_ensemble": doc["strongest_ensemble"]["prediction_id"],
                      "strongest_single": doc["strongest_single_predictor"]["prediction_id"]} for dataset, doc in documents.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
