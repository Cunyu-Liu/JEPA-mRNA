#!/usr/bin/env python3
"""Independent field-by-field reconstruction of the TS0 coverage recovery assets."""
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/ts0_official_coverage_recovery_20261127"
PANEL_ROOT = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
OUT = ROOT / "artifacts/ts0_official_coverage_recovery_20261127"
RELEASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
BENCH = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
SYSTEMS = ("vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner", "contrafold", "eternafold", "rnastructure_fold", "rnastructure_maxexpect")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pairs(values, length):
    answer, endpoints = set(), set()
    for value in values:
        i, j = map(int, value)
        require(0 <= i < j < length and i not in endpoints and j not in endpoints, "pair bounds/conflict")
        endpoints.update((i, j))
        answer.add((i, j))
    return answer


def bpseq(payload):
    bases, partners = [], []
    for raw in payload.decode("utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split()
        require(int(fields[0]) == len(bases) + 1, "BPSEQ index")
        bases.append(fields[1].upper().replace("T", "U"))
        partners.append(int(fields[2]))
    answer = set()
    for index, partner in enumerate(partners):
        if partner:
            require(1 <= partner <= len(bases) and partners[partner - 1] == index + 1, "BPSEQ partner")
            if partner != index + 1:
                answer.add(tuple(sorted((index, partner - 1))))
    return "".join(bases), pairs(answer, len(bases))


def raw_maps(documents):
    answer = {}
    for system in SYSTEMS[:4]:
        answer[system] = {row["name"]: row["pred_pairs"] for row in documents["vienna"]["results"][system]["per_sequence"]}
    for system in ("contrafold", "eternafold"):
        answer[system] = {row["name"]: row["pred_pairs"] for row in documents[system]["per_sequence"]}
    for system, key in (("rnastructure_fold", "fold"), ("rnastructure_maxexpect", "maxexpect")):
        answer[system] = {row["name"]: row["pred_pairs"] for row in documents["rnastructure"]["results"][key]["per_sequence"]}
    return answer


def original_coverage():
    manifest = load(PANEL_ROOT / "source_manifest.json")
    answer = {}
    for entry in manifest["datasets"]["bprna_ts0"]["predictions"]:
        if not entry["available"] or "ts0_official_coverage_recovery_20261127" in entry["path"]:
            continue
        require(sha(entry["path"]) == entry["sha256"], f"original raw SHA {entry['prediction_id']}")
        document = load(entry["path"])
        if entry["prediction_id"].startswith("rnastructure_fold/"):
            source = document["results"]["fold"]["per_sequence"]
        elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
            source = document["results"]["maxexpect"]["per_sequence"]
        else:
            source = document.get("per_sequence", document.get("predictions"))
        answer[entry["prediction_id"]] = {row["name"] for row in source}
    return answer


def main():
    manifest = load(SOURCE / "recovery_manifest.json")
    require(load(OUT / "recovery_audit.json") == manifest, "recovery audit copy")
    require(manifest["status"] == "PASS_7_FULL1305_IDENTITIES_2_PARTIAL", "recovery status")
    full = rows(PANEL_ROOT / "bprna_ts0_official_1305.jsonl.gz")
    projected = rows(PANEL_ROOT / "bprna_ts0_projected_1288.jsonl.gz")
    require(len(full) == 1305 and len(projected) == 1288, "panel sizes")
    require(sha(PANEL_ROOT / "bprna_ts0_official_1305.jsonl.gz") == manifest["official_panel"]["sha256"], "full panel SHA")
    by_name = {row["name"]: row for row in full}
    project_names = {row["name"] for row in projected}
    missing = [row for row in full if row["name"] not in project_names]
    require(len(missing) == 17 and sum(not row["pairs"] for row in missing) == 1, "missing population")
    documents = {}
    for family, item in manifest["raw_sources"].items():
        require(sha(item["path"]) == item["sha256"], f"raw SHA {family}")
        documents[family] = load(item["path"])
        require(documents[family]["data_sha256"] == "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882", f"raw data SHA {family}")
    raw = raw_maps(documents)
    for system in SYSTEMS:
        require(set(raw[system]) == project_names, f"raw coverage {system}")
    supplement = load(SOURCE / "missing17_cpu_predictions.json")
    require(sha(SOURCE / "missing17_cpu_predictions.json") == manifest["supplement"]["sha256"], "supplement SHA")
    require(supplement["device"] == "CPU_ONLY" and not supplement["gpu_used"] and supplement["cuda_visible_devices"] == "", "CPU boundary")
    require(supplement["missing_records"] == 17 and supplement["deterministic_repeats"] == 3, "supplement scope")
    for system in SYSTEMS:
        result = supplement["systems"][system]
        identity = [[row["name"], row["pred_pairs"]] for row in result["per_sequence"]]
        require(result["repeat_prediction_sha256"] == [value_sha(identity)] * 3, f"repeat identity {system}")
        require(len(supplement["frozen_replay"][system]) == 6 and all(row["pairs_exact"] for row in supplement["frozen_replay"][system]), f"replay {system}")
        covered = {row["name"] for row in result["per_sequence"]}
        blocked = {row["name"] for row in result.get("blocked_records", [])}
        require(covered.isdisjoint(blocked) and covered | blocked == {row["name"] for row in missing}, f"supplement IDs {system}")
        expected_blocked = {"bpRNA_RFAM_10644.bpseq", "bpRNA_RFAM_18822.bpseq", "bpRNA_RFAM_18935.bpseq"} if system.startswith("rnastructure_") else set()
        require(blocked == expected_blocked, f"blocked IDs {system}")
        for row in result["per_sequence"]:
            ref = by_name[row["name"]]
            require(row["seq"] == ref["seq"] and row["sequence_sha256"] == ref["sequence_sha256"], f"supplement sequence {system}")
            pairs(row["pred_pairs"], len(row["seq"]))
    require(sha(RELEASE) == manifest["release"]["sha256"], "release SHA")
    release, release_members = {}, {}
    with tarfile.open(RELEASE, "r:gz") as archive:
        members = [member for member in archive if member.isfile() and member.name.startswith("BPfold_test_results/bpRNA/") and member.name.endswith(".bpseq")]
        require(len(members) == 1305, "release population")
        for member in members:
            name = Path(member.name).name
            payload = archive.extractfile(member).read()
            sequence, predicted = bpseq(payload)
            require(name in by_name and sequence == by_name[name]["seq"], f"release identity {name}")
            release[name] = sorted(predicted)
            release_members[name] = {"covered": True, "member": member.name, "member_sha256": hashlib.sha256(payload).hexdigest()}
    require(set(release) == set(by_name), "release IDs")
    expected = {"bpfold_official_release/bprna_ts0": release}
    for system in SYSTEMS:
        prediction_id = f"{system}/bprna_ts0" if system in {"contrafold", "eternafold", "rnastructure_fold", "rnastructure_maxexpect"} else f"{system}/bprna_ts0/recovered_official17"
        expected[prediction_id] = dict(raw[system])
        expected[prediction_id].update({row["name"]: row["pred_pairs"] for row in supplement["systems"][system]["per_sequence"]})
    entries = {entry["prediction_id"]: entry for entry in manifest["prediction_entries"]}
    require(set(entries) == set(expected), "normalized identity set")
    for prediction_id, predictions in expected.items():
        entry = entries[prediction_id]
        path = ROOT / entry["path"]
        require(sha(path) == entry["sha256"], f"normalized SHA {prediction_id}")
        document = load(path)
        expected_rows = [{"name": ref["name"], "seq": ref["seq"], "sequence_sha256": ref["sequence_sha256"],
                          "pred_pairs": [list(pair) for pair in sorted(pairs(predictions[ref["name"]], len(ref["seq"])))]}
                         for ref in full if ref["name"] in predictions]
        expected_document = {"schema_version": "ts0.official_coverage_recovery.prediction.v1", "prediction_id": prediction_id,
                             "dataset_id": "bprna_ts0", "data": manifest["official_panel"]["path"],
                             "data_sha256": manifest["official_panel"]["sha256"], "n_sequences": len(expected_rows),
                             "prediction_origin": entry["prediction_origin"], "per_sequence": expected_rows}
        require(document == expected_document, f"normalized document {prediction_id}")
        require(len(expected_rows) == entry["n_prediction_records"], f"normalized n {prediction_id}")
    full_ids = sorted(pid for pid, values in expected.items() if len(values) == 1305)
    partial_ids = sorted(pid for pid, values in expected.items() if len(values) < 1305)
    require(sorted(manifest["new_full_coverage_identities"]) == full_ids and len(full_ids) == 7, "full identity list")
    require(sorted(manifest["recovered_partial_identities"]) == partial_ids and len(partial_ids) == 2, "partial identity list")
    original = original_coverage()
    audit = {row["name"]: row for row in manifest["per_id_audit"]}
    require(set(audit) == {row["name"] for row in missing}, "per-ID audit population")
    for ref in missing:
        name = ref["name"]
        row = audit[name]
        require(row["sequence_sha256"] == ref["sequence_sha256"] and row["pair_set_sha256"] == ref["pair_set_sha256"], f"audit hashes {name}")
        require(row["truth_empty"] == (not ref["pairs"]), f"audit truth empty {name}")
        require(row["existing_frozen_coverage"] == sorted(pid for pid, names in original.items() if name in names) == [], f"original coverage {name}")
        require(row["bpfold_official_release"] == release_members[name], f"release audit {name}")
        require(row["final_full_coverage_identities"] == [pid for pid in expected if name in expected[pid]], f"final coverage {name}")
        require(row["blocked_recovered_identities"] == [pid for pid in expected if name not in expected[pid]], f"final blocked {name}")
    files = [ROOT / "tools/recover_ts0_official_coverage_20261127.py", ROOT / "tools/verify_ts0_official_coverage_recovery_20261127.py",
             ROOT / "tests/test_ts0_official_coverage_recovery_20261127.py", ROOT / "docs/plans/2026-11-27-ts0-official-coverage-recovery.md",
             SOURCE / "recovery_manifest.json", SOURCE / "missing17_cpu_predictions.json", OUT / "recovery_audit.json"]
    files += [ROOT / entry["path"] for entry in manifest["prediction_entries"]]
    verification = {
        "schema_version": "ts0.official_coverage_recovery.independent_verification.round2.v1",
        "status": "PASS_INDEPENDENT_RECOVERY_RECONSTRUCTION", "generator_imported": False,
        "official_release_records_verified": 1305, "official_only_ids_verified": 17,
        "normalized_documents_compared_field_by_field": 9, "full1305_identities": full_ids,
        "partial1302_identities": partial_ids, "truth_empty_name": "bpRNA_SRP_192.bpseq",
        "projected_records_rerun": 0, "gpu_used": False, "external_test_model_selection": False,
        "deliverables": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification_round2.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "full1305": len(full_ids), "partial1302": len(partial_ids),
                      "normalized_documents": 9, "release_records": 1305}, sort_keys=True))


if __name__ == "__main__":
    main()
