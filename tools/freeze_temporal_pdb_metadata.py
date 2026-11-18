import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

from audit_temporal_pdb_availability import query

GRAPHQL = "https://data.rcsb.org/graphql"
ENTITY_QUERY = """query Entities($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    rcsb_polymer_entity_container_identifiers { entry_id entity_id auth_asym_ids asym_ids }
    entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
  }
}"""
ENTRY_QUERY = """query Entries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    exptl { method }
    rcsb_entry_info { resolution_combined }
  }
}"""


def post(document, variables):
    body = json.dumps({"query": document, "variables": variables}).encode()
    request = urllib.request.Request(GRAPHQL, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"RCSB GraphQL error: {error.code}: {error.read().decode(errors='replace')}") from error
    result = json.loads(raw)
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"]))
    return result, hashlib.sha256(raw).hexdigest()


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2025-07-01")
    parser.add_argument("--availability", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    availability_path = Path(args.availability)
    availability = json.loads(availability_path.read_text())
    assert availability["status"] == "TEMPORAL_ASSET_AVAILABILITY_ONLY_QA_NOT_PASSED"
    assert availability["cutoff_initial_release_date"] == args.cutoff
    expected = availability["candidate_rna_entities"]
    result = query(args.cutoff, "polymer_entity")
    payload = result["request"]
    payload["request_options"] = {"return_all_hits": True, "results_content_type": ["experimental"]}
    request = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        search_raw = response.read()
    search = json.loads(search_raw)
    identifiers = [row["identifier"] for row in search.get("result_set", [])]
    assert search["total_count"] == expected == len(identifiers) == len(set(identifiers))
    entities = {}
    response_hashes = []
    for batch in chunks(identifiers, 100):
        result, digest = post(ENTITY_QUERY, {"ids": batch})
        response_hashes.append(digest)
        for row in result["data"]["polymer_entities"]:
            assert row["rcsb_id"] not in entities
            entities[row["rcsb_id"]] = row
    entry_ids = sorted({identifier.split("_")[0] for identifier in identifiers})
    entries = {}
    for batch in chunks(entry_ids, 100):
        result, digest = post(ENTRY_QUERY, {"ids": batch})
        response_hashes.append(digest)
        for row in result["data"]["entries"]:
            assert row["rcsb_id"] not in entries
            entries[row["rcsb_id"]] = row
    assert set(entities) == set(identifiers) and len(entries) == len(entry_ids)
    records = []
    for identifier in identifiers:
        entity = entities[identifier]
        ids = entity["rcsb_polymer_entity_container_identifiers"]
        polymer = entity["entity_poly"]
        entry = entries[ids["entry_id"]]
        assert polymer["rcsb_entity_polymer_type"] == "RNA"
        assert entry["rcsb_accession_info"]["initial_release_date"] >= args.cutoff + "T00:00:00Z"
        methods = [item["method"] for item in (entry["exptl"] or []) if item.get("method")]
        assert polymer["pdbx_seq_one_letter_code_can"]
        assert polymer["rcsb_sample_sequence_length"] == len(polymer["pdbx_seq_one_letter_code_can"])
        records.append({"identifier": identifier, "entry_id": ids["entry_id"], "entity_id": ids["entity_id"],
                        "auth_asym_ids": ids["auth_asym_ids"], "asym_ids": ids["asym_ids"],
                        "initial_release_date": entry["rcsb_accession_info"]["initial_release_date"],
                        "method": "; ".join(methods) if methods else "UNKNOWN",
                        "resolution_combined": entry["rcsb_entry_info"]["resolution_combined"],
                        "length": polymer["rcsb_sample_sequence_length"], "sequence": polymer["pdbx_seq_one_letter_code_can"]})
    lengths = [r["length"] for r in records]
    report = {
        "status": "PASS_TEMPORAL_METADATA_FREEZE_LABEL_AND_INDEPENDENCE_QA_PENDING",
        "availability_sha256": hashlib.sha256(availability_path.read_bytes()).hexdigest(),
        "cutoff": args.cutoff, "search_source": "https://search.rcsb.org/rcsbsearch/v2/query", "metadata_source": GRAPHQL,
        "query": payload, "search_response_sha256": hashlib.sha256(search_raw).hexdigest(),
        "graphql_response_sha256": response_hashes,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_rna_entities": len(records), "n_distinct_entries": len({r["entry_id"] for r in records}),
        "method_unknown": sum(r["method"] == "UNKNOWN" for r in records),
        "length_16_to_8192": sum(16 <= length <= 8192 for length in lengths),
        "length_16_to_1022": sum(16 <= length <= 1022 for length in lengths),
        "sequence_is_canonical_polymer_not_coordinate_sequence": True, "records": records,
        "limitations": [
            "Canonical polymer strings are not observed coordinate sequences and are not pairing labels",
            "Candidate counts and length strata do not establish independence or label quality",
            "Time cutoff is fixed without model outcomes; it does not resolve the unknown historical nt snapshot",
            "mmCIF hashes, sequence-coordinate mapping, missing residues, pair extraction and homology gates are not yet passed",
            "No model inference, threshold choice or new training has been started",
        ],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
