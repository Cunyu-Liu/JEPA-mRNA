#!/usr/bin/env python3
"""Resolve remaining RNAformer identities using only RCSB metadata and source text."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_metadata_resolution.v1"
CUTOFF = "2024-01-19"
TARGET_STATUSES = {"AMBIGUOUS_EXACT_CHAIN_MATCHES_UNCHECKED", "NO_EXACT_CHAIN_MATCH_UNCHECKED"}
ENTITY_QUERY = """query Entities($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    rcsb_polymer_entity_container_identifiers { entry_id entity_id auth_asym_ids asym_ids }
    entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
    rcsb_polymer_entity { pdbx_description }
    rcsb_entity_source_organism { scientific_name ncbi_taxonomy_id }
  }
}"""
ENTRY_QUERY = """query Entries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
  }
}"""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def family_matches(source_family: str, description: str) -> bool:
    family = normalize_text(source_family)
    return bool(family and family != "unknown" and family in normalize_text(description))


def post_graphql(url: str, query: str, ids: list[str]) -> tuple[dict, str]:
    payload = {"query": query, "variables": {"ids": ids}}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    document = json.loads(raw)
    if document.get("errors"):
        raise RuntimeError(document["errors"])
    return document, sha256_bytes(raw)


def freeze_metadata(entity_ids: list[str], entry_ids: list[str], output: Path, url: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entities, entries = {}, {}
    for kind, ids, query in (("entities", entity_ids, ENTITY_QUERY), ("entries", entry_ids, ENTRY_QUERY)):
        (output / kind).mkdir(parents=True, exist_ok=True)
        for batch_index, start in enumerate(range(0, len(ids), 100)):
            batch = ids[start:start + 100]
            path = output / kind / f"batch_{batch_index:04d}.json"
            if path.exists():
                document = json.loads(path.read_text())
            else:
                response, digest = post_graphql(url, query, batch)
                document = {"source": url, "request_ids": batch, "response": response, "response_sha256": digest,
                            "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                write_json(path, document)
            rows = document["response"]["data"]["polymer_entities" if kind == "entities" else "entries"]
            assert {row["rcsb_id"] for row in rows} == set(batch)
            for row in rows:
                if kind == "entities":
                    entities[row["rcsb_id"]] = row
                else:
                    entries[row["rcsb_id"]] = row
    return {"entities": entities, "entries": entries}


def candidate_chains(entity: dict, sequence: str, family: str, entry: dict) -> tuple[list[dict], dict]:
    polymer = entity["entity_poly"]
    full = polymer["pdbx_seq_one_letter_code_can"].replace("\n", "").replace(" ", "").upper().replace("T", "U")
    ids = entity["rcsb_polymer_entity_container_identifiers"]
    release = entry["rcsb_accession_info"]["initial_release_date"]
    description = entity["rcsb_polymer_entity"]["pdbx_description"] or ""
    stages = {
        "rna_type": polymer["rcsb_entity_polymer_type"] == "RNA",
        "full_polymer_exact": full == sequence,
        "pre_or_at_rnaformer_release": release <= CUTOFF + "T23:59:59Z",
        "family_text": family_matches(family, description) if normalize_text(family) != "unknown" else None,
    }
    if not all(stages[key] for key in ("rna_type", "full_polymer_exact", "pre_or_at_rnaformer_release")):
        return [], {"stages": stages, "description": description, "release": release}
    if stages["family_text"] is False:
        return [], {"stages": stages, "description": description, "release": release}
    chains = [{"pdb_id": ids["entry_id"], "entity_id": ids["entity_id"], "auth_chain": auth_chain,
               "asym_ids": ids["asym_ids"], "description": description, "initial_release_date": release,
               "source_organisms": entity.get("rcsb_entity_source_organism") or [],
               "match_types": ["FULL_POLYMER_EXACT"]} for auth_chain in ids["auth_asym_ids"]]
    return chains, {"stages": stages, "description": description, "release": release}


def command_resolve(args) -> None:
    unresolved = [json.loads(line) for line in (args.unresolved).open() if line.strip()]
    targets = [row for row in unresolved if row["rebuild_status"] in TARGET_STATUSES]
    assert len(targets) == 87
    entity_ids = sorted({entity for row in targets for entity in row["search_candidate_entity_ids"]})
    first = freeze_metadata(entity_ids, [], args.output / "metadata", args.graphql)
    entities = first["entities"]
    entry_ids = sorted({entity["rcsb_polymer_entity_container_identifiers"]["entry_id"] for entity in entities.values()})
    second = freeze_metadata([], entry_ids, args.output / "metadata", args.graphql)
    entries = second["entries"]
    resolved = []
    for row in targets:
        candidates, evidence = [], {}
        for entity_id in row["search_candidate_entity_ids"]:
            entity = entities[entity_id]
            entry = entries[entity["rcsb_polymer_entity_container_identifiers"]["entry_id"]]
            chains, audit = candidate_chains(entity, row["sequence"], row["source_family"], entry)
            evidence[entity_id] = audit
            candidates.extend(chains)
        candidates.sort(key=lambda item: (item["pdb_id"], int(item["entity_id"]), item["auth_chain"]))
        if len(candidates) == 1:
            status = "NEW_UNIQUE_METADATA_CHAIN"
            selected = candidates[0]
        elif candidates:
            status = "METADATA_AMBIGUOUS_UNCHECKED"
            selected = None
        else:
            status = "NO_METADATA_ELIGIBLE_CHAIN_UNCHECKED"
            selected = None
        resolved.append({**row, "metadata_status": status, "metadata_candidates": candidates,
                         "selected_identity": selected, "metadata_evidence": evidence,
                         "identity_policy": "full-polymer exact + RNA + pre-cutoff + optional original family text; no pairs or date ranking",
                         "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"})
    document = {"schema_version": SCHEMA, "cutoff": CUTOFF,
                "cutoff_scope": "hard exclusion after official RNAformer test_sets.plk Last-Modified date; never select earliest candidate",
                "n_records": len(resolved), "status_counts": dict(collections.Counter(row["metadata_status"] for row in resolved)),
                "n_candidate_entities": len(entity_ids), "n_candidate_entries": len(entry_ids),
                "records": resolved, "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "metadata_resolution.json", document)
    print(json.dumps({"status": "PASS_METADATA_RESOLUTION_WITH_UNCHECKED_GAPS",
                      "status_counts": document["status_counts"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphql", default="https://data.rcsb.org/graphql")
    parser.set_defaults(function=command_resolve)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
