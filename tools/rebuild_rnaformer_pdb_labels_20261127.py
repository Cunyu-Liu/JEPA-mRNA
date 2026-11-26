#!/usr/bin/env python3
"""Rebuild RNAformer PDB labels from exact RCSB/mmCIF sequence matches and DSSR."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_rcsb_rebuild.v1"
PANELS = ("pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard")
PLK_SHA256 = "12000d35e6a0501838587a8eabb47e75572f56da8460b3ee98c51836e8aa9181"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_plk(path: Path) -> dict:
    import pandas as pd
    assert sha256_file(path) == PLK_SHA256
    shim = types.ModuleType("pandas.core.indexes.numeric")
    for name in ("Int64Index", "UInt64Index", "Float64Index", "NumericIndex"):
        setattr(shim, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = shim
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_records(path: Path) -> list[dict]:
    frames = load_plk(path)
    records = []
    for panel in PANELS:
        for index, row in frames[panel].iterrows():
            sequence = "".join(row["sequence"]).upper().replace("T", "U")
            pairs = sorted(tuple(sorted((int(a), int(b)))) for a, b in zip(row["pos1id"], row["pos2id"]))
            labels = {tuple(sorted((int(a), int(b)))): int(label)
                      for a, b, label in zip(row["pos1id"], row["pos2id"], row["pk"])}
            records.append({
                "source_record_id": f"{panel}#{index}",
                "panel": panel,
                "source_index": int(index),
                "sequence": sequence,
                "sequence_sha256": sha256_text(sequence),
                "rnaformer_raw_pairs": [list(pair) for pair in pairs],
                "rnaformer_pair_classes": {f"{pair[0]}:{pair[1]}": labels[pair] for pair in pairs},
                "source_family": str(row["family"]),
            })
    assert len(records) == 153
    assert len({row["source_record_id"] for row in records}) == 153
    return records


def search_payload(sequence: str) -> dict:
    return {
        "query": {"type": "terminal", "service": "sequence", "parameters": {
            "evalue_cutoff": 1,
            "identity_cutoff": 0.8,
            "sequence_type": "rna",
            "value": sequence,
        }},
        "return_type": "polymer_entity",
        "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]},
    }


def post(url: str, payload: dict, timeout: int = 120) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        if error.code == 204:
            return 204, b""
        raise


def freeze_one_search(sequence: str, directory: Path) -> dict:
    destination = directory / f"{sha256_text(sequence)}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    payload = search_payload(sequence)
    last_error = None
    for attempt in range(4):
        try:
            status, raw = post(SEARCH_URL, payload)
            response = {"result_set": [], "total_count": 0} if status == 204 else json.loads(raw)
            identifiers = [row["identifier"] for row in response.get("result_set", [])]
            assert len(identifiers) == len(set(identifiers))
            assert response.get("total_count", len(identifiers)) == len(identifiers)
            document = {
                "schema_version": SCHEMA,
                "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": SEARCH_URL,
                "http_status": status,
                "request": payload,
                "response": response,
                "response_sha256": sha256_bytes(raw),
                "candidate_entity_ids": identifiers,
                "candidate_scope": "retrieval only; identity 0.8 is not evidence of an exact match",
            }
            write_json(destination, document)
            return document
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = str(error)
            time.sleep(2 ** attempt)
    return {"schema_version": SCHEMA, "sequence_sha256": sha256_text(sequence),
            "status": "SEARCH_ERROR_UNCHECKED", "error": last_error,
            "candidate_entity_ids": [], "request": payload}


def command_search(args) -> None:
    records = source_records(args.rnaformer_plk)
    sequences = sorted({row["sequence"] for row in records})
    args.output.mkdir(parents=True, exist_ok=True)
    search_dir = args.output / "search"
    search_dir.mkdir(exist_ok=True)
    documents = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(freeze_one_search, sequence, search_dir): sequence for sequence in sequences}
        for future in concurrent.futures.as_completed(futures):
            sequence = futures[future]
            documents[sequence] = future.result()
            print(f"search {len(documents)}/{len(sequences)} candidates={len(documents[sequence]['candidate_entity_ids'])}", flush=True)
    index = {
        "schema_version": SCHEMA,
        "source_plk": {"path": str(args.rnaformer_plk), "sha256": sha256_file(args.rnaformer_plk)},
        "n_source_records": len(records),
        "n_unique_sequences": len(sequences),
        "records": [{key: value for key, value in row.items() if key != "rnaformer_pair_classes"} for row in records],
        "searches": {sha256_text(sequence): {
            "file": str(search_dir / f"{sha256_text(sequence)}.json"),
            "status": documents[sequence].get("status", "FROZEN"),
            "candidate_entity_ids": documents[sequence]["candidate_entity_ids"],
        } for sequence in sequences},
        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
    }
    write_json(args.output / "search_index.json", index)
    print(json.dumps({"status": "PASS_SEARCH_FREEZE_WITH_UNCHECKED_GAPS",
                      "unique_sequences": len(sequences),
                      "search_errors": sum(doc.get("status") == "SEARCH_ERROR_UNCHECKED" for doc in documents.values()),
                      "no_candidate": sum(not doc["candidate_entity_ids"] for doc in documents.values()),
                      "candidate_entity_memberships": sum(len(doc["candidate_entity_ids"]) for doc in documents.values())}, indent=2))


def is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(any(char == letter for char in iterator) for letter in needle)


def freeze_graphql(entity_ids: list[str], directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    entities = {}
    query = """query Entities($ids: [String!]!) {
      polymer_entities(entity_ids: $ids) {
        rcsb_id
        rcsb_polymer_entity_container_identifiers { entry_id entity_id auth_asym_ids asym_ids }
        entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
      }
    }"""
    for batch_index, start in enumerate(range(0, len(entity_ids), 100)):
        batch = entity_ids[start:start + 100]
        path = directory / f"batch_{batch_index:04d}.json"
        if path.exists():
            document = json.loads(path.read_text())
        else:
            payload = {"query": query, "variables": {"ids": batch}}
            status, raw = post(GRAPHQL_URL, payload, timeout=180)
            response = json.loads(raw)
            if response.get("errors"):
                raise RuntimeError(response["errors"])
            document = {"source": GRAPHQL_URL, "request": payload, "response": response,
                        "response_sha256": sha256_bytes(raw), "http_status": status}
            write_json(path, document)
        rows = document["response"]["data"]["polymer_entities"]
        assert {row["rcsb_id"] for row in rows} == set(batch)
        for row in rows:
            assert row["rcsb_id"] not in entities
            entities[row["rcsb_id"]] = row
    return entities


def download_mmcif(pdb_id: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size == 0:
        path.unlink()
    if not path.exists():
        url = MMCIF_URL.format(pdb_id=pdb_id)
        last_error = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=240) as response:
                    raw = response.read()
                temporary = path.with_suffix(".cif.part")
                temporary.write_bytes(raw)
                temporary.replace(path)
                break
            except urllib.error.URLError as error:
                last_error = str(error)
                time.sleep(2 ** attempt)
        else:
            return {"pdb_id": pdb_id, "status": "MMCIF_DOWNLOAD_ERROR_UNCHECKED", "error": last_error}
    return {"pdb_id": pdb_id, "status": "FROZEN", "path": str(path),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def mmcif_chain_sequences(path: Path, entity_ids: set[str]) -> list[dict]:
    import gemmi
    block = gemmi.cif.read(str(path)).sole_block()
    atom_observed = collections.defaultdict(set)
    atom_columns = ["group_PDB", "label_entity_id", "label_seq_id", "auth_asym_id"]
    for group, entity, seq_id, auth_chain in block.find("_atom_site.", atom_columns):
        if str(entity) in entity_ids and str(group) in {"ATOM", "HETATM"} and str(seq_id) not in {".", "?"}:
            atom_observed[(str(auth_chain), str(entity))].add(int(str(seq_id)))
    columns = ["asym_id", "entity_id", "seq_id", "mon_id", "auth_seq_num", "pdb_strand_id", "pdb_ins_code"]
    schemes = collections.defaultdict(list)
    for row in block.find("_pdbx_poly_seq_scheme.", columns):
        asym, entity, seq_id, monomer, auth_seq, auth_chain, insertion = map(str, row)
        if entity in entity_ids:
            schemes[(auth_chain, entity)].append({"asym_id": asym, "seq_id": int(seq_id), "monomer": monomer,
                                                  "auth_seq": auth_seq, "insertion": insertion})
    polymer_letters = {}
    for row in block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"]):
        if str(row[0]) in entity_ids:
            polymer_letters[str(row[0])] = str(row[1]).replace("\n", "").replace(";", "").replace(" ", "").upper().replace("T", "U")
    chains = []
    for (auth_chain, entity), rows in sorted(schemes.items()):
        rows.sort(key=lambda row: row["seq_id"])
        full = polymer_letters[entity]
        assert [row["seq_id"] for row in rows] == list(range(1, len(rows) + 1))
        assert len(rows) == len(full)
        observed_mask = [row["seq_id"] in atom_observed.get((auth_chain, entity), set()) for row in rows]
        scheme_mask = [row["auth_seq"] not in {"?", "."} for row in rows]
        sequence = "".join(full[row["seq_id"] - 1] for row in rows)
        chains.append({
            "auth_chain": auth_chain,
            "entity_id": entity,
            "asym_ids": sorted({row["asym_id"] for row in rows}),
            "full_sequence": sequence,
            "observed_mask": observed_mask,
            "observed_sequence": "".join(letter for letter, observed in zip(sequence, observed_mask) if observed),
            "scheme_atom_observation_disagreements": [index + 1 for index, (atom, scheme) in
                                                       enumerate(zip(observed_mask, scheme_mask)) if atom != scheme],
            "scheme_rows": rows,
        })
    return chains


def cached_pdb_chains(pdb_id: str, numeric_entities: set[str], download: dict, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{pdb_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        chains = mmcif_chain_sequences(Path(download["path"]), numeric_entities)
        result = {"status": "PASS", "chains": chains}
    except Exception as error:
        result = {"status": "MMCIF_PARSE_ERROR_UNCHECKED", "error": f"{type(error).__name__}: {error}"}
    temporary = cache.with_suffix(".json.part")
    write_json(temporary, result)
    temporary.replace(cache)
    return result


def command_resolve(args) -> None:
    index = json.loads((args.search / "search_index.json").read_text())
    records = index["records"]
    searches = index["searches"]
    score_one = {}
    for row in records:
        digest = row["sequence_sha256"]
        search_document = json.loads((args.search / "search" / f"{digest}.json").read_text())
        score_one[digest] = {hit["identifier"] for hit in search_document["response"].get("result_set", [])
                             if float(hit.get("score", 0.0)) == 1.0}
    entity_ids = sorted({entity for values in score_one.values() for entity in values})
    entities = freeze_graphql(entity_ids, args.output / "graphql") if entity_ids else {}
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_sequences = {}
    needed_entities = set()
    for row in records:
        sequence = row["sequence"]
        candidates = sorted(score_one[row["sequence_sha256"]])
        viable = []
        for entity_id in candidates:
            entity = entities[entity_id]
            polymer = entity["entity_poly"]
            full = polymer["pdbx_seq_one_letter_code_can"].replace("\n", "").replace(" ", "").upper().replace("T", "U")
            if polymer["rcsb_entity_polymer_type"] == "RNA" and len(sequence) <= len(full) and is_subsequence(sequence, full):
                viable.append(entity_id)
                needed_entities.add(entity_id)
        candidate_sequences[row["source_record_id"]] = viable
    pdb_ids = sorted({entities[entity]["rcsb_polymer_entity_container_identifiers"]["entry_id"] for entity in needed_entities})
    downloads = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_mmcif, pdb_id, args.output / "mmcif"): pdb_id for pdb_id in pdb_ids}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            downloads[result["pdb_id"]] = result
            print(f"mmCIF {len(downloads)}/{len(pdb_ids)} {result['pdb_id']} {result['status']}", flush=True)
    chains_by_entity, parse_results = {}, {}
    entities_by_pdb = collections.defaultdict(set)
    for entity_id in needed_entities:
        pdb_id = entities[entity_id]["rcsb_polymer_entity_container_identifiers"]["entry_id"]
        entities_by_pdb[pdb_id].add(entity_id.split("_")[-1])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(cached_pdb_chains, pdb_id, numeric_entities, downloads[pdb_id],
                                   args.output / "chain_sequences"): pdb_id
                   for pdb_id, numeric_entities in sorted(entities_by_pdb.items())
                   if downloads[pdb_id]["status"] == "FROZEN"}
        for future in concurrent.futures.as_completed(futures):
            pdb_id = futures.pop(future)
            result = future.result()
            parse_results[pdb_id] = {key: value for key, value in result.items() if key != "chains"}
            if result["status"] == "PASS":
                for full_chain in result["chains"]:
                    entity_id = f"{pdb_id}_{full_chain['entity_id']}"
                    chain = {key: value for key, value in full_chain.items() if key != "scheme_rows"}
                    chain.update({"pdb_id": pdb_id, "mmcif_sha256": downloads[pdb_id]["sha256"]})
                    chains_by_entity.setdefault(entity_id, []).append(chain)
            print(f"parse {len(parse_results)}/{len(entities_by_pdb)} {pdb_id} {result['status']}", flush=True)
    resolved = []
    for row in records:
        matches = []
        for entity_id in candidate_sequences[row["source_record_id"]]:
            for chain in chains_by_entity.get(entity_id, []):
                match_types = []
                if row["sequence"] == chain["full_sequence"]:
                    match_types.append("FULL_POLYMER_EXACT")
                if row["sequence"] == chain["observed_sequence"]:
                    match_types.append("OBSERVED_ONLY_EXACT")
                if match_types:
                    matches.append({"pdb_id": chain["pdb_id"], "entity_id": entity_id.split("_")[-1],
                                    "auth_chain": chain["auth_chain"], "asym_ids": chain["asym_ids"],
                                    "match_types": match_types, "mmcif_sha256": chain["mmcif_sha256"],
                                    "full_length": len(chain["full_sequence"]),
                                    "observed_length": len(chain["observed_sequence"]),
                                    "scheme_atom_observation_disagreements": chain["scheme_atom_observation_disagreements"]})
        matches.sort(key=lambda match: (match["pdb_id"], match["entity_id"], match["auth_chain"]))
        if len(matches) == 1:
            status = "UNIQUE_EXACT_CHAIN_MATCH"
        elif matches:
            status = "AMBIGUOUS_EXACT_CHAIN_MATCHES_UNCHECKED"
        elif searches[row["sequence_sha256"]].get("status") == "SEARCH_ERROR_UNCHECKED":
            status = "SEARCH_ERROR_UNCHECKED"
        elif any(downloads[entities[entity]["rcsb_polymer_entity_container_identifiers"]["entry_id"]]["status"] != "FROZEN"
                 for entity in candidate_sequences[row["source_record_id"]]):
            status = "MMCIF_DOWNLOAD_ERROR_UNCHECKED"
        elif any(parse_results.get(entities[entity]["rcsb_polymer_entity_container_identifiers"]["entry_id"], {}).get("status") == "MMCIF_PARSE_ERROR_UNCHECKED"
                 for entity in candidate_sequences[row["source_record_id"]]):
            status = "MMCIF_PARSE_ERROR_UNCHECKED"
        else:
            status = "NO_EXACT_CHAIN_MATCH_UNCHECKED"
        resolved.append({**row, "search_candidate_entity_ids": searches[row["sequence_sha256"]]["candidate_entity_ids"],
                         "score_one_candidate_entity_ids": sorted(score_one[row["sequence_sha256"]]),
                         "subsequence_candidate_entity_ids": candidate_sequences[row["source_record_id"]],
                         "exact_chain_matches": matches, "identity_status": status,
                         "selected_identity": matches[0] if len(matches) == 1 else None,
                         "selection_rule": "exactly one exact full-polymer or observed-only chain; no row-number or pair-based identity guessing",
                         "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"})
    document = {"schema_version": SCHEMA, "n_records": len(resolved),
                "status_counts": dict(collections.Counter(row["identity_status"] for row in resolved)),
                "n_unique_sequences": index["n_unique_sequences"], "n_search_candidate_entities": len(entity_ids),
                "n_subsequence_candidate_entities": len(needed_entities), "mmcif_downloads": downloads,
                "mmcif_parse_errors": {pdb: result for pdb, result in parse_results.items() if result["status"] != "PASS"},
                "records": resolved, "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "resolved_identities.json", document)
    print(json.dumps({"status": "PASS_IDENTITY_RESOLUTION_WITH_UNCHECKED_GAPS",
                      "status_counts": document["status_counts"]}, indent=2))


NT_RE = re.compile(r"^(?P<chain>[^.]+)\.(?P<name>.+?)(?P<seq>-?\d+)(?P<ins>(?:\^[A-Za-z]|[A-Za-z])?)$")


def run_dssr(pdb_id: str, mmcif: Path, binary: Path, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "dssr.json"
    if not output.exists():
        completed = subprocess.run([str(binary), f"-i={mmcif}", "--json", "--more", f"-o={output}.part"],
                                   capture_output=True, text=True, cwd=directory, timeout=600)
        (directory / "dssr.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode or not (directory / "dssr.json.part").exists():
            return {"status": "DSSR_ERROR_UNCHECKED", "returncode": completed.returncode}
        (directory / "dssr.json.part").replace(output)
    return {"status": "PASS", "path": str(output), "sha256": sha256_file(output),
            "dssr_binary_sha256": sha256_file(binary)}


def dssr_chain_pairs(dssr: dict, chain: dict) -> dict:
    by_author = {}
    for row, observed in zip(chain["scheme_rows"], chain["observed_mask"]):
        if not observed:
            continue
        insertion = "" if row["insertion"] in {"?", "."} else row["insertion"].lstrip("^").upper()
        key = (row["auth_seq"], insertion)
        assert key not in by_author
        by_author[key] = (row["seq_id"] - 1, observed)
    observed_to_full = [index for index, observed in enumerate(chain["observed_mask"]) if observed]
    full_to_observed = {full: observed for observed, full in enumerate(observed_to_full)}
    full_pairs, observed_pairs, records, errors = [], [], [], []
    other_chain_pairs = 0
    for pair in dssr.get("pairs", []):
        left, right = NT_RE.fullmatch(pair["nt1"]), NT_RE.fullmatch(pair["nt2"])
        if not left or not right:
            errors.append(f"invalid DSSR nucleotide ID: {pair}")
            continue
        if left.group("chain") != chain["auth_chain"] or right.group("chain") != chain["auth_chain"]:
            other_chain_pairs += 1
            continue
        mapped = []
        for match in (left, right):
            key = (match.group("seq"), match.group("ins").lstrip("^").upper())
            if key not in by_author:
                errors.append(f"endpoint absent from mmCIF author scheme: {pair}")
                break
            full_index, observed = by_author[key]
            if not observed:
                errors.append(f"endpoint absent from atom_site observed mask: {pair}")
                break
            if full_index not in full_to_observed:
                errors.append(f"endpoint has no observed-only coordinate: {pair}")
                break
            mapped.append(full_index)
        if len(mapped) != 2:
            continue
        full_pair = sorted(mapped)
        observed_pair = sorted(full_to_observed[index] for index in full_pair)
        if full_pair in full_pairs:
            errors.append(f"duplicate mapped DSSR pair: {pair}")
            continue
        full_pairs.append(full_pair)
        observed_pairs.append(observed_pair)
        records.append({"nt1": pair["nt1"], "nt2": pair["nt2"], "LW": pair.get("LW"),
                        "full_pair": full_pair, "observed_pair": observed_pair})
    order = sorted(range(len(full_pairs)), key=lambda index: full_pairs[index])
    return {"status": "PASS" if not errors else "DSSR_MAPPING_ERROR_UNCHECKED",
            "errors": errors, "other_chain_pairs": other_chain_pairs,
            "full_pairs": [full_pairs[index] for index in order],
            "observed_pairs": [observed_pairs[index] for index in order],
            "pair_records": [records[index] for index in order]}


def command_build(args) -> None:
    identity_document = json.loads((args.identities / "resolved_identities.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rebuilt, unresolved = [], []
    dssr_assets = {}
    for row in identity_document["records"]:
        if set(row["sequence"]) - set("ACGU"):
            unresolved.append({**row, "rebuild_status": "SOURCE_BAD_ALPHABET_UNCHECKED"})
            continue
        if row["identity_status"] != "UNIQUE_EXACT_CHAIN_MATCH":
            unresolved.append({**row, "rebuild_status": row["identity_status"]})
            continue
        identity = row["selected_identity"]
        pdb_id, entity = identity["pdb_id"], identity["entity_id"]
        mmcif = args.identities / "mmcif" / f"{pdb_id}.cif"
        chain = next(chain for chain in mmcif_chain_sequences(mmcif, {entity})
                     if chain["auth_chain"] == identity["auth_chain"])
        if chain["scheme_atom_observation_disagreements"]:
            unresolved.append({**row, "rebuild_status": "ATOM_SCHEME_CONFLICT_UNCHECKED"})
            continue
        if pdb_id not in dssr_assets:
            dssr_assets[pdb_id] = run_dssr(pdb_id, mmcif, args.dssr, args.output / "dssr" / pdb_id)
        asset = dssr_assets[pdb_id]
        if asset["status"] != "PASS":
            unresolved.append({**row, "rebuild_status": asset["status"], "dssr_asset": asset})
            continue
        dssr = json.loads(Path(asset["path"]).read_text())
        mapped = dssr_chain_pairs(dssr, chain)
        if mapped["status"] != "PASS":
            unresolved.append({**row, "rebuild_status": mapped["status"], "dssr_errors": mapped["errors"],
                               "dssr_asset": asset})
            continue
        observed_to_full = [index for index, observed in enumerate(chain["observed_mask"]) if observed]
        reference_id = f"RCSB:{pdb_id}_{entity}:{identity['auth_chain']}"
        original_pairs = sorted(tuple(pair) for pair in row["rnaformer_raw_pairs"])
        comparison_coordinate = "OBSERVED_ONLY" if "OBSERVED_ONLY_EXACT" in identity["match_types"] else "FULL_POLYMER"
        comparison_pairs = mapped["observed_pairs"] if comparison_coordinate == "OBSERVED_ONLY" else mapped["full_pairs"]
        dssr_comparison_pairs = sorted(tuple(pair) for pair in comparison_pairs)
        common = {
            "source_record_id": row["source_record_id"], "panel": row["panel"], "reference_id": reference_id,
            "pdb_id": pdb_id, "entity_id": entity, "auth_chain": identity["auth_chain"],
            "asym_ids": chain["asym_ids"], "identity_match_types": identity["match_types"],
            "mmcif_sha256": identity["mmcif_sha256"], "dssr_json_sha256": asset["sha256"],
            "dssr_binary_sha256": asset["dssr_binary_sha256"], "dssr_program": dssr.get("program"),
            "full_sequence": chain["full_sequence"], "observed_mask": "".join("1" if x else "0" for x in chain["observed_mask"]),
            "observed_sequence": chain["observed_sequence"], "observed_to_full_indices": observed_to_full,
            "full_pairs": mapped["full_pairs"], "observed_pairs": mapped["observed_pairs"],
            "dssr_pair_records": mapped["pair_records"], "rnaformer_raw_pairs": row["rnaformer_raw_pairs"],
            "rnaformer_dssr_pair_match": original_pairs == dssr_comparison_pairs,
            "rnaformer_dssr_pair_comparison_coordinate": comparison_coordinate,
            "rnaformer_dssr_pair_delta": {
                "rnaformer_only": [list(pair) for pair in sorted(set(original_pairs) - set(dssr_comparison_pairs))],
                "dssr_only": [list(pair) for pair in sorted(set(dssr_comparison_pairs) - set(original_pairs))],
            },
            "unobserved_negative_policy": "MASKED_NEVER_NEGATIVE",
            "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION",
        }
        full_record = {**common, "protocol": "FULL_POLYMER_OBSERVED_MASK_V1",
                       "sequence": chain["full_sequence"], "pairs": mapped["full_pairs"],
                       "negative_rule": "only observed_mask=1 positions may contribute observed-unpaired negatives; mask=0 is excluded"}
        observed_record = {**common, "protocol": "OBSERVED_ONLY_V1",
                           "sequence": chain["observed_sequence"], "pairs": mapped["observed_pairs"],
                           "negative_rule": "all retained positions are atom_site-observed; excluded full-polymer positions are absent, not negative"}
        rebuilt.append({"source_record_id": row["source_record_id"], "panel": row["panel"],
                        "rebuild_status": "REBUILT_DSSR_REFERENCE", "full": full_record, "observed": observed_record})
    panel_names = {"pdb_ts1": "TS1", "pdb_ts2": "TS2", "pdb_ts3": "TS3", "pdb_ts_hard": "TS-hard"}
    for panel, label in panel_names.items():
        for protocol, key in (("full_polymer_observed_mask", "full"), ("observed_only", "observed")):
            with (args.output / f"{label}.{protocol}.jsonl").open("w") as handle:
                for row in rebuilt:
                    if row["panel"] == panel:
                        handle.write(json.dumps(row[key], sort_keys=True) + "\n")
    with (args.output / "unresolved_records.jsonl").open("w") as handle:
        for row in unresolved:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    recompute = {"schema_version": SCHEMA, "policy": "New DSSR references require complete metric recomputation; no model selection",
                 "rebuilt": [{"source_record_id": row["source_record_id"], "panel": row["panel"],
                              "reference_id": row["full"]["reference_id"],
                              "rnaformer_dssr_pair_match": row["full"]["rnaformer_dssr_pair_match"]}
                             for row in rebuilt],
                 "unresolved": [{"source_record_id": row["source_record_id"], "status": row["rebuild_status"]}
                                for row in unresolved],
                 "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "recompute_sets.json", recompute)
    summary = {"schema_version": SCHEMA, "n_source_records": len(identity_document["records"]),
               "n_rebuilt": len(rebuilt), "n_unresolved": len(unresolved),
               "panel_rebuilt": dict(collections.Counter(row["panel"] for row in rebuilt)),
               "pair_match_counts": dict(collections.Counter(row["full"]["rnaformer_dssr_pair_match"] for row in rebuilt)),
               "unresolved_status_counts": dict(collections.Counter(row["rebuild_status"] for row in unresolved)),
               "dssr_assets": dssr_assets, "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "rebuild_summary.json", summary)
    lines = [
        "# RNAformer PDB RCSB/mmCIF/DSSR rebuild", "",
        "External Test is audit-only. No prediction, metric, checkpoint, or model-selection outcome is used.", "",
        f"- Source candidates: 153; rebuilt with a unique exact chain and valid DSSR mapping: {len(rebuilt)}.",
        f"- Unresolved/UNCHECKED records: {len(unresolved)}; status counts: {summary['unresolved_status_counts']}.",
        f"- RNAformer-vs-DSSR raw pair exact-match counts: {summary['pair_match_counts']}. Differences are preserved and require complete metric recomputation.", "",
        "## Protocols", "",
        "- `FULL_POLYMER_OBSERVED_MASK_V1`: full polymer sequence and full-polymer 0-based DSSR pairs; only `observed_mask=1` positions may contribute observed-unpaired negatives. Unobserved positions are masked, never negative.",
        "- `OBSERVED_ONLY_V1`: only atom_site-observed residues remain, with observed-only 0-based DSSR pairs and `observed_to_full_indices` for coordinate recovery.",
        "- RCSB search score and subsequence tests are candidate filters only. Identity requires exactly one exact full-polymer or observed-only `(PDB, entity, auth_chain)` match. One-to-many and zero-match records remain `UNCHECKED`.",
        "- `atom_site` is primary for observation; `pdbx_poly_seq_scheme.auth_seq_num` is cross-checked. DSSR pairs are retained without canonical/nested projection.", "",
        "See `recompute_sets.json` and `unresolved_records.jsonl` for the complete rebuild and gap sets.", "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(args.output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(args.output / "artifact_manifest.json", {"schema_version": SCHEMA, "files": files,
                                                         "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"})
    print(json.dumps({"status": "PASS_DOUBLE_PROTOCOL_BUILD_WITH_UNCHECKED_GAPS", **summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--rnaformer-plk", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.set_defaults(function=command_search)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--search", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.set_defaults(function=command_resolve)
    build = subparsers.add_parser("build")
    build.add_argument("--identities", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--dssr", type=Path, default=Path("/home/rna_jev/third_party/DSSR-v1.9.10/x3dna-dssr"))
    build.set_defaults(function=command_build)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
