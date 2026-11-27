#!/usr/bin/env python3
"""Build incremental metadata-resolved references and scheme-filtered DSSR repairs."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import shutil
from pathlib import Path

SCHEMA = "foldnar.rnaformer_pdb_rcsb_rebuild.v1"


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_builder():
    path = Path(__file__).with_name("rebuild_rnaformer_pdb_labels_20261127.py")
    spec = importlib.util.spec_from_file_location("rnaformer_rebuild_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filtered_pairs(builder, dssr, chain):
    by_author = {}
    for row, observed in zip(chain["scheme_rows"], chain["observed_mask"]):
        if observed:
            insertion = "" if row["insertion"] in {"?", "."} else row["insertion"].lstrip("^").upper()
            by_author[(row["auth_seq"], insertion)] = row["seq_id"] - 1
    observed_to_full = [index for index, observed in enumerate(chain["observed_mask"]) if observed]
    full_to_observed = {full: observed for observed, full in enumerate(observed_to_full)}
    retained, excluded = [], []
    for pair in dssr.get("pairs", []):
        left, right = builder.NT_RE.fullmatch(pair["nt1"]), builder.NT_RE.fullmatch(pair["nt2"])
        if not left or not right or left.group("chain") != chain["auth_chain"] or right.group("chain") != chain["auth_chain"]:
            continue
        keys = [(match.group("seq"), match.group("ins").lstrip("^").upper()) for match in (left, right)]
        if any(key not in by_author for key in keys):
            excluded.append({"nt1": pair["nt1"], "nt2": pair["nt2"],
                             "reason": "ENDPOINT_OUTSIDE_SELECTED_POLYMER_ENTITY",
                             "official_scheme_scope": "pair excluded without changing identity or treating an unobserved residue as negative"})
            continue
        full_pair = sorted(by_author[key] for key in keys)
        retained.append({"nt1": pair["nt1"], "nt2": pair["nt2"], "LW": pair.get("LW"),
                         "full_pair": full_pair, "observed_pair": sorted(full_to_observed[index] for index in full_pair)})
    retained.sort(key=lambda row: row["full_pair"])
    assert len({tuple(row["full_pair"]) for row in retained}) == len(retained)
    return {"full_pairs": [row["full_pair"] for row in retained],
            "observed_pairs": [row["observed_pair"] for row in retained],
            "pair_records": retained, "excluded_pair_records": excluded}


def make_reference(builder, row, identity, mmcif, dssr_binary, output, repair):
    entity = identity["entity_id"]
    chains = builder.mmcif_chain_sequences(mmcif, {entity})
    chain = next((chain for chain in chains if chain["auth_chain"] == identity["auth_chain"]), None)
    if chain is None:
        return None, "MMCIF_CHAIN_VERIFICATION_FAILED_UNCHECKED"
    if not repair and row["sequence"] != chain["full_sequence"]:
        return None, "MMCIF_FULL_POLYMER_VERIFICATION_FAILED_UNCHECKED"
    if repair and not ((row["sequence"] == chain["full_sequence"] and "FULL_POLYMER_EXACT" in identity["match_types"]) or
                       (row["sequence"] == chain["observed_sequence"] and "OBSERVED_ONLY_EXACT" in identity["match_types"])):
        return None, "MMCIF_REPAIR_SEQUENCE_VERIFICATION_FAILED_UNCHECKED"
    if chain["scheme_atom_observation_disagreements"]:
        return None, "ATOM_SCHEME_CONFLICT_UNCHECKED"
    asset = builder.run_dssr(identity["pdb_id"], mmcif, dssr_binary, output / "dssr" / identity["pdb_id"])
    if asset["status"] != "PASS":
        return None, "DSSR_ERROR_UNCHECKED"
    dssr = json.loads(Path(asset["path"]).read_text())
    if repair:
        mapped = filtered_pairs(builder, dssr, chain)
    else:
        mapped = builder.dssr_chain_pairs(dssr, chain)
        if mapped["status"] != "PASS":
            return None, "DSSR_MAPPING_ERROR_UNCHECKED"
        mapped["excluded_pair_records"] = []
    observed_to_full = [index for index, observed in enumerate(chain["observed_mask"]) if observed]
    original_pairs = sorted(tuple(pair) for pair in row["rnaformer_raw_pairs"])
    comparison = mapped["observed_pairs"] if "OBSERVED_ONLY_EXACT" in identity["match_types"] else mapped["full_pairs"]
    comparison = sorted(tuple(pair) for pair in comparison)
    identity = {**identity, "mmcif_sha256": sha256_file(mmcif)}
    common = {"source_record_id": row["source_record_id"], "panel": row["panel"],
              "reference_id": f"RCSB:{identity['pdb_id']}_{entity}:{identity['auth_chain']}",
              "pdb_id": identity["pdb_id"], "entity_id": entity, "auth_chain": identity["auth_chain"],
              "asym_ids": chain["asym_ids"], "identity_match_types": identity["match_types"],
              "mmcif_sha256": identity["mmcif_sha256"], "dssr_json_sha256": asset["sha256"],
              "dssr_binary_sha256": asset["dssr_binary_sha256"], "dssr_program": dssr.get("program"),
              "full_sequence": chain["full_sequence"], "observed_mask": "".join("1" if x else "0" for x in chain["observed_mask"]),
              "observed_sequence": chain["observed_sequence"], "observed_to_full_indices": observed_to_full,
              "full_pairs": mapped["full_pairs"], "observed_pairs": mapped["observed_pairs"],
              "dssr_pair_records": mapped["pair_records"],
              "dssr_excluded_pair_records": mapped["excluded_pair_records"],
              "scheme_repair_protocol": "OFFICIAL_SEQ_SCHEME_ENDPOINT_FILTER_V1" if repair else None,
              "rnaformer_raw_pairs": row["rnaformer_raw_pairs"],
              "rnaformer_dssr_pair_match": original_pairs == comparison,
              "rnaformer_dssr_pair_comparison_coordinate": "OBSERVED_ONLY" if "OBSERVED_ONLY_EXACT" in identity["match_types"] else "FULL_POLYMER",
              "rnaformer_dssr_pair_delta": {
                  "rnaformer_only": [list(pair) for pair in sorted(set(original_pairs) - set(comparison))],
                  "dssr_only": [list(pair) for pair in sorted(set(comparison) - set(original_pairs))]},
              "unobserved_negative_policy": "MASKED_NEVER_NEGATIVE",
              "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    full = {**common, "protocol": "FULL_POLYMER_OBSERVED_MASK_V1", "sequence": chain["full_sequence"],
            "pairs": mapped["full_pairs"],
            "negative_rule": "only observed_mask=1 positions may contribute observed-unpaired negatives; mask=0 is excluded"}
    observed = {**common, "protocol": "OBSERVED_ONLY_V1", "sequence": chain["observed_sequence"],
                "pairs": mapped["observed_pairs"],
                "negative_rule": "all retained positions are atom_site-observed; excluded full-polymer positions are absent, not negative"}
    return {"row": row, "identity": identity, "full": full, "observed": observed}, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dssr", type=Path, default=Path("/home/rna_jev/third_party/DSSR-v1.9.10/x3dna-dssr"))
    args = parser.parse_args()
    builder = load_builder()
    metadata = json.loads((args.metadata / "metadata_resolution.json").read_text())
    previous_unresolved = [json.loads(line) for line in (args.previous / "unresolved_records.jsonl").open() if line.strip()]
    repairs = [row for row in previous_unresolved if row["rebuild_status"] == "DSSR_MAPPING_ERROR_UNCHECKED"]
    args.output.mkdir(parents=True, exist_ok=True)
    added, remaining = [], []
    for row in metadata["records"]:
        if row["metadata_status"] != "NEW_UNIQUE_METADATA_CHAIN":
            remaining.append({**row, "rebuild_status": row["metadata_status"]})
            continue
        identity = row["selected_identity"]
        mmcif = args.output / "mmcif" / f"{identity['pdb_id']}.cif"
        if not mmcif.exists():
            source = args.identities / "mmcif" / f"{identity['pdb_id']}.cif"
            if source.exists():
                mmcif.parent.mkdir(exist_ok=True)
                mmcif.write_bytes(source.read_bytes())
            else:
                result = builder.download_mmcif(identity["pdb_id"], args.output / "mmcif")
                if result["status"] != "FROZEN":
                    remaining.append({**row, "rebuild_status": "MMCIF_DOWNLOAD_ERROR_UNCHECKED"})
                    continue
        result, error = make_reference(builder, row, identity, mmcif, args.dssr, args.output, False)
        if error:
            remaining.append({**row, "rebuild_status": error})
        else:
            added.append(result)
    repair_results = []
    for row in repairs:
        identity = row["selected_identity"]
        mmcif = args.identities / "mmcif" / f"{identity['pdb_id']}.cif"
        result, error = make_reference(builder, row, identity, mmcif, args.dssr, args.output, True)
        if error:
            remaining.append({**row, "rebuild_status": error})
        else:
            repair_results.append(result)
    for row in previous_unresolved:
        if row["rebuild_status"] == "SOURCE_BAD_ALPHABET_UNCHECKED":
            remaining.append(row)
    labels = {"pdb_ts1": "TS1", "pdb_ts2": "TS2", "pdb_ts3": "TS3", "pdb_ts_hard": "TS-hard"}
    all_new = added + repair_results
    for panel, label in labels.items():
        for protocol, key in (("full_polymer_observed_mask", "full"), ("observed_only", "observed")):
            with (args.output / f"{label}.{protocol}.incremental.jsonl").open("w") as handle:
                for result in all_new:
                    if result["row"]["panel"] == panel:
                        handle.write(json.dumps(result[key], sort_keys=True) + "\n")
    with (args.output / "remaining_records.jsonl").open("w") as handle:
        for row in remaining:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    recompute = {"schema_version": SCHEMA,
                 "new_metadata_references": [{"source_record_id": result["row"]["source_record_id"],
                                               "reference_id": result["full"]["reference_id"]} for result in added],
                 "scheme_repaired_references": [{"source_record_id": result["row"]["source_record_id"],
                                                  "reference_id": result["full"]["reference_id"],
                                                  "n_excluded_pairs": len(result["full"]["dssr_excluded_pair_records"])}
                                                 for result in repair_results],
                 "remaining": [{"source_record_id": row["source_record_id"], "status": row["rebuild_status"]} for row in remaining],
                 "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "recompute_sets.json", recompute)
    summary = {"schema_version": SCHEMA, "n_new_metadata_references": len(added),
               "n_scheme_repaired_references": len(repair_results), "n_remaining": len(remaining),
               "remaining_status_counts": dict(collections.Counter(row["rebuild_status"] for row in remaining)),
               "excluded_pairs_by_repair": {result["row"]["source_record_id"]: len(result["full"]["dssr_excluded_pair_records"])
                                             for result in repair_results},
               "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "incremental_summary.json", summary)
    augmented = json.loads((args.identities / "resolved_identities.json").read_text())
    updates = {}
    for result in all_new:
        updates[result["row"]["source_record_id"]] = result
    for row in augmented["records"]:
        if row["source_record_id"] in updates:
            result = updates[row["source_record_id"]]
            row["selected_identity"] = result["identity"]
            row["identity_status"] = "UNIQUE_EXACT_CHAIN_MATCH"
    write_json(args.output / "resolved_identities.json", augmented)
    combined = args.output / "combined"
    combined.mkdir(exist_ok=True)
    shutil.copytree(args.previous / "dssr", combined / "dssr", dirs_exist_ok=True)
    shutil.copytree(args.output / "dssr", combined / "dssr", dirs_exist_ok=True)
    previous_recompute = json.loads((args.previous / "recompute_sets.json").read_text())
    combined_rebuilt = list(previous_recompute["rebuilt"])
    for panel, label in labels.items():
        for protocol, key in (("full_polymer_observed_mask", "full"), ("observed_only", "observed")):
            previous_rows = [json.loads(line) for line in (args.previous / f"{label}.{protocol}.jsonl").open() if line.strip()]
            new_rows = [result[key] for result in all_new if result["row"]["panel"] == panel]
            with (combined / f"{label}.{protocol}.jsonl").open("w") as handle:
                for row in previous_rows + new_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    for result in all_new:
        combined_rebuilt.append({"source_record_id": result["row"]["source_record_id"],
                                 "panel": result["row"]["panel"], "reference_id": result["full"]["reference_id"],
                                 "rnaformer_dssr_pair_match": result["full"]["rnaformer_dssr_pair_match"]})
    combined_rebuilt.sort(key=lambda row: row["source_record_id"])
    combined_recompute = {"schema_version": SCHEMA, "policy": "New DSSR references require complete metric recomputation; no model selection",
                          "rebuilt": combined_rebuilt,
                          "unresolved": [{"source_record_id": row["source_record_id"], "status": row["rebuild_status"]}
                                         for row in remaining],
                          "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(combined / "recompute_sets.json", combined_recompute)
    with (combined / "unresolved_records.jsonl").open("w") as handle:
        for row in remaining:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    pair_matches = collections.Counter(result["full"]["rnaformer_dssr_pair_match"] for result in all_new)
    previous_summary = json.loads((args.previous / "rebuild_summary.json").read_text())
    combined_summary = {"schema_version": SCHEMA, "n_source_records": 153,
                        "n_rebuilt": len(combined_rebuilt), "n_unresolved": len(remaining),
                        "panel_rebuilt": dict(collections.Counter(row["panel"] for row in combined_rebuilt)),
                        "pair_match_counts": {
                            "true": previous_summary["pair_match_counts"].get("true", 0) + pair_matches[True],
                            "false": previous_summary["pair_match_counts"].get("false", 0) + pair_matches[False]},
                        "unresolved_status_counts": dict(collections.Counter(row["rebuild_status"] for row in remaining)),
                        "new_metadata_references": len(added), "scheme_repaired_references": len(repair_results),
                        "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(combined / "rebuild_summary.json", combined_summary)
    report = ["# RNAformer PDB metadata resolution and observed-safe references", "",
              "External Test is audit-only. No pairs were used to select identity and no model ranking or selection was performed.", "",
              f"- New unique metadata/full-polymer references: {len(added)}.",
              f"- Official seq-scheme DSSR repairs: {len(repair_results)}; excluded endpoint counts: {summary['excluded_pairs_by_repair']}.",
              f"- Combined references: {len(combined_rebuilt)}; remaining UNCHECKED: {len(remaining)}.",
              f"- Remaining status counts: {combined_summary['unresolved_status_counts']}.", "",
              "Post-RNAformer entries are hard-excluded, never date-ranked. Original family text is used only when present. Full-polymer exact sequence, RNA type and chain identity are mandatory.", ""]
    (combined / "report.md").write_text("\n".join(report))
    files = {}
    for path in sorted(combined.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(combined))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(combined / "artifact_manifest.json", {"schema_version": SCHEMA, "files": files,
                                                      "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"})
    print(json.dumps({"status": "PASS_INCREMENTAL_REFERENCE_BUILD", **summary,
                      "combined_rebuilt": len(combined_rebuilt)}, indent=2))


if __name__ == "__main__":
    main()
