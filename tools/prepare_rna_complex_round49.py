#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

import gemmi

NT_RE = re.compile(r"^(?P<chain>[^.]+)\.(?P<name>.+?)(?P<seq>-?\d+)(?P<ins>[A-Za-z]?)$")
STANDARD = {"A", "C", "G", "U"}
CANONICAL = {"G-C", "C-G", "A-U", "U-A", "G-U", "U-G"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_letter(name, parents):
    if name in STANDARD:
        return name
    info = gemmi.find_tabulated_residue(name)
    letter = info.one_letter_code.upper() if info else ""
    if letter not in STANDARD:
        letter = parents.get(name, {}).get("parent", "")
    if letter not in STANDARD:
        raise ValueError(f"unsupported modified RNA residue: {name}")
    return letter


def parse_nt(value):
    match = NT_RE.match(value)
    if not match:
        raise ValueError(f"invalid DSSR nucleotide id: {value}")
    row = match.groupdict()
    return row["chain"], int(row["seq"]), row["ins"], row["name"]


def nested_canonical(pairs):
    ordered = sorted(pairs)
    crossing = False
    for left, right in ordered:
        for other_left, other_right in ordered:
            if left < other_left < right < other_right:
                crossing = True
    conflict = crossing or len({i for pair in pairs for i in pair}) != 2 * len(pairs)
    return [] if conflict else ordered


def value(block, category, tag):
    result = block.find_value(f"{category}.{tag}")
    return result.strip() if result else None


def extract(cif_path, dssr_path, dssr_sha256, min_length=30):
    cif_path, dssr_path = Path(cif_path), Path(dssr_path)
    parent_path = dssr_path.with_name("parents.json")
    parent_document = json.loads(parent_path.read_text())
    if parent_document["failures"] or parent_document["cif_sha256"] != sha256(cif_path):
        raise ValueError("parent map is incomplete or belongs to another mmCIF")
    parents = parent_document["parents"]
    block = gemmi.cif.read(str(cif_path)).sole_block()
    pdb_id = value(block, "_entry", "id").upper()
    columns = ["asym_id", "entity_id", "seq_id", "mon_id", "auth_seq_num", "pdb_strand_id", "pdb_ins_code"]
    chains, rejected = {}, []
    entity_types = {str(row[0]): str(row[1]) for row in block.find("_entity_poly.", ["entity_id", "type"])}
    if not entity_types:
        raise ValueError("missing entity polymer metadata")
    for row in block.find("_pdbx_poly_seq_scheme.", columns):
        asym, entity, seq_id, monomer, auth_seq, auth_chain, insertion = map(str, row)
        key = (int(auth_seq), "" if insertion == "." else insertion)
        try:
            if entity_types.get(entity) != "polyribonucleotide":
                raise ValueError("not a pure RNA polymer entity")
            letter = one_letter(monomer, parents)
        except (ValueError, TypeError) as error:
            rejected.append({"auth_chain": auth_chain, "reason": str(error)})
            continue
        if key in chains.setdefault(auth_chain, {}):
            raise ValueError(f"duplicate residue mapping: {key}")
        chains[auth_chain][key] = {"asym_id": asym, "entity_id": entity, "seq_id": int(seq_id), "letter": letter, "monomer": monomer}
    dssr = json.loads(dssr_path.read_text())
    raw_by_chain = {chain: [] for chain in chains}
    raw_records = {chain: [] for chain in chains}
    errors = []
    for pair in dssr.get("pairs", []):
        nt1, nt2 = parse_nt(pair["nt1"]), parse_nt(pair["nt2"])
        if nt1[0] != nt2[0]:
            continue
        chain = nt1[0]
        if chain not in chains:
            errors.append(f"unknown chain: {chain}")
            continue
        try:
            left, right = chains[chain][nt1[1:3]], chains[chain][nt2[1:3]]
            if one_letter(nt1[3], parents) != left["letter"] or one_letter(nt2[3], parents) != right["letter"]:
                raise ValueError("DSSR/mmCIF base identity mismatch")
            a, b = sorted((left["seq_id"], right["seq_id"]))
            if (a, b) in raw_by_chain[chain]:
                raise ValueError("duplicate DSSR pair")
            raw_by_chain[chain].append((a, b))
            raw_records[chain].append(pair)
        except (KeyError, ValueError) as error:
            errors.append(f"{pair['nt1']}--{pair['nt2']}: {error}")
    if errors:
        raise ValueError("DSSR mapping failed: " + "; ".join(errors[:12]))
    output = []
    for chain, mapping in sorted(chains.items()):
        ordered = sorted(mapping.values(), key=lambda row: row["seq_id"])
        seq_ids = [row["seq_id"] for row in ordered]
        if seq_ids != list(range(1, len(seq_ids) + 1)):
            rejected.append({"auth_chain": chain, "reason": "non-contiguous entity sequence"})
            continue
        sequence = "".join(row["letter"] for row in ordered)
        raw = sorted(raw_by_chain[chain])
        canonical_pairs = []
        for pair in raw_records[chain]:
            a, b = sorted((mapping[parse_nt(pair["nt1"])[1:3]]["seq_id"], mapping[parse_nt(pair["nt2"])[1:3]]["seq_id"]))
            base_pair = sequence[a - 1] + "-" + sequence[b - 1]
            if base_pair in CANONICAL and pair.get("LW") == "cWW":
                canonical_pairs.append([a, b])
        canonical = nested_canonical(canonical_pairs)
        if len(sequence) < min_length or not raw:
            rejected.append({"auth_chain": chain, "reason": "short or no intrachain DSSR pair", "length": len(sequence)})
            continue
        output.append({"id": f"{pdb_id}_{chain}", "pdb_id": pdb_id, "auth_chain": chain,
                       "asym_ids": sorted({row["asym_id"] for row in ordered}),
                       "entity_ids": sorted({row["entity_id"] for row in ordered}),
                       "sequence": sequence, "length": len(sequence), "raw_pairs": raw,
                       "canonical_nested_pairs": canonical,
                       "raw_pair_records": raw_records[chain],
                       "n_pseudoknot_or_noncanonical": len(raw) - len(canonical)})
    return {"pdb_id": pdb_id, "experimental_method": value(block, "_exptl", "method"),
            "resolution_angstrom": value(block, "_refine", "ls_d_res_high"),
            "deposition_date": value(block, "_pdbx_database_status", "recvd_initial_deposition_date"),
            "cif_sha256": sha256(cif_path), "dssr_json_sha256": sha256(dssr_path),
            "parent_map_sha256": sha256(parent_path), "modified_parent_map": parents,
            "dssr_binary_sha256": dssr_sha256, "dssr_program": dssr.get("program"),
            "n_dssr_pairs": dssr.get("num_pairs"), "chains": output, "rejected_chains": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True)
    parser.add_argument("--dssr-json", required=True)
    parser.add_argument("--dssr-binary-sha256", required=True)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = extract(args.cif, args.dssr_json, args.dssr_binary_sha256, args.min_length)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pdb_id": result["pdb_id"], "accepted": len(result["chains"]),
                      "rejected": len(result["rejected_chains"])}, sort_keys=True))


if __name__ == "__main__":
    main()
