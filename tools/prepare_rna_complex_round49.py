#!/usr/bin/env python3
import argparse
from bisect import bisect_left, bisect_right
from functools import lru_cache
import hashlib
import json
import re
import sys
from pathlib import Path

import gemmi

NT_RE = re.compile(r"^(?P<chain>[^.]+)\.(?P<name>.+?)(?P<seq>-?\d+)(?P<ins>(?:\^[A-Za-z]|[A-Za-z])?)$")
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
    ordered = sorted(tuple(pair) for pair in pairs)
    if any(left >= right for left, right in ordered):
        raise ValueError("invalid base-pair interval")
    starts = [left for left, _ in ordered]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * len(ordered) + 100))

    @lru_cache(maxsize=None)
    def best(index, limit):
        if index >= len(ordered):
            return 0
        excluded = best(index + 1, limit)
        left, right = ordered[index]
        if right > limit:
            return excluded
        inside = best(bisect_right(starts, left), right - 1)
        following = bisect_left(starts, right + 1)
        return max(excluded, 1 + inside + best(following, limit))

    result = []
    pending = [(0, max((right for _, right in ordered), default=0))]
    while pending:
        index, limit = pending.pop()
        if index >= len(ordered):
            continue
        left, right = ordered[index]
        excluded = best(index + 1, limit)
        following = bisect_left(starts, right + 1)
        included = (1 + best(bisect_right(starts, left), right - 1) +
                    best(following, limit)) if right <= limit else -1
        if included >= excluded:
            result.append([left, right])
            pending.append((following, limit))
            pending.append((bisect_right(starts, left), right - 1))
        else:
            pending.append((index + 1, limit))
    return result


def value(block, category, tag):
    result = block.find_value(f"{category}.{tag}")
    return result.strip() if result else None


def observed_residue_map(dssr, mapping, parents):
    if not dssr.get("nts"):
        raise ValueError("DSSR full nts list is required")
    nucleotides = sorted((nt for nt in dssr["nts"] if nt.get("nt_type") == "RNA"),
                         key=lambda nt: nt["index"])
    by_chain = {}
    for nt in nucleotides:
        chain = parse_nt(nt["nt_id"])[0]
        by_chain.setdefault(chain, []).append(nt)
    lookup = {}
    for chain, nts in by_chain.items():
        if chain not in mapping:
            continue
        observed = sorted((row for row in mapping[chain].values() if row["observed"]),
                          key=lambda row: row["seq_id"])
        if len(observed) != len(nts):
            raise ValueError(f"DSSR/mmCIF observed residue count mismatch: {chain}")
        for nt, row in zip(nts, observed):
            if one_letter(nt["nt_name"], parents) != row["letter"]:
                raise ValueError(f"DSSR/mmCIF observed base mismatch: {nt['nt_id']}")
            if nt["nt_id"] in lookup:
                raise ValueError(f"duplicate DSSR nucleotide id: {nt['nt_id']}")
            lookup[nt["nt_id"]] = row
    return lookup


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
        try:
            if entity_types.get(entity) != "polyribonucleotide":
                raise ValueError("not a pure RNA polymer entity")
            key = int(seq_id)
            letter = one_letter(monomer, parents)
        except (ValueError, TypeError) as error:
            rejected.append({"auth_chain": auth_chain, "reason": str(error)})
            continue
        if key in chains.setdefault(auth_chain, {}):
            raise ValueError(f"duplicate residue mapping: {key}")
        chains[auth_chain][key] = {"asym_id": asym, "entity_id": entity, "seq_id": int(seq_id),
                                   "observed": auth_seq != "?", "letter": letter, "monomer": monomer}
    dssr = json.loads(dssr_path.read_text())
    raw_by_chain = {chain: [] for chain in chains}
    raw_records = {chain: [] for chain in chains}
    errors = []
    nt_lookup = observed_residue_map(dssr, chains, parents)
    for pair in dssr.get("pairs", []):
        nt1, nt2 = parse_nt(pair["nt1"]), parse_nt(pair["nt2"])
        if nt1[0] != nt2[0]:
            continue
        chain = nt1[0]
        if chain not in chains:
            rejected.append({"auth_chain": chain, "reason": "no mapped RNA residues; non-RNA DSSR chain excluded"})
            continue
        try:
            left, right = nt_lookup[pair["nt1"]], nt_lookup[pair["nt2"]]
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
            a, b = sorted((nt_lookup[pair["nt1"]]["seq_id"], nt_lookup[pair["nt2"]]["seq_id"]))
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
            "dssr_seq_id_mode": "observed_nts_v1",
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
