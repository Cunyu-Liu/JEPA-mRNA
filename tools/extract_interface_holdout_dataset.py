import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
THRESHOLD = 5.0
ATOM_FIELDS = [
    "label_asym_id", "label_entity_id", "label_seq_id", "label_comp_id",
    "auth_asym_id", "auth_seq_id", "label_atom_id", "label_alt_id", "type_symbol",
    "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parents(cif_path, cif_sha256):
    document = json.loads(cif_path.with_name("parents.json").read_text())
    if document["cif_sha256"] != cif_sha256:
        raise ValueError("PARENT_MAP_CIF_HASH_MISMATCH")
    return document["parents"]


def canonical_base(monomer, parents):
    if monomer in {"A", "C", "G", "U"}:
        return monomer
    info = gemmi.find_tabulated_residue(monomer)
    letter = info.one_letter_code.upper() if info else ""
    if letter not in {"A", "C", "G", "U"}:
        letter = parents.get(monomer, {}).get("parent", "")
    if letter not in {"A", "C", "G", "U"}:
        raise ValueError(f"UNSUPPORTED_MODIFIED_RNA_RESIDUE_{monomer}")
    return letter


def scheme_rows(block, asym_id):
    table = block.find("_pdbx_poly_seq_scheme.", ["asym_id", "entity_id", "seq_id", "mon_id", "pdb_strand_id"])
    rows = [{"asym_id": row.str(0), "entity_id": row.str(1), "seq_id": int(row.str(2)),
             "mon_id": row.str(3), "pdb_strand_id": row.str(4)} for row in table if row.str(0) == asym_id]
    rows.sort(key=lambda row: row["seq_id"])
    if not rows or [row["seq_id"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("INVALID_OR_INCOMPLETE_POLY_SEQ_SCHEME")
    return rows


def model_atoms(block):
    columns = {name: [row.str(i) for row in block.find("_atom_site.", ATOM_FIELDS)]
               for i, name in enumerate(ATOM_FIELDS)}
    for i in range(len(columns["label_asym_id"])):
        yield {name: values[i] for name, values in columns.items()}


def experimental_metadata(block):
    methods = [str(row.str(0)) for row in block.find("_exptl.", ["method"])]
    resolution = block.find_value("_refine.ls_d_res_high")
    if resolution in (None, "?", "."):
        resolution = block.find_value("_em_3d_reconstruction.resolution")
    return {"experimental_methods": methods,
            "resolution_angstrom": None if resolution in (None, "?", ".") else float(resolution)}


def extract_record(record, block, cif_sha256, parents=None):
    parents = parents or {}
    asym_ids = record["source"]["asym_ids"]
    if len(asym_ids) != 1:
        raise ValueError("MULTIPLE_RNA_ASYM_IDS_NOT_SUPPORTED")
    asym_id = asym_ids[0]
    scheme = scheme_rows(block, asym_id)
    entities = {row.str(0): str(row.str(1)) for row in block.find("_entity_poly.", ["entity_id", "type"])}
    if entities.get(scheme[0]["entity_id"]) != "polyribonucleotide":
        raise ValueError("SOURCE_IS_NOT_POLYRIBONUCLEOTIDE")
    sequence = "".join(canonical_base(row["mon_id"], parents) for row in scheme)
    if sequence != record["seq"]:
        raise ValueError("REFERENCE_SEQUENCE_MISMATCH")
    protein_asyms = {}
    for row in block.find("_pdbx_poly_seq_scheme.", ["asym_id", "entity_id"]):
        if entities.get(row.str(1)) in ("polypeptide(L)", "polypeptide(D)"):
            protein_asyms[row.str(0)] = row.str(1)
    if not protein_asyms:
        raise ValueError("NO_PEPTIDE_POLYMER")
    protein_atoms = []
    rna_atoms = defaultdict(list)
    for atom in model_atoms(block):
        if atom["pdbx_PDB_model_num"] != "1" or atom["type_symbol"].upper() == "H":
            continue
        if atom["label_alt_id"] not in ("", ".", "?", "A"):
            continue
        position = np.asarray([float(atom[f"Cartn_{axis}"]) for axis in "xyz"], dtype=np.float64)
        if atom["label_asym_id"] == asym_id:
            if atom["label_entity_id"] != scheme[0]["entity_id"] or atom["label_seq_id"] in (".", "?"):
                raise ValueError("RNA_ATOM_OUTSIDE_SOURCE_POLYMER")
            rna_atoms[int(atom["label_seq_id"])].append((atom, position))
        elif atom["label_asym_id"] in protein_asyms:
            protein_atoms.append((atom, position))
    if not rna_atoms or not protein_atoms:
        raise ValueError("NO_OBSERVED_RNA_OR_PEPTIDE_ATOMS")
    tree = cKDTree(np.stack([position for _, position in protein_atoms]))
    observed = set()
    rows = []
    for label_seq, atoms in sorted(rna_atoms.items()):
        if not 1 <= label_seq <= len(scheme):
            raise ValueError("RNA_LABEL_SEQ_OUT_OF_RANGE")
        if label_seq in observed:
            raise AssertionError("duplicate label_seq")
        observed.add(label_seq)
        monomer = scheme[label_seq - 1]["mon_id"]
        if any(atom["label_comp_id"] != monomer for atom, _ in atoms):
            raise ValueError("COORDINATE_SEQUENCE_MONOMER_MISMATCH")
        distances, indices = tree.query(np.stack([position for _, position in atoms]), k=1)
        nearest_index = int(np.argmin(distances))
        distance = float(np.asarray(distances).reshape(-1)[nearest_index])
        partner_atom, _ = protein_atoms[int(np.asarray(indices).reshape(-1)[nearest_index])]
        rows.append({
            "residue_index": label_seq - 1, "residue_name": monomer,
            "canonical_base": canonical_base(monomer, parents),
            "auth_chain": atoms[0][0]["auth_asym_id"], "auth_seq_id": atoms[0][0]["auth_seq_id"],
            "nearest_angstrom": distance, "interface": bool(distance < THRESHOLD),
            "nearest_partner_label_asym_id": partner_atom["label_asym_id"],
            "nearest_partner_label_entity_id": partner_atom["label_entity_id"],
            "nearest_partner_residue": partner_atom["label_comp_id"],
            "nearest_partner_atom": partner_atom["label_atom_id"],
        })
    return {
        "name": record["name"], "pdb": record["name"].split("_")[0], "group": record["group"],
        "sequence": sequence, "label_asym_id": asym_id, "chain": asym_id,
        "auth_chains": sorted({row["pdb_strand_id"] for row in scheme}),
        "reference_policy": "exact full-polymer label_seq alignment with hash-verified modified-parent normalization",
        "contact_definition": "deposited model 1 observed RNA polymer residue to peptide polymer heavy atom <5A",
        "negative_definition": "observed residue in same RNA polymer with distance >=5A; unobserved residues excluded",
        "rna_rna_pairs_used_as_contacts": False, "cif_sha256": cif_sha256,
        **experimental_metadata(block), "residues": rows,
    }


def merge(split, shard_count):
    expected = {json.loads(line)["name"]: json.loads(line)
                for line in (ROOT / f"decontamination/complex_{split}.jsonl").open() if line.strip()}
    records = []
    entries = {}
    for index in range(shard_count):
        prefix = OUT / f"{split}_contacts.part{index}"
        records.extend(json.loads(line) for line in Path(str(prefix) + ".jsonl").open())
        part = json.loads(Path(str(prefix) + ".manifest.json").read_text())
        assert part["shard_index"] == index and part["shard_count"] == shard_count
        assert not entries.keys() & part["entries"].keys()
        entries.update(part["entries"])
    assert {record["name"] for record in records} == {name for name, entry in entries.items() if "status" not in entry}
    assert len(records) + sum("status" in entry for entry in entries.values()) == len(expected)
    records.sort(key=lambda record: (record["pdb"], record["name"]))
    destination = OUT / f"{split}_contacts.jsonl"
    with destination.open("w") as handle:
        for record in records:
            assert record["name"] in expected and len(record["residues"]) > 0
            handle.write(json.dumps(record) + "\n")
    manifest = {"status": f"PASS_INTERFACE_{split.upper()}_LABEL_MAPPING", "threshold_angstrom": THRESHOLD,
                "scientific_scope": "coordinate/sequence verified labels; not an interface model result",
                "n_records": len(expected), "n_valid": len(records),
                "n_blocked": sum("status" in entry for entry in entries.values()),
                "n_with_interface_and_negative": sum(any(row["interface"] for row in record["residues"])
                    and any(not row["interface"] for row in record["residues"]) for record in records),
                "entries": entries, "output": str(destination), "output_sha256": sha256(destination)}
    (OUT / f"{split}_contacts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "n_records", "n_valid", "n_blocked", "output_sha256")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["holdout", "train"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args.split, args.shard_count)
        return
    records = [json.loads(line) for line in (ROOT / f"decontamination/complex_{args.split}.jsonl").open() if line.strip()]
    by_pdb = defaultdict(list)
    for record in records:
        by_pdb[record["name"].split("_")[0]].append(record)
    pdbs = sorted(by_pdb)[args.shard_index::args.shard_count]
    prefix = OUT / f"{args.split}_contacts.part{args.shard_index}"
    entries = {}
    with Path(str(prefix) + ".jsonl").open("w") as handle:
        for pdb in pdbs:
            path = ROOT / f"entries/{pdb}/{pdb}.cif"
            digest = sha256(path)
            parents = load_parents(path, digest)
            block = gemmi.cif.read(str(path)).sole_block()
            label_path = path.with_name("labels.json")
            label_document = json.loads(label_path.read_text())
            for record in by_pdb[pdb]:
                try:
                    assert sha256(label_path) == record["source"]["source_json_sha256"]
                    chain = next(chain for chain in label_document["chains"] if chain["id"] == record["source"]["id"])
                    assert chain["sequence"] == record["seq"] == record["source"]["sequence"]
                    output = extract_record(record, block, digest, parents)
                    handle.write(json.dumps(output) + "\n")
                    entries[record["name"]] = {
                        "pdb": pdb, "cif_sha256": digest, "n_reference_residues": len(record["seq"]),
                        "n_observed_residues": len(output["residues"]),
                        "n_interface": sum(row["interface"] for row in output["residues"]),
                        "n_noninterface": sum(not row["interface"] for row in output["residues"]),
                        "experimental_methods": output["experimental_methods"],
                        "resolution_angstrom": output["resolution_angstrom"]}
                except (ValueError, AssertionError) as error:
                    entries[record["name"]] = {"pdb": pdb, "cif_sha256": digest,
                                               "status": f"BLOCKED_{error}"}
    manifest = {"shard_index": args.shard_index, "shard_count": args.shard_count,
                "n_source_records": sum(len(by_pdb[pdb]) for pdb in pdbs), "entries": entries}
    Path(str(prefix) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"split": args.split, "shard": args.shard_index, "source": manifest["n_source_records"],
                      "valid": sum("status" not in entry for entry in entries.values())}, indent=2))


if __name__ == "__main__":
    main()
