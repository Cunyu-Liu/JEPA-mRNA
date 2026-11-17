import hashlib
import json
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
THRESHOLD = 5.0


def heavy_atoms(chain):
    positions = []
    residues = [chain] if isinstance(chain, gemmi.Residue) else chain
    for residue in residues:
        for atom in residue:
            if atom.element.name != "H":
                positions.append(atom.pos.tolist())
    return np.asarray(positions, dtype=np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (ROOT / "decontamination/complex_holdout.jsonl").open() if line.strip()]
    by_pdb = defaultdict(list)
    for record in records:
        by_pdb[record["name"].split("_")[0]].append(record)
    destination = OUT / "holdout_contacts.jsonl"
    manifest = {"status": "PASS_INTERFACE_HOLDOUT_LABELS", "threshold_angstrom": THRESHOLD,
                "scientific_scope": "labels only; no interface model result", "entries": {}}
    with destination.open("w") as handle:
        for pdb, pdb_records in sorted(by_pdb.items()):
            path = ROOT / f"entries/{pdb}/{pdb}.cif"
            structure = gemmi.read_structure(str(path))
            model = structure[0]
            available = {chain.name for chain in model}
            peptide_chains = set()
            for entity in structure.entities:
                if str(entity.polymer_type) == "PolymerType.PeptideL":
                    peptide_chains.update(set(entity.subchains) & available)
            if not peptide_chains:
                for record in pdb_records:
                    manifest["entries"][record["name"]] = {"status": "BLOCKED_NO_MODEL_PEPTIDE_CHAIN"}
                continue
            protein = np.concatenate([heavy_atoms(model[name]) for name in peptide_chains])
            for record in pdb_records:
                chain_name = record["source"]["asym_ids"][0]
                if chain_name not in available:
                    manifest["entries"][record["name"]] = {"status": "BLOCKED_RNA_CHAIN_NOT_IN_MODEL"}
                    continue
                rows = []
                for index, residue in enumerate(model[chain_name]):
                    atoms = heavy_atoms(residue)
                    if not len(atoms):
                        continue
                    distance = np.sqrt(((atoms[:, None, :] - protein[None, :, :]) ** 2).sum(axis=-1)).min()
                    rows.append({"residue_index": index, "residue_name": residue.name,
                                 "nearest_angstrom": float(distance), "interface": bool(distance < THRESHOLD)})
                if not rows:
                    raise AssertionError(f"empty interface labels for {record['name']}")
                output = {"name": record["name"], "pdb": pdb, "group": record["group"],
                          "sequence": record["seq"], "chain": chain_name,
                          "contact_definition": "RNA residue to peptide heavy atom <5A",
                          "rna_rna_pairs_used_as_contacts": False,
                          "residues": rows}
                handle.write(json.dumps(output) + "\n")
                manifest["entries"][record["name"]] = {
                    "n_interface": sum(row["interface"] for row in rows),
                    "n_noninterface": sum(not row["interface"] for row in rows),
                }
    manifest["n_records"] = len(records)
    manifest["n_unique_pdbs"] = len(by_pdb)
    manifest["n_blocked"] = sum("status" in value for value in manifest["entries"].values())
    manifest["n_with_interface_and_negative"] = sum(
        value.get("n_interface", 0) > 0 and value.get("n_noninterface", 0) > 0
        for value in manifest["entries"].values())
    manifest["output"] = str(destination)
    manifest["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (OUT / "holdout_contacts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "n_records", "n_unique_pdbs", "n_with_interface_and_negative", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
