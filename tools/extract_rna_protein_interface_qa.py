import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/manifests/v2/interface_qa")
THRESHOLD = 5.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    summary = {"status": "PASS_INTERFACE_ASSET_GATE_LIMITED", "threshold_angstrom": THRESHOLD,
               "scientific_scope": "two-entry label/negative QA only; no trained interface result", "entries": {}}
    for pdb in ("10BE", "10FB"):
        path = ROOT / f"entries/{pdb}/{pdb}.cif"
        structure = gemmi.read_structure(str(path))
        model = structure[0]
        entity_types = {}
        for entity in structure.entities:
            for chain_name in entity.subchains:
                entity_types[chain_name] = str(entity.polymer_type)
        rna_chains = [name for name in entity_types if entity_types[name] == "PolymerType.Rna"]
        protein_chains = [name for name in entity_types if entity_types[name] == "PolymerType.PeptideL"]
        if not rna_chains or not protein_chains:
            raise AssertionError(f"missing RNA/protein partner for {pdb}")
        protein = np.concatenate([heavy_atoms(model[name]) for name in protein_chains])
        rows = []
        for chain_name in rna_chains:
            chain = model[chain_name]
            for index, residue in enumerate(chain):
                atoms = heavy_atoms(residue)
                if not len(atoms):
                    continue
                distances = np.sqrt(((atoms[:, None, :] - protein[None, :, :]) ** 2).sum(axis=-1))
                nearest = float(distances.min())
                rows.append({"chain": chain_name, "residue_index": index, "residue_name": residue.name,
                             "nearest_protein_heavy_atom_angstrom": nearest,
                             "interface": nearest < THRESHOLD})
        positives = sum(row["interface"] for row in rows)
        negatives = len(rows) - positives
        if not positives or not negatives:
            raise AssertionError(f"positive/negative definition failed for {pdb}")
        assert all(row["nearest_protein_heavy_atom_angstrom"] < THRESHOLD for row in rows if row["interface"])
        assert all(row["nearest_protein_heavy_atom_angstrom"] >= THRESHOLD for row in rows if not row["interface"])
        report = {"pdb": pdb, "cif_sha256": sha256(path), "rna_chains": rna_chains,
                  "protein_chains": protein_chains, "n_residues": len(rows),
                  "n_interface": positives, "n_noninterface": negatives,
                  "contact_definition": "minimum RNA-residue to peptide heavy-atom distance < 5 Å",
                  "negative_definition": "same RNA chains, minimum distance >= 5 Å",
                  "rna_rna_pairs_used_as_contacts": False, "rows": rows}
        (OUT / f"{pdb}.json").write_text(json.dumps(report, indent=2) + "\n")
        summary["entries"][pdb] = {"n_interface": positives, "n_noninterface": negatives,
                                   "report_sha256": sha256(OUT / f"{pdb}.json")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
