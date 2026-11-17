import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

DATA = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
TAGS = ["group_PDB", "label_asym_id", "label_entity_id", "label_seq_id", "label_comp_id",
        "label_atom_id", "label_alt_id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(block, category, names):
    return [{name: row.str(index) for index, name in enumerate(names)}
            for row in block.find(category, names)]


def parent_base(monomer, parents):
    if monomer in {"A", "C", "G", "U"}:
        return monomer
    info = gemmi.find_tabulated_residue(monomer)
    letter = info.one_letter_code.upper() if info else ""
    if letter not in {"A", "C", "G", "U"}:
        letter = parents.get(monomer, {}).get("parent", "")
    if letter not in {"A", "C", "G", "U"}:
        raise ValueError(f"unsupported modified RNA residue: {monomer}")
    return letter


def main():
    source = {}
    output = {}
    qa = {"status": "PASS_INDEPENDENT_INTERFACE_LABEL_QA", "scientific_scope": "raw-mmCIF coordinate, identity, partner, distance and split QA; not model performance", "splits": {}}
    for split in ("train", "holdout"):
        source[split] = {r["name"]: r for r in map(json.loads, (DATA / f"decontamination/complex_{split}.jsonl").open())}
        output[split] = {r["name"]: r for r in map(json.loads, (ROOT / f"{split}_contacts.jsonl").open())}
    for split in ("train", "holdout"):
        by_pdb = defaultdict(list)
        for record in output[split].values():
            by_pdb[record["pdb"]].append(record)
        checked_atoms = 0
        checked_residues = 0
        single_sign = []
        for pdb, records in sorted(by_pdb.items()):
            path = DATA / f"entries/{pdb}/{pdb}.cif"
            block = gemmi.cif.read(str(path)).sole_block()
            parent_document = json.loads(path.with_name("parents.json").read_text())
            assert parent_document["cif_sha256"] == digest(path)
            parents = parent_document["parents"]
            label_path = path.with_name("labels.json")
            label_document = json.loads(label_path.read_text())
            polymers = {r["entity_id"]: r["type"] for r in rows(block, "_entity_poly.", ["entity_id", "type"])}
            peptide_chains = {r["asym_id"] for r in rows(block, "_pdbx_poly_seq_scheme.", ["asym_id", "entity_id"])
                              if polymers.get(r["entity_id"]) in ("polypeptide(L)", "polypeptide(D)")}
            schemes = defaultdict(list)
            for row in rows(block, "_pdbx_poly_seq_scheme.", ["asym_id", "seq_id", "mon_id"]):
                schemes[row["asym_id"]].append(row)
            protein = []
            rna = defaultdict(list)
            for atom in rows(block, "_atom_site.", TAGS):
                if atom["pdbx_PDB_model_num"] != "1" or atom["type_symbol"].upper() == "H":
                    continue
                if atom["label_alt_id"] not in ("", ".", "?", "A"):
                    continue
                xyz = np.array([float(atom["Cartn_x"]), float(atom["Cartn_y"]), float(atom["Cartn_z"])])
                if atom["label_asym_id"] in peptide_chains:
                    protein.append((atom, xyz))
                elif atom["label_seq_id"] not in (".", "?"):
                    rna[atom["label_asym_id"]].append((atom, xyz))
            assert protein
            tree = cKDTree(np.stack([xyz for _, xyz in protein]))
            checked_atoms += len(protein)
            for record in records:
                original = source[split][record["name"]]
                assert record["group"] == original["group"] and record["sequence"] == original["seq"]
                assert record["label_asym_id"] in original["source"]["asym_ids"]
                assert not record["rna_rna_pairs_used_as_contacts"]
                assert digest(path) == record["cif_sha256"]
                assert digest(label_path) == original["source"]["source_json_sha256"]
                source_chain = next(chain for chain in label_document["chains"] if chain["id"] == original["source"]["id"])
                assert source_chain["sequence"] == original["source"]["sequence"] == original["seq"]
                assert all(1 <= left < right <= len(original["seq"]) for left, right in original["pairs"])
                scheme = sorted(schemes[record["label_asym_id"]], key=lambda row: int(row["seq_id"]))
                assert "".join(parent_base(row["mon_id"], parents) for row in scheme) == original["seq"]
                observed = defaultdict(list)
                for atom, xyz in rna[record["label_asym_id"]]:
                    observed[int(atom["label_seq_id"]) - 1].append((atom, xyz))
                assert len(observed) == len(record["residues"])
                for saved in record["residues"]:
                    index = saved["residue_index"]
                    assert 0 <= index < len(original["seq"])
                    atoms = observed[index]
                    assert all(atom["label_comp_id"] == scheme[index]["mon_id"] for atom, _ in atoms)
                    assert saved["residue_name"] == scheme[index]["mon_id"]
                    assert saved["canonical_base"] == original["seq"][index]
                    distances, indices = tree.query(np.stack([xyz for _, xyz in atoms]))
                    distances = np.asarray(distances).reshape(-1)
                    indices = np.asarray(indices).reshape(-1)
                    nearest = int(np.argmin(distances))
                    expected_distance = float(distances[nearest])
                    assert abs(expected_distance - saved["nearest_angstrom"]) < 1e-8
                    assert saved["interface"] == (expected_distance < 5.0)
                    partner = protein[int(indices[nearest])][0]
                    assert saved["nearest_partner_label_asym_id"] == partner["label_asym_id"]
                    assert saved["nearest_partner_label_entity_id"] in polymers
                    assert polymers[saved["nearest_partner_label_entity_id"]] in ("polypeptide(L)", "polypeptide(D)")
                    checked_residues += 1
                missing = set(range(len(original["seq"]))) - set(observed)
                assert missing.isdisjoint(row["residue_index"] for row in record["residues"])
                signs = {row["interface"] for row in record["residues"]}
                if len(signs) == 1:
                    single_sign.append(record["name"])
        manifest = json.loads((ROOT / f"{split}_contacts_manifest.json").read_text())
        assert manifest["n_records"] == len(source[split])
        assert manifest["n_valid"] == len(output[split])
        assert manifest["output_sha256"] == digest(ROOT / f"{split}_contacts.jsonl")
        assert len(manifest["entries"]) == len(source[split])
        assert set(output[split]) == {name for name, value in manifest["entries"].items() if "status" not in value}
        report = {
            "n_source": len(source[split]), "n_valid": len(output[split]), "n_blocked": manifest["n_blocked"],
            "n_observed_residues": checked_residues, "n_peptide_atoms_scanned": checked_atoms,
            "n_positive": sum(row["interface"] for r in output[split].values() for row in r["residues"]),
            "n_negative": sum(not row["interface"] for r in output[split].values() for row in r["residues"]),
            "single_sign_chains": single_sign,
            "experimental_methods": dict(Counter(method for r in output[split].values() for method in r["experimental_methods"])),
            "missing_resolution": sum(r["resolution_angstrom"] is None for r in output[split].values()),
            "blocked_reasons": dict(Counter(e["status"] for e in manifest["entries"].values() if "status" in e)),
            "contacts_sha256": digest(ROOT / f"{split}_contacts.jsonl"),
        }
        (ROOT / f"{split}_label_qa.json").write_text(json.dumps(report, indent=2) + "\n")
        qa["splits"][split] = report
    train = output["train"].values()
    holdout = output["holdout"].values()
    qa["train_holdout_group_overlap"] = sorted({r["group"] for r in train} & {r["group"] for r in holdout})
    qa["train_holdout_pdb_overlap"] = sorted({r["pdb"] for r in train} & {r["pdb"] for r in holdout})
    qa["train_holdout_exact_sequence_overlap"] = len({r["sequence"] for r in train} & {r["sequence"] for r in holdout})
    assert not qa["train_holdout_group_overlap"] and not qa["train_holdout_pdb_overlap"]
    (ROOT / "label_qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
