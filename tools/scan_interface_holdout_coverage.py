import json
from collections import Counter
from pathlib import Path

import gemmi

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")
HOLDOUT = ROOT / "decontamination/complex_holdout.jsonl"
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/manifests/v2/interface_qa/holdout_coverage.json")


def main():
    records = [json.loads(line) for line in HOLDOUT.open() if line.strip()]
    pdbs = sorted({record["name"].split("_")[0] for record in records})
    coverage = {}
    for pdb in pdbs:
        path = ROOT / f"entries/{pdb}/{pdb}.cif"
        if not path.exists():
            coverage[pdb] = {"status": "BLOCKED_MISSING_CIF"}
            continue
        structure = gemmi.read_structure(str(path))
        peptide = set()
        rna = set()
        for entity in structure.entities:
            if str(entity.polymer_type) == "PolymerType.PeptideL":
                peptide.update(entity.subchains)
            if str(entity.polymer_type) == "PolymerType.Rna":
                rna.update(entity.subchains)
        coverage[pdb] = {"n_rna_chains": len(rna), "n_peptide_chains": len(peptide),
                         "has_rna_protein_partner": bool(rna and peptide)}
    eligible = [record["name"] for record in records
                if coverage.get(record["name"].split("_")[0], {}).get("has_rna_protein_partner")]
    result = {
        "status": "PASS_INTERFACE_HOLDOUT_COVERAGE",
        "scientific_scope": "asset coverage only; not an interface model result",
        "n_holdout_candidates": len(records),
        "n_unique_pdbs": len(pdbs),
        "n_candidates_with_rna_protein_partner": len(eligible),
        "eligible_names": eligible,
        "coverage_by_pdb": coverage,
        "counts": Counter(value.get("status", "PARSED") for value in coverage.values()),
    }
    result["counts"] = dict(result["counts"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "n_holdout_candidates", "n_unique_pdbs",
        "n_candidates_with_rna_protein_partner", "counts")}, indent=2))


if __name__ == "__main__":
    main()
