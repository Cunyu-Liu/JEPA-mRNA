import json
import sys
import unittest
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from extract_interface_holdout_dataset import extract_record

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49")


class InterfaceLabelMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        cls.blocks = {}
        for name in ("30GA_1", "12DM_3"):
            lines = (ROOT / "decontamination/complex_holdout.jsonl").read_text().splitlines()
            cls.records[name] = next(json.loads(line) for line in lines if json.loads(line)["name"] == name)
            cls.blocks[name] = gemmi.cif.read(str(ROOT / f"entries/{name[:4]}/{name[:4]}.cif")).sole_block()

    def test_rna_polymer_and_label_seq_mapping(self):
        result = extract_record(self.records["30GA_1"], self.blocks["30GA_1"], "fixture")
        self.assertEqual(len(result["sequence"]), 93)
        self.assertEqual(len(result["residues"]), 60)
        self.assertEqual(result["residues"][0]["residue_index"], 20)
        self.assertEqual(result["residues"][0]["residue_name"], "C")
        self.assertEqual(result["residues"][0]["auth_chain"], "1")
        self.assertTrue(all(row["nearest_partner_label_entity_id"] != "1" for row in result["residues"]))

    def test_whole_rna_not_same_named_protein_chain(self):
        result = extract_record(self.records["12DM_3"], self.blocks["12DM_3"], "fixture")
        self.assertEqual(len(result["sequence"]), 1473)
        self.assertEqual(len(result["residues"]), 1372)
        self.assertEqual(result["residues"][0]["residue_index"], 3)
        self.assertTrue(all(row["residue_name"] in "ACGU" for row in result["residues"]))

    def test_unobserved_residues_are_not_negative_labels(self):
        result = extract_record(self.records["30GA_1"], self.blocks["30GA_1"], "fixture")
        observed = {row["residue_index"] for row in result["residues"]}
        self.assertTrue(set(range(93)) - observed)
        self.assertFalse(observed & (set(range(93)) - observed))

    def test_sequence_mismatch_rejected(self):
        record = dict(self.records["30GA_1"])
        record["seq"] = "A" + record["seq"][1:]
        with self.assertRaisesRegex(ValueError, "REFERENCE_SEQUENCE_MISMATCH"):
            extract_record(record, self.blocks["30GA_1"], "fixture")

    def test_multiple_asym_ids_rejected(self):
        record = dict(self.records["30GA_1"])
        record["source"] = {**record["source"], "asym_ids": ["A", "B"]}
        with self.assertRaisesRegex(ValueError, "MULTIPLE_RNA_ASYM_IDS"):
            extract_record(record, self.blocks["30GA_1"], "fixture")


class ModifiedParentMappingTest(unittest.TestCase):
    def test_modified_residue_uses_exact_parent_and_one_based_label_seq(self):
        lines = (ROOT / "decontamination/complex_train.jsonl").read_text().splitlines()
        record = next(json.loads(line) for line in lines if json.loads(line)["name"] == "11EL_A")
        parents = json.loads((ROOT / "entries/11EL/parents.json").read_text())
        block = gemmi.cif.read(str(ROOT / "entries/11EL/11EL.cif")).sole_block()
        result = extract_record(record, block, parents["cif_sha256"], parents["parents"])
        self.assertEqual(len(result["sequence"]), 1543)
        modified = next(row for row in result["residues"] if row["residue_name"] == "PSU")
        self.assertEqual(modified["residue_index"], 515)
        self.assertEqual(modified["canonical_base"], "U")
        self.assertEqual(result["sequence"][515], "U")
        self.assertTrue(all(1 <= left < right <= len(record["seq"]) for left, right in record["pairs"]))


class IndependentCifReadTest(unittest.TestCase):
    def test_quoted_polypeptide_type_is_normalized(self):
        from verify_interface_labels import rows
        block = gemmi.cif.read(str(ROOT / "entries/10FZ/10FZ.cif")).sole_block()
        entity_types = {row["entity_id"]: row["type"] for row in rows(
            block, "_entity_poly.", ["entity_id", "type"])}
        self.assertEqual(entity_types["1"], "polyribonucleotide")
        self.assertEqual(entity_types["2"], "polypeptide(L)")


if __name__ == "__main__":
    unittest.main()
