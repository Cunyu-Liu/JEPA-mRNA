import json
import tempfile
import unittest
from pathlib import Path

from audit_rna_complex_round49 import source_metadata
from verify_rna_complex_decontamination_round49 import validate_group_partition


class ComplexConservationTests(unittest.TestCase):
    def test_partner_entity_types_exclude_rna_and_keep_protein(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.cif"
            path.write_text("""data_TEST
loop_
_entity_poly.entity_id
_entity_poly.type
1 polyribonucleotide
2 polypeptide(L)
3 polyribonucleotide
#
""")
            metadata = source_metadata(path, {"1"})
            self.assertEqual(metadata["partner_entity_types"], ["polypeptide(L)"])
            self.assertEqual(metadata["source_organisms"], [])

    def test_transitive_clusters_and_entry_join_form_one_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("0\t0\n1\t1\n2\t2\n3\t3\n4\t4\n0\t1\n1\t2\n")
            rows = []
            for index, pdb_id in enumerate(["P0", "P0", "P0", "P0", "P4"]):
                rows.append({"name": f"chain{index}", "candidate_index": index,
                             "group": "g1" if index < 4 else "g2",
                             "source": {"pdb_id": pdb_id}})
            manifest = {"groups": {"g1": {"ids": [f"chain{i}" for i in range(4)]},
                                    "g2": {"ids": ["chain4"]}}}
            validate_group_partition(rows, path, manifest)
            manifest["groups"] = {"g1": {"ids": [f"chain{i}" for i in range(3)]},
                                   "g2": {"ids": ["chain3", "chain4"]}}
            with self.assertRaises(AssertionError):
                validate_group_partition(rows, path, manifest)

    def test_unknown_or_missing_cluster_members_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("0\t0\n0\t1\n")
            rows = [{"name": "chain0", "candidate_index": 0, "group": "g1",
                     "source": {"pdb_id": "P0"}}]
            with self.assertRaises(AssertionError):
                validate_group_partition(rows, path, {"groups": {"g1": {"ids": ["chain0"]}}})


if __name__ == "__main__":
    unittest.main()
