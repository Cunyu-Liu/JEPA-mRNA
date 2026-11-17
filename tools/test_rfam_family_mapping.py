import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_rfam_family_mapping import region_key


class RfamFamilyMappingTest(unittest.TestCase):
    def test_forward_and_reverse_coordinates(self):
        self.assertEqual(region_key("AAAP01003520.1_1076-876.bpseq"), "AAAP01003520.1/1076-876")
        self.assertEqual(region_key("JF974315.1_30902-30955.bpseq"), "JF974315.1/30902-30955")

    def test_non_coordinate_name_is_unmapped(self):
        self.assertIsNone(region_key("bpRNA_CRW_15847.bpseq"))
        self.assertIsNone(region_key("not-a-bpseq-name"))


if __name__ == "__main__":
    unittest.main()
