import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_rfam_family_by_sequence import resolve_sequence_family


class SequenceFamilyTest(unittest.TestCase):
    def test_same_family_across_regions(self):
        regions = {"A/1-4": {("RF00001", "full", "1")}, "B/5-8": {("RF00001", "full", "1")}}
        sequence_regions = {"ACGU": {"A/1-4", "B/5-8"}}
        status, keys, families = resolve_sequence_family("ACGU", sequence_regions, regions)
        self.assertEqual(status, "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY")
        self.assertEqual(len(keys), 2)
        self.assertEqual(families, ["RF00001"])

    def test_ambiguous_and_missing_sequences(self):
        regions = {"A/1-4": {("RF00001", "full", "1")}, "B/5-8": {("RF00002", "full", "1")}}
        sequence_regions = {"ACGU": {"A/1-4", "B/5-8"}}
        self.assertEqual(resolve_sequence_family("ACGU", sequence_regions, regions)[0], "AMBIGUOUS_EXACT_SEQUENCE_FAMILIES")
        self.assertEqual(resolve_sequence_family("NNNN", sequence_regions, regions)[0], "NOT_EXACT_IN_RFAM14.9")
        self.assertEqual(resolve_sequence_family("AAAA", {"AAAA": {"C/1-4"}}, regions)[0], "EXACT_IN_FASTA_NO_FAMILY_METADATA")


if __name__ == "__main__":
    unittest.main()
