import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fetch_rna_parent_map_round49


class ParentMapTests(unittest.TestCase):
    def run_lookup(self, response):
        with tempfile.TemporaryDirectory() as temporary:
            cif = Path(temporary) / "input.cif"
            output = Path(temporary) / "parents.json"
            cif.write_text("fixture")
            block = MagicMock()
            block.find.side_effect = lambda category, fields: (
                [["1", "polyribonucleotide"]] if category == "_entity_poly." else [["1", "N"]])
            arguments = ["fetch_rna_parent_map_round49.py", "--cif", str(cif), "--output", str(output)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(fetch_rna_parent_map_round49.gemmi.cif, "read") as read, \
                 patch.object(fetch_rna_parent_map_round49.gemmi, "find_tabulated_residue", return_value=None), \
                 patch.object(fetch_rna_parent_map_round49.requests, "get", return_value=response):
                read.return_value.sole_block.return_value = block
                fetch_rna_parent_map_round49.main()
            return json.loads(output.read_text())

    def test_http_failure_remains_retryable(self):
        response = MagicMock(status_code=503)
        with self.assertRaisesRegex(SystemExit, "transient"):
            self.run_lookup(response)

    def test_permanently_unknown_parent_is_recorded(self):
        response = MagicMock(status_code=200, content=b"{}")
        response.json.return_value = {"chem_comp": {"type": "RNA linking", "one_letter_code": "N"}}
        result = self.run_lookup(response)
        self.assertEqual(result["failures"], [{"monomer": "N", "reason": "no unambiguous RNA parent"}])


if __name__ == "__main__":
    unittest.main()
