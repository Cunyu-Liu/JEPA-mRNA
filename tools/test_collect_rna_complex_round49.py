import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect_rna_complex_round49 import collect


class CollectRnaComplexTests(unittest.TestCase):
    def test_pair_only_cache_is_preserved_and_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / "1EHZ"
            entry.mkdir()
            (entry / "1EHZ.cif").write_text("mmCIF fixture")
            (entry / "dssr.json").write_text('{"pairs": []}')
            (entry / "error.txt").write_text("old failure")

            def run(command, **kwargs):
                if command[0] == "dssr":
                    self.assertIn("--more", command)
                    self.assertNotIn("--pair-only", command)
                    output = Path(command[-1].split("=", 1)[1])
                    self.assertEqual(output.suffix, ".part")
                    output.write_text('{"nts": [], "pairs": []}')
                    return subprocess.CompletedProcess(command, 0, "", "")
                if "prepare_rna_complex_round49.py" in command[1]:
                    output = Path(command[command.index("--output") + 1])
                    output.write_text(json.dumps({"chains": [], "rejected_chains": [],
                                                  "dssr_seq_id_mode": "observed_nts_v1"}))
                    return subprocess.CompletedProcess(command, 0, "", "")
                self.assertIn("fetch_rna_parent_map_round49.py", command[1])
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("collect_rna_complex_round49.subprocess.run", side_effect=run):
                pdb_id, result = collect("1EHZ", root, "python", "dssr")
            self.assertEqual(pdb_id, "1EHZ")
            self.assertEqual(result["accepted_chains"], 0)
            self.assertEqual(json.loads((entry / "dssr.pair_only.json").read_text()), {"pairs": []})
            self.assertIn("nts", json.loads((entry / "dssr.json").read_text()))
            self.assertFalse((entry / "dssr.json.part").exists())
            self.assertFalse((entry / "error.txt").exists())

    def test_full_dssr_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / "1EHZ"
            entry.mkdir()
            (entry / "1EHZ.cif").write_text("mmCIF fixture")
            (entry / "dssr.json").write_text('{"nts": [], "pairs": []}')

            def run(command, **kwargs):
                self.assertNotEqual(command[0], "dssr")
                if "prepare_rna_complex_round49.py" in command[1]:
                    output = Path(command[command.index("--output") + 1])
                    output.write_text(json.dumps({"chains": [], "rejected_chains": [],
                                                  "dssr_seq_id_mode": "observed_nts_v1"}))
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("collect_rna_complex_round49.subprocess.run", side_effect=run):
                collect("1EHZ", root, "python", "dssr")
            self.assertFalse((entry / "dssr.pair_only.json").exists())


if __name__ == "__main__":
    unittest.main()
