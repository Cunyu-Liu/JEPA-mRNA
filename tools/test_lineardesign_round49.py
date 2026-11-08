import json
import tempfile
import unittest
from pathlib import Path

from prepare_lineardesign_round49 import dbn_pairs, parse_output, select_seed
from summarize_lineardesign_round49 import summarize


STDOUT = """j=0
mRNA sequence:  AUGAACGAUACGGAGGCGAUC
mRNA structure: ......(((.((....)))))
mRNA folding free energy: -1.10 kcal/mol; mRNA CAI: 0.695
"""


class LinearDesignPreparationTests(unittest.TestCase):
    def test_parse_official_output_and_nested_pairs(self):
        result = parse_output(STDOUT)
        self.assertEqual(result["sequence"], "AUGAACGAUACGGAGGCGAUC")
        self.assertEqual(result["pairs"], [[6, 20], [7, 19], [8, 18], [10, 17], [11, 16]])
        self.assertEqual(result["reported_mfe_kcal_mol"], -1.10)
        self.assertEqual(result["reported_cai"], 0.695)

    def test_invalid_structure_and_missing_output_fail_closed(self):
        with self.assertRaises(ValueError):
            dbn_pairs("(.")
        with self.assertRaises(ValueError):
            dbn_pairs("([)]")
        with self.assertRaises(ValueError):
            parse_output("mRNA sequence: AUG\n")

    def test_model_selection_only_uses_vl0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, loss in [(0, 20.0), (1, 18.0), (2, 19.0)]:
                run = root / "runs" / f"arm_s{seed}"
                run.mkdir(parents=True)
                (run / "best.pt").write_bytes(b"checkpoint")
                (run / "run_meta.json").write_text(json.dumps({
                    "status": "completed", "converged": True, "early_stopped": True,
                    "best_validation_loss": loss, "best_step": 1000 + seed,
                    "schedule": {"steps": 0}, "validation": {"patience": 6}}))
            selected = select_seed(root, "complex_d0", "arm_s")
            self.assertEqual(selected["seed"], 1)
            self.assertEqual(selected["best_step"], 1001)
            self.assertNotIn("candidate_f1", selected)

    def test_pair_agreement_is_not_structure_recovery(self):
        rows = [{"name": "example", "seq": "AUGCAU", "pairs": [[0, 5]]}]
        predictions = [{"pred_pairs": [[0, 5], [1, 4]]}]
        result = summarize(predictions, rows)
        self.assertEqual(result["micro_f1_vs_lineardesign_mfe"], 2 / 3)
        self.assertEqual(result["per_candidate"][0]["n_lineardesign_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
