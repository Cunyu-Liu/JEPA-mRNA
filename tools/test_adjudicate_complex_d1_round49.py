import unittest

from adjudicate_complex_d1_round49 import route_assessment


class ComplexDecisionTests(unittest.TestCase):
    def row(self, split, delta, p=0.01, small=False):
        result = {"split": split, "delta_micro_f1": delta, "p_two_sided": p,
                  "status": "PASS", "small_sample_descriptive": small}
        if split in {"bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb"}:
            result["p_holm_main8"] = p
        return result

    def test_positive_b_without_main_regression(self):
        assessment = route_assessment([self.row("rfam_fam", 0.03), self.row("bprna_ts0", 0.005, 0.3)])
        self.assertTrue(assessment["b_group_improvement"])
        self.assertTrue(assessment["meets_preregistered_b_route"])

    def test_main_regression_or_small_b_is_not_success(self):
        self.assertFalse(route_assessment([self.row("rfam_fam", 0.03), self.row("bprna_ts0", -0.03)])["meets_preregistered_b_route"])
        self.assertFalse(route_assessment([self.row("rfam_fam", 0.03, small=True)])["b_group_improvement"])

    def test_mixed_b_direction_is_not_general_improvement(self):
        rows = [self.row("rfam_fam", 0.03), self.row("chanrg_test", -0.03)]
        assessment = route_assessment(rows)
        self.assertTrue(assessment["significant_negative_b"])
        self.assertFalse(assessment["meets_preregistered_b_route"])


if __name__ == "__main__":
    unittest.main()
