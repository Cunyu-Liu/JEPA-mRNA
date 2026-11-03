import unittest

from aggregate_own_clean_leaderboard_round39 import BLOCKED, asset_identity


class OwnCleanLeaderboardTest(unittest.TestCase):
    def test_asset_identity_variants(self):
        result = {"weights_sha256": "weights", "binary_sha256": "binary", "parameters_sha256": "parameters",
                  "protocol": {"model_report": {"r2d_head": {"weights_sha256": "head"}}}}
        self.assertEqual(asset_identity(result), {"weights_sha256": "weights", "binary_sha256": "binary",
                                                  "parameters_sha256": "parameters", "r2d_head_sha256": "head"})

    def test_model_report_and_commit(self):
        result = {"commit": "commit", "protocol": {"model_report": {"weights_sha256": "weights"}}}
        self.assertEqual(asset_identity(result), {"weights_sha256": "weights", "repository_commit": "commit"})

    def test_blocked_has_no_fake_metric(self):
        self.assertIn("efold", BLOCKED)
        self.assertTrue(BLOCKED["efold"].startswith("BLOCKED_PROVENANCE"))


if __name__ == "__main__":
    unittest.main()
