import unittest
from unittest.mock import patch

from speed_xens2_adaptive_round39 import batch_for_length, parse_plan, run_once_adaptive


class AdaptiveSpeedTest(unittest.TestCase):
    def test_plan_boundaries_and_rejections(self):
        plan = parse_plan("256:64,384:32,600:16")
        self.assertEqual([batch_for_length(plan, length) for length in [1, 256, 257, 384, 385, 600]],
                         [64, 64, 32, 32, 16, 16])
        with self.assertRaises(ValueError):
            batch_for_length(plan, 601)
        with self.assertRaises(AssertionError):
            parse_plan("256:64,128:32")

    def test_mocked_batches_preserve_input_order(self):
        records = [{"name": "long", "seq": "A" * 6, "gt": set()},
                   {"name": "short", "seq": "AA", "gt": set()},
                   {"name": "mid", "seq": "AAAA", "gt": set()}]
        calls = []
        def forward(models, seqs, device):
            calls.append([len(seq) for seq in seqs])
            return seqs
        with patch("speed_xens2_adaptive_round39.production.forward_our", side_effect=forward), \
                patch("speed_xens2_adaptive_round39.production.decode_our",
                      side_effect=lambda scores, seqs: [set() for _ in seqs]), \
                patch("speed_xens2_adaptive_round39.torch.cuda.synchronize"):
            result = run_once_adaptive({"name": "xens2"}, records, [(2, 2), (4, 1), (6, 1)], None)
        self.assertEqual(calls, [[2], [4], [6]])
        self.assertEqual(result["predictions"], [[], [], []])
        self.assertEqual(result["n_timed"], 2)


if __name__ == "__main__":
    unittest.main()
