import unittest

import torch

from run_rnaformer import install_query_chunked_attention
from RNAformer.module.axial_attention import Attention2d


class Attention2dChunkTest(unittest.TestCase):
    def test_query_chunks_match_original_exactly(self):
        torch.manual_seed(7)
        attention = Attention2d(16, 4, True, "fp32", False, True, 0.02, 2).eval()
        original = attention.forward
        pair_act = torch.randn(1, 7, 9, 16)
        mask = torch.zeros(1, 7, 9, dtype=torch.bool)
        mask[:, 2, 5] = True
        with torch.no_grad():
            expected = original(pair_act, mask)
        wrapper = type("Wrapper", (), {"modules": lambda self: [attention]})()
        report = install_query_chunked_attention(wrapper, threshold=4, block_size=3)
        self.assertEqual(report["attention2d_modules"], 1)
        with torch.no_grad():
            actual = attention(pair_act, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_short_path_delegates_to_original(self):
        attention = Attention2d(16, 4, True, "fp32", False, True, 0.02, 2).eval()
        original = attention.forward
        pair_act = torch.randn(1, 3, 5, 16)
        with torch.no_grad():
            expected = original(pair_act, None)
        wrapper = type("Wrapper", (), {"modules": lambda self: [attention]})()
        install_query_chunked_attention(wrapper, threshold=4, block_size=2)
        with torch.no_grad():
            actual = attention(pair_act, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
