import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_interface_baseline import Probe, loss, residue_batches


class PackedLossTest(unittest.TestCase):
    def test_sample_conservation(self):
        records = [{"x": torch.ones(length, 16), "structure": torch.empty(length, 0), "y": torch.ones(length)} for length in (4, 3, 4, 3, 4, 2)]
        batches = list(residue_batches(records, False, max_rows=7))
        self.assertEqual([len(y) for _, y in batches], [7, 7, 6])
        self.assertTrue(torch.equal(torch.cat([y for _, y in batches]), torch.ones(20)))

    def test_loss_and_gradients_match_per_chain(self):
        for use_structure in (False, True):
            torch.manual_seed(3407)
            records = []
            for _ in range(7):
                records.append({"x": torch.randn(5, 16), "structure": torch.randn(5, 4) if use_structure else torch.empty(5, 0),
                                "y": torch.randint(0, 2, (5,)).float()})
            model = Probe(20 if use_structure else 16).cuda()
            model.zero_grad(set_to_none=True)
            expected = sum(F.binary_cross_entropy_with_logits(model(record, use_structure), record["y"].cuda(), reduction="sum")
                           for record in records) / sum(len(record["y"]) for record in records)
            expected.backward()
            expected_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
            model.zero_grad(set_to_none=True)
            actual = loss(records, model, use_structure)
            actual.backward()
            self.assertTrue(torch.allclose(expected, actual, atol=1e-7, rtol=1e-7))
            for parameter, expected_gradient in zip(model.parameters(), expected_gradients):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.allclose(parameter.grad, expected_gradient, atol=2e-6, rtol=2e-6),
                                (expected.item(), actual.item(), (parameter.grad - expected_gradient).abs().max().item()))
            print("packed_loss_match", use_structure, float(expected), float(actual))


if __name__ == "__main__":
    unittest.main()
