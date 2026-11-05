import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import calibrate_xens2_round39 as calibration
from rnajepa.harness import valid_pair_mask


class Xens2CalibrationTest(unittest.TestCase):
    def test_flatten_example_uses_strict_upper_valid_pairs(self):
        seq = "AUGCAU"
        gt = {(0, 5)}
        scores = np.arange(len(seq) ** 2, dtype=np.float64).reshape(len(seq), len(seq))
        flat_scores, labels, mask = calibration.flatten_example(seq, gt, scores)
        self.assertTrue(np.array_equal(mask, valid_pair_mask(seq)))
        self.assertEqual(labels.sum(), 1)
        self.assertEqual(len(flat_scores), np.triu(mask, k=1).sum())
        self.assertEqual(labels[np.flatnonzero(np.triu(mask, k=1)).tolist().index(5)], 1)

    def test_confidence_uses_pair_positions_not_flat_indices(self):
        mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
        confidence = calibration.confidence_for_pairs({(0, 3), (1, 2)}, np.arange(6.0), mask)
        self.assertEqual(confidence, 2.5)
        self.assertEqual(calibration.confidence_for_pairs(set(), np.arange(6.0), mask), 0.0)

    def test_locked_fit_requires_raw_cuda_vl0(self):
        payload = {"status": "PASS_VL0_FIT_ONLY", "data_sha256": "vl0-sha",
                   "protocol": {"device_type": "cuda", "cached_embeddings": False},
                   "vl0_fit": {"fit_split": "bprna_vl0", "fit_data_sha256": "vl0-sha",
                               "objective": "nll", "a": 1.0, "b": 0.0}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(calibration.load_locked_fit(path, "vl0-sha")["a"], 1.0)
            invalid = copy.deepcopy(payload)
            invalid["protocol"]["cached_embeddings"] = True
            path.write_text(json.dumps(invalid))
            with self.assertRaises(AssertionError):
                calibration.load_locked_fit(path, "vl0-sha")
            invalid = copy.deepcopy(payload)
            invalid["vl0_fit"]["fit_split"] = "bprna_ts0"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(AssertionError):
                calibration.load_locked_fit(path, "vl0-sha")

    def test_full_mocked_pipeline_preserves_raw_prediction_and_writes_independent_arrays(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "bprna_vl0.jsonl"
            data.write_text(json.dumps({"name": "one", "seq": "AUGCAU", "pairs": [[0, 5]]}) + "\n")
            out = root / "result.json"
            parameter = SimpleNamespace(is_cuda=True)
            model = SimpleNamespace(parameters=lambda: iter([parameter]))
            models = {key: model for key in ["enc0", "enc1", "base_encoder", "head0", "head1", "r2d_head"]}
            weights = calibration.production.R2D_CKPT
            weights_sha = calibration.sha256(weights)
            models.update({"name": "xens2", "report": {
                "weights": str(weights), "weights_sha256": weights_sha,
                "n_tensors_in_checkpoint": 504, "n_encoder_tensors_loaded": 498,
                "expected_absent": {"pooler.dense.weight": "unused", "pooler.dense.bias": "unused"},
                "r2d_head": {"strict_load": True}}})
            scores = np.full((6, 6), -1.0)
            scores[0, 5] = scores[5, 0] = 3.0
            argv = ["calibrate_xens2_round39.py", "--fit-vl0", "--data", str(data), "--out", str(out)]
            with patch.object(sys, "argv", argv), \
                    patch.object(calibration.torch.cuda, "is_available", return_value=True), \
                    patch.object(calibration.production, "load_models", return_value=models), \
                    patch.object(calibration.production, "forward_our", return_value=np.array([scores])) as forward, \
                    patch.object(calibration, "fit_platt_scaling", return_value=(1.0, 0.0)):
                calibration.main()
                calibration.main()
                self.assertEqual(forward.call_count, 3)
            result = json.loads(out.read_text())
            self.assertEqual(result["f1_runs"], [1.0, 1.0, 1.0])
            self.assertEqual(result["per_sequence"][0]["pred_pairs"], [[0, 5]])
            self.assertEqual(result["raw_confidence"], result["calibrated_confidence"])
            with np.load(out.with_suffix(".npz"), allow_pickle=False) as arrays:
                self.assertTrue(np.array_equal(arrays["raw_probabilities"], arrays["calibrated_probabilities"]))
                self.assertEqual(arrays["labels"].sum(), 1)
            from verify_xens2_calibration_round39 import validate_result
            from numpy.lib.npyio import NpzFile
            original_getitem = NpzFile.__getitem__
            reads = {}
            def counted_getitem(archive, key):
                reads[key] = reads.get(key, 0) + 1
                return original_getitem(archive, key)
            with patch.object(NpzFile, "__getitem__", counted_getitem):
                verified = validate_result(out, require_full_vl0=False)
            self.assertEqual(verified["n_sequences"], 1)
            self.assertEqual(verified["strict_micro_f1"], 1.0)
            self.assertTrue(all(count == 1 for count in reads.values()))


if __name__ == "__main__":
    unittest.main()
