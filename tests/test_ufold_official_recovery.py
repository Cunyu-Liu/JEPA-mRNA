#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_ufold_official_bprna_new.py"
SPEC = importlib.util.spec_from_file_location("ufold_official_recovery", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairContractTest(unittest.TestCase):
    def test_canonical_min_loop_and_zero_based_pairs(self):
        pairs = MODULE.pair_set([[0, 4], [1, 5]], 6, "AUCGUA")
        self.assertEqual(pairs, {(0, 4), (1, 5)})

    def test_noncanonical_pair_rejected(self):
        with self.assertRaisesRegex(MODULE.ContractError, "Non-canonical"):
            MODULE.pair_set([[0, 4]], 5, "AAAAA")

    def test_short_loop_rejected(self):
        with self.assertRaisesRegex(MODULE.ContractError, "Minimum-loop"):
            MODULE.pair_set([[0, 3]], 5, "AUCGA")

    def test_partner_conflict_rejected(self):
        with self.assertRaisesRegex(MODULE.ContractError, "Partner conflict"):
            MODULE.pair_set([[0, 4], [0, 5]], 6, "AUCGUU")

    def test_mutual_ct_pairs_are_preserved(self):
        pairs = MODULE.pairs_from_partner_indices([5, 0, 0, 0, 1], "AUGCU")
        self.assertEqual(pairs, [(0, 4)])

    def test_nonmutual_ct_pair_rejected(self):
        with self.assertRaisesRegex(MODULE.ContractError, "not mutual"):
            MODULE.pairs_from_partner_indices([5, 0, 0, 0, 0], "AUGCU")


class OfficialIdentityTest(unittest.TestCase):
    def test_real_bprna_new_contract(self):
        rows = MODULE.load_bprna_new()
        self.assertEqual(len(rows), 5388)
        self.assertEqual(min(len(row["seq"]) for row in rows), 33)
        self.assertEqual(max(len(row["seq"]) for row in rows), 489)
        self.assertEqual(sum(len(row["pairs"]) for row in rows), 135184)
        report = MODULE.contract_report(rows)
        self.assertIsNone(report["predictions"])
        self.assertFalse(report["model"]["ensemble"])
        self.assertTrue(report["training_exposure"]["bpnew_in_official_default_train_files"])
        self.assertFalse(MODULE.training_exposure().get("model_selection_permitted", False))

    def test_official_source_and_offline_checkpoint(self):
        source = MODULE.validate_official_source()
        self.assertEqual(source.name, f"UFold-{MODULE.OFFICIAL_COMMIT}")
        self.assertEqual(MODULE.EXPECTED_CHECKPOINT, "ufold_train_alldata.pt")
        self.assertIsNone(MODULE.EXPECTED_CHECKPOINT_SHA256)

    def test_checkpoint_hash_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / MODULE.EXPECTED_CHECKPOINT
            checkpoint.write_bytes(b"not-a-checkpoint")
            with self.assertRaisesRegex(MODULE.ContractError, "not certified"):
                MODULE.validate_checkpoint_identity(checkpoint, None)

    def test_checkpoint_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / MODULE.EXPECTED_CHECKPOINT
            checkpoint.write_bytes(b"not-a-checkpoint")
            with self.assertRaisesRegex(MODULE.ContractError, "SHA256 mismatch"):
                MODULE.validate_checkpoint_identity(checkpoint, "0" * 64)

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / MODULE.EXPECTED_CHECKPOINT
            with self.assertRaisesRegex(MODULE.ContractError, "unavailable"):
                MODULE.validate_checkpoint_identity(checkpoint, "0" * 64)

    def test_official_feature_and_ct_round_trip_cpu(self):
        import numpy as np

        network, data_generator, _, utils = MODULE.load_official_modules()
        model = network.U_Net(img_ch=17)
        self.assertEqual(sum(p.numel() for p in model.parameters()), MODULE.MODEL_PARAMETER_COUNT)
        features, seq_ori = MODULE.build_official_input("AUGCUA", data_generator, np)
        self.assertEqual(features.shape, (17, 80, 80))
        self.assertEqual(seq_ori.shape, (80, 4))
        sequence = "AUGCU"
        encoded = utils.seq_encoding(sequence)
        contact = np.zeros((5, 5), dtype=np.float32)
        contact[0, 4] = contact[4, 0] = 1.0
        ct = utils.contact2ct(contact, encoded, len(sequence))
        pairs = MODULE.pairs_from_partner_indices(ct["pair_index"].tolist(), sequence)
        self.assertEqual(pairs, [(0, 4)])

    def test_recovery_manifest_never_fakes_predictions(self):
        path = Path(__file__).parents[1] / "manifests/foldnar_v1/ufold_official_recovery_20261127.json"
        document = json.loads(path.read_text())
        self.assertEqual(document["status"], "BLOCKED_AUTHOR_WEIGHTS_UNAVAILABLE")
        self.assertIsNone(document["predictions"])
        self.assertIsNone(document["weights"]["sha256"])
        self.assertFalse(document["full_ready"])
        self.assertEqual(document["source"]["commit"], MODULE.OFFICIAL_COMMIT)
        self.assertEqual(document["source"]["archive_sha256"], MODULE.SOURCE_ARCHIVE_SHA256)
        self.assertEqual(document["dataset"]["sha256"], MODULE.DATA_SHA256)
        self.assertIn("bpnew", document["training_exposure"]["official_default_train_files"])


if __name__ == "__main__":
    unittest.main()
