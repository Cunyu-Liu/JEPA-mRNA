#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_e2efold_official_bprna_new.py"
SPEC = importlib.util.spec_from_file_location("e2efold_official_recovery", MODULE_PATH)
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

    def test_out_of_range_pair_rejected(self):
        with self.assertRaisesRegex(MODULE.ContractError, "Out-of-range"):
            MODULE.pair_set([[0, 5]], 5, "AUCGA")

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

    def test_official_config_and_expected_checkpoint(self):
        config = MODULE.load_model_spec()
        self.assertEqual(config["model_type"], "att_simple_fix")
        self.assertEqual(config["pp_model"], "mixed")
        self.assertEqual(MODULE.EXPECTED_CHECKPOINT,
                         "e2e_att_simple_fix_mixed_s20_d10_rnastralign_all_600_f1_position_matrix.pt")

    def test_checkpoint_name_and_hash_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "other.pt"
            wrong.write_bytes(b"not-a-checkpoint")
            with self.assertRaisesRegex(MODULE.ContractError, "must be named"):
                MODULE.validate_checkpoint_identity(wrong, MODULE.sha256(wrong))
            with self.assertRaisesRegex(MODULE.ContractError, "unavailable"):
                MODULE.validate_checkpoint_identity(
                    Path(directory) / MODULE.EXPECTED_CHECKPOINT, "0" * 64)

    def test_checkpoint_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / MODULE.EXPECTED_CHECKPOINT
            checkpoint.write_bytes(b"not-a-checkpoint")
            with self.assertRaisesRegex(MODULE.ContractError, "SHA256 mismatch"):
                MODULE.validate_checkpoint_identity(checkpoint, "0" * 64)

    def test_official_contact2ct_round_trip_cpu(self):
        import numpy as np

        _, utils = MODULE.load_official_modules()
        sequence = "AUGCU"
        encoded = utils.seq_encoding(sequence)
        contact = np.zeros((5, 5), dtype=np.float32)
        contact[0, 4] = contact[4, 0] = 1.0
        ct = utils.contact2ct(contact, encoded, len(sequence))
        self.assertEqual(MODULE.ct_to_pairs(ct, sequence), [(0, 4)])

    def test_recovery_manifest_never_fakes_predictions(self):
        path = Path(__file__).parents[1] / "manifests/foldnar_v1/e2efold_official_recovery_20261126.json"
        document = json.loads(path.read_text())
        self.assertEqual(document["status"], "BLOCKED_OFFICIAL_WEIGHTS_UNAVAILABLE")
        self.assertIsNone(document["predictions"])
        self.assertIsNone(document["weights"]["sha256"])
        self.assertEqual(document["source"]["commit"], MODULE.OFFICIAL_COMMIT)
        self.assertEqual(document["source"]["archive_sha256"], MODULE.SOURCE_ARCHIVE_SHA256)


if __name__ == "__main__":
    unittest.main()
