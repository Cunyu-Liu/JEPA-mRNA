import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audit_rna_complex_round49 import SPLITS, sha256, source_metadata, validate_collection


class AuditIntegrationTests(unittest.TestCase):
    def test_exact_homology_and_group_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            random.seed(71)
            clean = ["".join(random.choice("ACGU") for _ in range(120)) for _ in range(8)]
            exact = "ACGU" * 30
            homologous = "".join("A" if i % 12 == 0 else base for i, base in enumerate(clean[0]))
            sequences = [exact, homologous] + clean + ["G" * 120]
            labels = [{"id": f"P{i}_A", "pdb_id": f"P{i}", "sequence": sequence,
                       "raw_pairs": [[1, 120]],
                       "canonical_nested_pairs": [] if i == len(sequences) - 1 else [[1, 120]]}
                      for i, sequence in enumerate(sequences)]
            duplicate = dict(labels[4], id="P4_B")
            labels.append(duplicate)
            results = {}
            for i, chain in enumerate(labels):
                directory = root / "entries" / f"P{i}"
                directory.mkdir(parents=True)
                label_path = directory / "labels.json"
                label_path.write_text(json.dumps({"pdb_id": f"P{i}", "chains": [chain],
                                                  "rejected_chains": [], "dssr_seq_id_mode": "observed_nts_v1"}))
                results[f"P{i}"] = {"accepted_chains": 1, "rejected_chains": 0,
                                    "labels_sha256": sha256(label_path)}
            collection_path = root / "entries" / "collection.json"
            collection_path.write_text(json.dumps({"complete": True, "dssr_seq_id_mode": "observed_nts_v1",
                                                   "n_entries": len(labels), "results": results}))
            ids_path = root / "ids.json"
            ids_path.write_text(json.dumps({"result_set": [f"P{i}" for i in range(len(labels))]}))
            _, label_paths = validate_collection(root / "entries", ids_path)
            self.assertEqual(len(label_paths), len(labels))
            collection_path.write_text(json.dumps({"complete": False, "n_entries": len(labels), "results": results}))
            with self.assertRaises(AssertionError):
                validate_collection(root / "entries", ids_path)
            collection_path.write_text(json.dumps({"complete": True, "dssr_seq_id_mode": "observed_nts_v1",
                                                   "n_entries": len(labels), "results": results}))
            pretrain = root / "pretrain.fa"
            pretrain.write_text(">exact\n" + exact + "\n")
            protected = root / "protected"
            protected.mkdir()
            for name in SPLITS + ["train", "vl0"]:
                path = (protected / f"{name}.jsonl") if name in SPLITS else root / f"{name}.jsonl"
                path.write_text(json.dumps({"seq": clean[0] if name == "train" else "C" * 120}) + "\n")
            for suffix in ["", ".recheck"]:
                output = root / f"out{suffix}"
                command = [sys.executable, "/home/rna_jev/tools/audit_rna_complex_round49.py",
                           "--entries-root", str(root / "entries"), "--expected-ids-json", str(ids_path),
                           "--pretrain-fasta", str(pretrain),
                           "--pretrain-sha256", sha256(pretrain), "--clean-splits", str(protected),
                           "--train-jsonl", str(root / "train.jsonl"), "--vl0-jsonl", str(root / "vl0.jsonl"),
                           "--out-dir", str(output)]
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    self.fail(completed.stdout + "\n" + completed.stderr)
                verification = subprocess.run([sys.executable, "/home/rna_jev/tools/verify_rna_complex_decontamination_round49.py",
                                               "--out-dir", str(output)], capture_output=True, text=True)
                if verification.returncode:
                    self.fail(verification.stdout + "\n" + verification.stderr)
                manifest = json.loads((output / "decontamination.json").read_text())
                self.assertEqual(manifest["n_unique_sequences"], 11)
                self.assertEqual(manifest["n_removed"], 3)
                self.assertEqual(manifest["n_unsupported_nested"], 1)
                self.assertEqual(manifest["n_clean"], 7)
                self.assertGreater(manifest["n_holdout"], 0)
                train = [json.loads(line) for line in (output / "complex_train.jsonl").read_text().splitlines()]
                holdout = [json.loads(line) for line in (output / "complex_holdout.jsonl").read_text().splitlines()]
                self.assertEqual(len(train) + len(holdout), 7)
                self.assertTrue(train and holdout)
                self.assertFalse({row["group"] for row in train} & {row["group"] for row in holdout})
                self.assertFalse({row["source"]["pdb_id"] for row in train} & {row["source"]["pdb_id"] for row in holdout})
            first = json.loads((root / "out/decontamination.json").read_text())
            second = json.loads((root / "out.recheck/decontamination.json").read_text())
            first_command, second_command = first.pop("search_command"), second.pop("search_command")
            second_command = [part.replace("out.recheck", "out") for part in second_command]
            self.assertEqual(first_command[:5] + first_command[6:], second_command[:5] + second_command[6:])
            first_cluster, second_cluster = first.pop("cluster_search_command"), second.pop("cluster_search_command")
            second_cluster = [part.replace("out.recheck", "out") for part in second_cluster]
            self.assertEqual(first_cluster[:5] + first_cluster[6:], second_cluster[:5] + second_cluster[6:])
            self.assertEqual(first, second)
            self.assertEqual((root / "out/complex_train.jsonl").read_bytes(),
                             (root / "out.recheck/complex_train.jsonl").read_bytes())
            self.assertEqual((root / "out/complex_holdout.jsonl").read_bytes(),
                             (root / "out.recheck/complex_holdout.jsonl").read_bytes())

    def test_real_mmcif_source_metadata(self):
        path = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rna_complex_round49/asset_gate/1EHZ.cif")
        if not path.is_file():
            self.skipTest("real asset gate is unavailable")
        metadata = source_metadata(path, {"1"})
        self.assertIn("Saccharomyces cerevisiae", metadata["source_organisms"])
        self.assertEqual(metadata["partner_entity_types"], [])


if __name__ == "__main__":
    unittest.main()
