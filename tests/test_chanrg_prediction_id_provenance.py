import copy
import json
import tempfile
import unittest
from pathlib import Path

from reexport_chanrg_prediction_ids import (
    ARM_SPECS,
    SPLITS,
    export_all,
    file_record,
    reexport_prediction,
    sha256_file,
)
from verify_chanrg_prediction_id_provenance import verify_manifest


class ChanrgPredictionIdProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / "source"
        self.raw_root = self.root / "raw"
        self.output_root = self.root / "output"
        self.source_root.mkdir()
        for split in SPLITS:
            source = [
                {"id": f"{split}-a", "seq": "ACGU", "pairs": [[0, 3]]},
                {"id": f"{split}-b", "seq": "GGGG", "pairs": []},
            ]
            (self.source_root / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in source), encoding="utf-8")
            for arm, (_model, template) in ARM_SPECS.items():
                raw_path = self.raw_root / template.format(split=split)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for source_row in source:
                    n_gt = len(source_row["pairs"])
                    row = {
                        "name": "", "f1": 1.0, "pred_pairs": source_row["pairs"],
                        "n_pred": n_gt, "n_gt": n_gt,
                    }
                    if arm != "xens2":
                        row.update({
                            "length": len(source_row["seq"]),
                            "n_pred_pairs": n_gt, "n_gt_pairs": n_gt,
                        })
                    rows.append(row)
                raw = {
                    "n_sequences": len(rows), "n_skipped": 0,
                    "pair_level": {
                        "micro": {"tp": 1, "fp": 0, "fn": 0,
                                  "precision": 1.0, "recall": 1.0, "f1": 1.0},
                        "macro": {"f1": 1.0},
                    },
                    "per_sequence": rows,
                }
                raw_path.write_text(json.dumps(raw), encoding="utf-8")
        registry_rows = []
        for split in SPLITS:
            for arm, (model, template) in ARM_SPECS.items():
                raw_path = self.raw_root / template.format(split=split)
                registry_rows.append({
                    "prediction_id": f"{model}/{split}", "path": str(raw_path),
                    "status": "FROZEN", "sha256": sha256_file(raw_path),
                })
        registry_rows.append({"prediction_id": "unrelated/missing", "path": None,
                              "status": "BLOCKED", "sha256": None})
        registry_rows.append({"prediction_id": "unrelated/no-path", "status": "BLOCKED"})
        self.registry = self.root / "registry.json"
        self.registry.write_text(json.dumps({"rows": registry_rows}), encoding="utf-8")
        self.manifest_path = self.output_root / "manifest.json"
        self.manifest = export_all(self.source_root, self.raw_root, self.output_root,
                                   self.registry)

    def tearDown(self):
        self.temp.cleanup()

    def raw_path(self, split="chanrg_test", arm="r2d"):
        return self.raw_root / ARM_SPECS[arm][1].format(split=split)

    def output_path(self, split="chanrg_test", arm="r2d"):
        return self.output_root / "predictions" / arm / f"{split}.json"

    def test_all_names_follow_frozen_ids_and_independent_verification_passes(self):
        self.assertEqual(len(self.manifest["rows"]), 12)
        for split in SPLITS:
            source = [json.loads(line) for line in
                      (self.source_root / f"{split}.jsonl").read_text().splitlines()]
            for arm in ARM_SPECS:
                raw = json.loads(self.raw_path(split, arm).read_text())
                repaired = json.loads(self.output_path(split, arm).read_text())
                self.assertEqual([row["name"] for row in repaired["per_sequence"]],
                                 [row["id"] for row in source])
                for raw_row, repaired_row in zip(raw["per_sequence"], repaired["per_sequence"]):
                    self.assertEqual(raw_row["name"], "")
                    repaired_row["name"] = ""
                self.assertEqual(raw, repaired)
        report = verify_manifest(self.manifest_path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["n_prediction_rows_total"], 24)

    def test_reexport_is_byte_reproducible(self):
        second = self.root / "second"
        export_all(self.source_root, self.raw_root, second, self.registry)
        for split in SPLITS:
            for arm in ARM_SPECS:
                relative = Path("predictions") / arm / f"{split}.json"
                self.assertEqual((self.output_root / relative).read_bytes(),
                                 (second / relative).read_bytes())

    def test_swapped_raw_rows_are_rejected(self):
        path = self.raw_path()
        raw = json.loads(path.read_text())
        raw["per_sequence"].reverse()
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "row 0"):
            reexport_prediction(self.source_root / "chanrg_test.jsonl", path,
                                self.root / "blocked.json", arm="r2d", split="chanrg_test")

    def test_duplicate_source_ids_are_rejected(self):
        path = self.source_root / "chanrg_test.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["id"] = rows[0]["id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "duplicate frozen source ids"):
            reexport_prediction(path, self.raw_path(), self.root / "blocked.json",
                                arm="r2d", split="chanrg_test")

    def test_nonempty_raw_name_is_rejected(self):
        path = self.raw_path()
        raw = json.loads(path.read_text())
        raw["per_sequence"][0]["name"] = "unprovenanced"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "unexpected raw name"):
            reexport_prediction(self.source_root / "chanrg_test.jsonl", path,
                                self.root / "blocked.json", arm="r2d", split="chanrg_test")

    def test_frozen_registry_hash_mismatch_is_rejected(self):
        path = self.raw_path()
        raw = json.loads(path.read_text())
        raw["unrelated_metadata"] = "changed after freeze"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "frozen registry"):
            reexport_prediction(self.source_root / "chanrg_test.jsonl", path,
                                self.root / "blocked.json", arm="r2d", split="chanrg_test",
                                frozen_registry=self.registry)

    def test_existing_output_is_not_overwritten(self):
        output = self.output_path()
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            reexport_prediction(self.source_root / "chanrg_test.jsonl", self.raw_path(), output,
                                arm="r2d", split="chanrg_test", frozen_registry=self.registry)
        self.assertEqual(output.read_bytes(), before)

    def test_independent_verifier_rejects_reordered_ids(self):
        output = self.output_path()
        repaired = json.loads(output.read_text())
        names = [row["name"] for row in repaired["per_sequence"]]
        repaired["per_sequence"][0]["name"], repaired["per_sequence"][1]["name"] = names[1], names[0]
        output.write_text(json.dumps(repaired), encoding="utf-8")
        manifest = copy.deepcopy(self.manifest)
        row = next(row for row in manifest["rows"]
                   if row["arm"] == "r2d" and row["split"] == "chanrg_test")
        row["repaired_prediction"] = file_record(output)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "ID provenance mismatch"):
            verify_manifest(self.manifest_path)

    def test_independent_verifier_rejects_non_name_mutation(self):
        output = self.output_path()
        repaired = json.loads(output.read_text())
        repaired["per_sequence"][0]["f1"] = 0.0
        output.write_text(json.dumps(repaired), encoding="utf-8")
        manifest = copy.deepcopy(self.manifest)
        row = next(row for row in manifest["rows"]
                   if row["arm"] == "r2d" and row["split"] == "chanrg_test")
        row["repaired_prediction"] = file_record(output)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "non-name content changed"):
            verify_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
