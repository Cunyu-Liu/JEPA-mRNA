import hashlib
import json
import tempfile
from pathlib import Path

from prepare_rinalmo_full_d1_round53 import prepare, sha256_file


def make_fixture():
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    decon = root / "decon"
    decon.mkdir()
    train = {"seq": "AUGCA", "pairs": [[2, 5]], "canonical_nested_pairs": [[2, 5]], "name": "train_long"}
    holdout = {"seq": "GGCCAA", "pairs": [[1, 6]], "canonical_nested_pairs": [[1, 6]], "name": "holdout"}
    (decon / "complex_train.jsonl").write_text(json.dumps(train) + "\n")
    (decon / "complex_holdout.jsonl").write_text(json.dumps(holdout) + "\n")
    (decon / "decontamination.json").write_text("{}")
    verification = {"status": "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT",
                    "decontamination_sha256": sha256_file(decon / "decontamination.json"),
                    "n_train": 1, "n_holdout": 1,
                    "train_sha256": sha256_file(decon / "complex_train.jsonl"),
                    "holdout_sha256": sha256_file(decon / "complex_holdout.jsonl")}
    (decon / "verification.json").write_text(json.dumps(verification))
    tr1c = root / "tr1c.jsonl"
    tr1c.write_text(json.dumps({"seq": "AAAA", "pairs": [], "name": "old"}) + "\n")
    return directory, decon, tr1c


for run in range(3):
    directory, decon, tr1c = make_fixture()
    root = Path(directory.name)
    manifest = prepare(decon, tr1c, root / "out", root / "out/manifest.json")
    assert manifest["n_total_train"] == 2 and manifest["n_complex_holdout"] == 1
    rows = [json.loads(line) for line in (root / "out/train_d1.jsonl").read_text().splitlines()]
    assert rows[1]["pairs"] == [[1, 4]] and rows[1]["canonical_nested_pairs"] == [[1, 4]]
    assert manifest["train_over_1024"] == 0
    directory.cleanup()

directory, decon, tr1c = make_fixture()
row = json.loads((decon / "complex_train.jsonl").read_text())
row["pairs"] = [[1, 4], [2, 5]]
row["canonical_nested_pairs"] = row["pairs"]
(decon / "complex_train.jsonl").write_text(json.dumps(row) + "\n")
try:
    prepare(decon, tr1c, Path(directory.name) / "out2", Path(directory.name) / "out2/manifest.json")
except AssertionError:
    pass
else:
    raise AssertionError("crossing pair accepted")
directory.cleanup()

directory, decon, tr1c = make_fixture()
(decon / "verification.json").write_text("{}")
try:
    prepare(decon, tr1c, Path(directory.name) / "out3", Path(directory.name) / "out3/manifest.json")
except (AssertionError, KeyError):
    pass
else:
    raise AssertionError("invalid verification accepted")
directory.cleanup()
print("PASS full-D1 conversion, conservation, crossing and verification failure injection")
