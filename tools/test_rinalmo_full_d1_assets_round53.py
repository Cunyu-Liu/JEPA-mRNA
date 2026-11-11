import json
import tempfile
from pathlib import Path

import numpy as np

from build_rinalmo_full_d1_teacher_round53 import sha256_file, validate_teacher


for run in range(3):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sequence = "AUGC"
        shard = root / "shard_00000.npz"
        np.savez(shard, sequences=np.asarray([sequence], dtype=object),
                 probs=np.asarray([np.eye(4)], dtype=object))
        manifest = {"teacher": "thermo:viennarna", "version_lock": "ViennaRNA 2.7.2",
                    "n_sequences": 1, "shards": [{"file": shard.name, "sha256": sha256_file(shard)}],
                    "meta": {"corpus_sha256": "abc"}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        result = validate_teacher(root, {sequence}, "abc")
        assert result["n_sequences"] == 1
        manifest["meta"]["corpus_sha256"] = "wrong"
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            validate_teacher(root, {sequence}, "abc")
        except AssertionError:
            pass
        else:
            raise AssertionError("teacher corpus identity mismatch accepted")
print("PASS teacher sequence/probability/shape and corpus identity failure injection")
