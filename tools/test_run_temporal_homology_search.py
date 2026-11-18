import hashlib
import json
import sys

import run_temporal_homology_search as wrapper
from test_aggregate_temporal_homology import sequence

EXPECTED = "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"


def test_only_backend_arguments_are_forwarded(tmp_path, monkeypatch):
    fasta = tmp_path / "eligible.fasta"
    fasta.write_text("".join(f">q{index}|exact_detected=0\n{sequence(index)}\n" for index in range(1131)))
    output = tmp_path / "homology"
    output.mkdir()
    (output / "query_manifest.json").write_text(json.dumps({"n_queries": 1131, "metadata_sha256": EXPECTED,
                                                               "query_fasta_sha256": EXPECTED}))
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    monkeypatch.setattr(wrapper.hashlib, "sha256", lambda value: type("Digest", (), {"hexdigest": lambda self: EXPECTED})())
    called = []
    monkeypatch.setattr(wrapper.run_pretrain_homology_search, "main", lambda: called.append(sys.argv.copy()))
    monkeypatch.setattr(sys, "argv", ["wrapper", "--metadata", str(metadata), "--eligible-fasta", str(fasta),
                                      "--output-root", str(output), "--source", "rfam_14_9",
                                      "--threads", "7", "--split-memory-limit", "31G"])
    wrapper.main()
    assert called == [["wrapper", "--source", "rfam_14_9", "--threads", "7",
                       "--split-memory-limit", "31G"]]
    assert wrapper.run_pretrain_homology_search.OUTPUT == output
