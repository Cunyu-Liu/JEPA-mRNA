"""Tests for ``data/ss/prepare_decision_data.py`` (benchmark corpus preparation).

These run without GPU, without network and without the real corpora: every case
is a hand-built string or a temporary CSV.  What they pin down is the set of
properties the project's credibility depends on -- that a record we accept is
actually legal, that a record we reject is rejected for a named reason, and that
the two encodings of the same structure (bpseq partner column vs dot-bracket)
cannot silently disagree.

Run::

    python -m pytest tests/test_ss_data_prep.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "data" / "ss" / "prepare_decision_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_decision_data", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prep = _load_module()


# ---------------------------------------------------------------------------
# bpseq parsing
# ---------------------------------------------------------------------------
def test_bpseq_round_trip_nested_hairpin():
    text = "\n".join(f"{i + 1} {b} {p}" for i, (b, p) in enumerate(
        [("U", 6), ("C", 5), ("G", 0), ("A", 0), ("G", 2), ("A", 1)]))
    seq, pairs = prep.read_bpseq_text(text)
    assert seq == "UCGAGA"
    assert pairs == [(0, 5), (1, 4)]
    assert prep.pairs_to_dotbracket(6, pairs) == "((..))"


def test_bpseq_accepts_t_and_normalises_to_u():
    seq, pairs = prep.read_bpseq_text("1 T 2\n2 A 1\n")
    assert seq == "UA"
    assert pairs == [(0, 1)]


def test_bpseq_skips_comments_and_blank_lines():
    seq, pairs = prep.read_bpseq_text("# header\n\n1 A 0\n2 C 0\n\n")
    assert seq == "AC"
    assert pairs == []


def test_bpseq_rejects_noncontiguous_index():
    with pytest.raises(prep.RejectError) as info:
        prep.read_bpseq_text("1 A 0\n3 C 0\n")
    assert info.value.reason == "bpseq_noncontiguous"


def test_bpseq_rejects_asymmetric_partner():
    # 1 -> 2, but 2 -> 0.  This is the corruption a truncated file produces.
    with pytest.raises(prep.RejectError) as info:
        prep.read_bpseq_text("1 A 2\n2 C 0\n")
    assert info.value.reason == "partner_asymmetric"


def test_bpseq_rejects_partner_out_of_range():
    with pytest.raises(prep.RejectError) as info:
        prep.read_bpseq_text("1 A 9\n")
    assert info.value.reason == "partner_out_of_range"


def test_bpseq_rejects_malformed_row():
    with pytest.raises(prep.RejectError) as info:
        prep.read_bpseq_text("1 A\n")
    assert info.value.reason == "bpseq_malformed"


# ---------------------------------------------------------------------------
# dot-bracket
# ---------------------------------------------------------------------------
def test_dotbracket_round_trip():
    pairs = [(0, 7), (1, 6), (2, 5)]
    db = prep.pairs_to_dotbracket(9, pairs)
    # 2 pairs with 5, so 3 and 4 are the loop; index 8 is unpaired too.
    assert db == "(((..)))."
    assert len(db) == 9
    assert prep.dotbracket_to_pairs(db) == pairs


def test_dotbracket_rejects_unbalanced_close():
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("(..))")
    assert info.value.reason == "unbalanced_bracket"


def test_dotbracket_rejects_unclosed_open():
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("((..)")
    assert info.value.reason == "unbalanced_bracket"


def test_dotbracket_rejects_unknown_char():
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("([)]")
    assert info.value.reason == "unknown_structure_char"


# ---------------------------------------------------------------------------
# crossing detection (pseudoknots must be routable, not silently averaged in)
# ---------------------------------------------------------------------------
def test_no_crossing_for_nested_pairs():
    assert prep.find_crossing_pairs([(0, 5), (1, 4), (2, 3)]) == []


def test_crossing_detected_for_pseudoknot():
    assert prep.find_crossing_pairs([(0, 5), (2, 7)]) == [((0, 5), (2, 7))]


def test_adjacent_pairs_do_not_cross():
    assert prep.find_crossing_pairs([(0, 3), (4, 7)]) == []


def test_check_record_flags_pseudoknot():
    facts = prep.check_record("ACGUACGUAC", [(0, 7), (2, 9)])
    assert facts["is_pseudoknot"] is True
    assert facts["n_crossing"] == 1


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_check_record_accepts_legal_structure():
    facts = prep.check_record("ACGUACGUACGU", [(0, 11), (1, 10)])
    assert facts == {"n_pairs": 2, "n_crossing": 0, "is_pseudoknot": False}


def test_check_record_rejects_ambiguous_base():
    with pytest.raises(prep.RejectError) as info:
        prep.check_record("ACGUN", [])
    assert info.value.reason == "bad_alphabet"


def test_check_record_rejects_hairpin_below_minimum():
    with pytest.raises(prep.RejectError) as info:
        prep.check_record("ACGUACGU", [(0, 3)])
    assert info.value.reason == "hairpin_too_short"


def test_check_record_accepts_minimum_loop_of_three():
    # span == 4 is the smallest legal hairpin (three unpaired bases).
    assert prep.check_record("ACGUACGU", [(0, 4)])["n_pairs"] == 1


def test_check_record_rejects_out_of_range_pair():
    with pytest.raises(prep.RejectError) as info:
        prep.check_record("ACGU", [(0, 9)])
    assert info.value.reason == "pair_out_of_range"


def test_check_record_rejects_duplicate_pair():
    with pytest.raises(prep.RejectError) as info:
        prep.check_record("ACGUACGU", [(0, 7), (0, 7)])
    assert info.value.reason == "duplicate_pair"


def test_check_record_rejects_empty():
    with pytest.raises(prep.RejectError) as info:
        prep.check_record("", [])
    assert info.value.reason == "empty"


# ---------------------------------------------------------------------------
# RiNALMo CSV: the two encodings must agree
# ---------------------------------------------------------------------------
CSV_HEADER = "id,sequence,structure,base_pairs,len\n"


def _write_csv(tmp_path, body: str) -> str:
    path = tmp_path / "bench.csv"
    path.write_text(CSV_HEADER + body, encoding="utf-8")
    return str(path)


def test_csv_row_parsed(tmp_path):
    path = _write_csv(tmp_path, 'r1,ACGUACGU,"((....))","[[0, 7], [1, 6]]",8\n')
    rows = list(prep.read_rinalmo_csv(path))
    assert len(rows) == 1
    assert rows[0]["name"] == "r1"
    assert rows[0]["seq"] == "ACGUACGU"
    assert rows[0]["pairs"] == [(0, 7), (1, 6)]


def test_csv_empty_pairs_ok(tmp_path):
    path = _write_csv(tmp_path, 'r1,ACGU,"....","[]",4\n')
    rows = list(prep.read_rinalmo_csv(path))
    assert rows[0]["pairs"] == []


def test_csv_rejects_length_mismatch(tmp_path):
    path = _write_csv(tmp_path, 'r1,ACGU,"....","[]",9\n')
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(path))
    assert info.value.reason == "csv_len_mismatch"


def test_csv_rejects_structure_length_mismatch(tmp_path):
    path = _write_csv(tmp_path, 'r1,ACGUACGU,"....","[]",8\n')
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(path))
    assert info.value.reason == "csv_structure_len_mismatch"


def test_csv_rejects_disagreeing_columns(tmp_path):
    # dot-bracket says one pair, base_pairs says none -- must not pass silently.
    path = _write_csv(tmp_path, 'r1,ACGUACGU,"((....))","[]",8\n')
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(path))
    assert info.value.reason == "csv_pair_column_disagrees"


def test_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,sequence\nr1,ACGU\n", encoding="utf-8")
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(str(path)))
    assert info.value.reason == "csv_missing_columns"


# ---------------------------------------------------------------------------
# end-to-end: the CLI writes what it promises, and rejects are accounted for
# ---------------------------------------------------------------------------
def test_cli_bpseq_dir_end_to_end(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "good.bpseq").write_text("1 A 0\n2 C 0\n", encoding="utf-8")
    # span 4 (three unpaired bases) is the smallest legal hairpin.
    (corpus / "also_good.bpseq").write_text("1 G 5\n2 C 0\n3 A 0\n4 A 0\n5 G 1\n",
                                            encoding="utf-8")
    (corpus / "bad_base.bpseq").write_text("1 A 0\n2 N 0\n", encoding="utf-8")

    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "out.manifest.json"
    rc = prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
                    "--manifest", str(manifest)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {r["name"] for r in records} == {"good.bpseq", "also_good.bpseq"}
    for record in records:
        assert set(record) >= {"name", "seq", "structure", "pairs", "n_pairs",
                               "is_pseudoknot", "source"}
        # the emitted dot-bracket must reproduce the emitted pairs
        assert prep.dotbracket_to_pairs(record["structure"]) == \
            [tuple(p) for p in record["pairs"]]

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["n_accepted"] == 2
    assert meta["n_rejected"] == 1
    assert meta["reject_reasons"] == {"bad_alphabet": 1}
    # conservation: nothing vanishes without a reason code
    assert meta["n_accepted"] + meta["n_rejected"] == 3
    assert meta["out_sha256"] and len(meta["out_sha256"]) == 64
    assert meta["provenance"]["n_files"] == 3


def test_cli_rejects_sidecar_written(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "bad.bpseq").write_text("1 A 0\n2 N 0\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out)])
    rejects = Path(str(out) + ".rejects.jsonl")
    rows = [json.loads(line) for line in rejects.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["reason"] == "bad_alphabet"


def test_cli_limit_stops_early(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    for i in range(5):
        (corpus / f"s{i}.bpseq").write_text("1 A 0\n2 C 0\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out), "--limit", "2"])
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_empty_dir_is_an_error(tmp_path, capsys):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    rc = prep.main(["--bpseq-dir", str(corpus), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2
    assert "no .bpseq files" in capsys.readouterr().err


def test_dir_fingerprint_detects_membership_change(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "a.bpseq").write_text("1 A 0\n", encoding="utf-8")
    before = prep.dir_fingerprint(str(corpus))
    (corpus / "b.bpseq").write_text("1 A 0\n", encoding="utf-8")
    after = prep.dir_fingerprint(str(corpus))
    assert before["n_files"] == 1 and after["n_files"] == 2
    assert before["names_sha256"] != after["names_sha256"]


# ---------------------------------------------------------------------------
# length bucketing must match the protocol frozen in the benchmark decision
# ---------------------------------------------------------------------------
def test_length_histogram_matches_protocol_buckets():
    hist = prep._length_histogram([50, 100, 150, 300, 500, 800, 1500])
    assert hist == {"<=100": 2, "<=200": 1, "<=400": 1, "<=600": 1,
                    "<=1000": 1, ">1000": 1}
