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
    # base_pairs is 1-based in the real format; (1,8) and (2,7) are 0-based (0,7) and (1,6)
    path = _write_csv(tmp_path, 'r1,ACGUACGU,"((....))","[[1, 8], [2, 7]]",8\n')
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
# reference .plk source
# ---------------------------------------------------------------------------
def _write_plk(path, frames, *, name_col="Id", with_pk=True, with_structure=True):
    """Write a ``.plk`` in the RNAformer release layout.

    ``frames`` maps a split name to a list of rows, each row being
    ``(seq, pairs, pk_labels, structure_or_None)``.  Built with the pandas that
    is actually installed, so the read side is exercised without needing the
    pandas-1.x compatibility shim (which is separately unit-tested).
    """
    pd = pytest.importorskip("pandas")
    out = {}
    for split, rows in frames.items():
        records = []
        for index, (seq, pairs, labels, structure) in enumerate(rows):
            record = {
                "sequence": list(seq),
                "pos1id": [i for i, _ in pairs],
                "pos2id": [j for _, j in pairs],
                "set": split,
                "length": len(seq),
                "is_pdb": False,
                "has_pk": any(labels),
                "has_multiplet": False,
                "has_nc": False,
            }
            if with_pk:
                record["pk"] = list(labels)
            if with_structure:
                record["structure"] = list(structure) if structure is not None \
                    else list("." * len(seq))
            if name_col:
                record[name_col] = f"{split}_{index}"
            records.append(record)
        out[split] = pd.DataFrame(records)
    if len(out) == 1:
        pd.to_pickle(next(iter(out.values())), path)
    else:
        pd.to_pickle(out, path)
    return path


def test_row_pairs_normalises_reversed_and_sorts():
    pairs, labels = prep._row_pairs(
        {"pos1id": [7, 1], "pos2id": [2, 9], "pk": [1, 0]}, 10)
    assert pairs == [(1, 9), (2, 7)]
    # labels must follow their pairs through the sort, not stay in input order
    assert labels == [0, 1]


def test_row_pairs_rejects_out_of_range():
    with pytest.raises(prep.RejectError) as excinfo:
        prep._row_pairs({"pos1id": [0], "pos2id": [10]}, 10)
    assert excinfo.value.reason == "pair_out_of_range"


def test_row_pairs_rejects_pk_length_mismatch():
    with pytest.raises(prep.RejectError) as excinfo:
        prep._row_pairs({"pos1id": [0, 1], "pos2id": [9, 8], "pk": [0]}, 10)
    assert excinfo.value.reason == "plk_pk_column_mismatch"


def test_filter_removes_exactly_the_pk_labelled_pairs():
    """pk-drop must remove the crossing pairs and leave the nested ones alone."""
    seq = "A" * 12
    pairs = [(0, 11), (1, 10), (2, 6), (3, 7)]
    labels = [0, 0, 1, 1]           # the (2,6)/(3,7) pair crosses (0,11)
    stats = {}
    kept, dropped = prep.filter_raw_pairs(
        seq, pairs, labels, pseudoknot_from_pk=True,
        pair_type_policy="keep", multiplet_policy="keep", stats=stats)
    assert kept == [(0, 11), (1, 10)]
    assert dropped == [(2, 6), (3, 7)]
    assert stats["dropped_pseudoknot_pairs"] == 2


def test_filter_drops_non_canonical_pairs():
    seq = "A" * 6 + "A" + "A" * 5        # pairs (0,11) and (1,6) both A-A
    pairs = [(0, 11), (1, 6)]
    stats = {}
    kept, _ = prep.filter_raw_pairs(
        seq, pairs, [0, 0], pseudoknot_from_pk=False,
        pair_type_policy="canonical-drop", multiplet_policy="keep", stats=stats)
    assert kept == []
    assert stats["dropped_noncanonical_pairs"] == 2


def test_filter_keeps_canonical_and_wobble():
    seq = "AUG" + "C" * 5 + "CAU"          # (0,10) A-U, (1,9) U-A, (2,8) G-C
    pairs = [(0, 10), (1, 9), (2, 8)]
    stats = {}
    kept, _ = prep.filter_raw_pairs(
        seq, pairs, [0, 0, 0], pseudoknot_from_pk=False,
        pair_type_policy="canonical-drop", multiplet_policy="keep", stats=stats)
    assert kept == pairs and stats.get("dropped_noncanonical_pairs", 0) == 0


def test_filter_keeps_the_wobble_pair_specifically():
    """G-U must survive: it is the canonical third pair, not a non-canonical one."""
    seq = "G" + "A" * 9 + "U"              # length 11; (0,10) is G-U
    stats = {}
    kept, _ = prep.filter_raw_pairs(
        seq, [(0, 10)], [0], pseudoknot_from_pk=False,
        pair_type_policy="canonical-drop", multiplet_policy="keep", stats=stats)
    assert kept == [(0, 10)]


def test_multiplet_position_is_resolved_to_one_partner():
    """A base with two partners must be reduced to a single partner.

    The downstream crossing pass only handles the shared-*left*-endpoint form
    (``(2,4)`` + ``(2,9)``), and it resolves it by crossing degree rather than by
    span.  The shared-*right* form is not detected at all, and
    :func:`pairs_to_dotbracket` would then silently write ``")"`` twice.
    """
    seq = "A" * 12
    pairs = [(2, 4), (2, 9)]
    stats = {}
    kept, _ = prep.filter_raw_pairs(
        seq, pairs, [0, 0], pseudoknot_from_pk=False,
        pair_type_policy="keep", multiplet_policy="drop", stats=stats)
    assert kept == [(2, 9)]              # longest span wins
    assert stats["dropped_multiplet_pairs"] == 1
    assert stats["multiplet_positions"] == 1
    # and the surviving set is representable
    prep.pairs_to_dotbracket(12, kept)


def test_shared_right_endpoint_is_not_a_crossing_but_corrupts_the_render():
    """Pins the premise of the rule above with the case the pass really misses.

    This is the evidence that ``multiplet_policy`` is load-bearing rather than
    belt-and-braces: ``find_crossing_pairs`` returns nothing, and
    ``pairs_to_dotbracket`` returns an *unbalanced* string with one ``")"`` --
    one fewer pair than the ``pairs`` list still claims.  Nothing in the record
    path re-reads that string, so the inconsistency would be written to the
    corpus as-is; only a later round-trip discovers it, and by then it is a
    rejected record rather than a filtered pair.
    """
    shared_right = [(2, 9), (4, 9)]
    assert prep.find_crossing_pairs(shared_right) == []
    rendered = prep.pairs_to_dotbracket(12, shared_right)
    assert rendered.count("(") == 2 and rendered.count(")") == 1
    with pytest.raises(prep.RejectError) as excinfo:
        prep.dotbracket_to_pairs(rendered)
    assert excinfo.value.reason == "unbalanced_bracket"

    stats = {}
    kept, _ = prep.filter_raw_pairs(
        "A" * 12, shared_right, [0, 0], pseudoknot_from_pk=False,
        pair_type_policy="keep", multiplet_policy="drop", stats=stats)
    assert kept == [(2, 9)]
    # and the surviving render round-trips
    assert prep.dotbracket_to_pairs(prep.pairs_to_dotbracket(12, kept)) == kept


def test_shared_left_endpoint_is_a_crossing_but_resolved_by_degree():
    """Documents why the multiplet rule is not redundant even for this form."""
    shared_left = [(2, 4), (2, 9)]
    assert prep.find_crossing_pairs(shared_left) != []


def test_multiplet_drop_leaves_a_consistent_record(tmp_path):
    """End-to-end: the rendered structure and the pair list must agree."""
    # seq[9] = "U" makes both (2,9) and (4,9) canonical A-U pairs, so the only
    # reason either can disappear is the multiplet rule.
    seq = "A" * 9 + "U" + "A" * 2
    path = _write_plk(tmp_path / "d.plk", {
        "train": [(seq, [(2, 9), (4, 9)], [0, 0], None)]})
    out = tmp_path / "out.jsonl"
    prep.main(["--ref-plk", str(path), "--ref-plk-set", "train",
               "--min-loop-policy", "drop", "--pseudoknot-policy", "drop",
               "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["pairs"] == [[2, 9]]
    assert record["structure"].count("(") == len(record["pairs"])


def test_read_ref_plk_selects_by_set_column(tmp_path):
    path = _write_plk(tmp_path / "d.plk", {
        "train": [("AUGCAUGC", [(0, 7), (1, 6)], [0, 0], None)],
        "valid": [("AUGCAUGC", [(0, 7)], [0], None)],
    })
    rows = list(prep.read_ref_plk(str(path), set_name="valid",
                                  name_col="Id", pair_type_policy="keep"))
    assert [name for name, _ in rows] == ["valid#valid_0"]
    assert rows[0][1][1] == [(0, 7)]


def test_read_ref_plk_selects_by_dict_key(tmp_path):
    path = _write_plk(tmp_path / "t.plk", {
        "pdb_ts1": [("AUGCAUGC", [(0, 7)], [0], None)],
        "pdb_ts2": [("AUGCAUGC", [(0, 7)], [0], None)],
    })
    rows = list(prep.read_ref_plk(str(path), set_name="pdb_ts1",
                                  pair_type_policy="keep"))
    assert len(rows) == 1 and rows[0][0].startswith("pdb_ts1#")


def test_read_ref_plk_unknown_set_yields_nothing(tmp_path):
    path = _write_plk(tmp_path / "d.plk", {
        "train": [("AUGCAUGC", [(0, 7)], [0], None)]})
    assert list(prep.read_ref_plk(str(path), set_name="nope")) == []


def test_read_ref_plk_uses_pk_labels_not_crossing_heuristic(tmp_path):
    """pk labelled pairs go, and the nested pair implicated in no crossing stays."""
    path = _write_plk(tmp_path / "d.plk", {
        "train": [("AAAAAAAAGGGG", [(0, 11), (1, 10), (2, 6), (3, 7)],
                   [0, 0, 1, 1], None)]})
    rows = list(prep.read_ref_plk(str(path), set_name="train",
                                  pair_type_policy="keep"))
    assert rows[0][1][1] == [(0, 11), (1, 10)]


def test_read_ref_plk_canonical_drop_is_the_default(tmp_path):
    """The default must filter pair *type*, not only pseudoknots.

    Every pair here is nested (no ``pk`` label) and every one is non-canonical,
    so a reader that ignored pair type would report two pairs where the model
    can only ever emit zero.
    """
    path = _write_plk(tmp_path / "d.plk", {
        "train": [("AAAAAAAAAAAA", [(0, 11), (1, 10)], [0, 0], None)]})
    rows = list(prep.read_ref_plk(str(path), set_name="train"))
    assert rows[0][1][1] == []


def test_cli_ref_plk_end_to_end(tmp_path):
    # (0,11) A-U, (1,10) U-G, (2,9) G-C: all canonical, all nested
    seq = "AUG" + "A" * 6 + "CGU"
    structure = "(((" + "." * 6 + ")))"
    path = _write_plk(tmp_path / "d.plk", {
        "train": [(seq, [(0, 11), (1, 10), (2, 9)], [0, 0, 0], structure)]})
    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "m.json"
    rc = prep.main(["--ref-plk", str(path), "--ref-plk-set", "train",
                    "--min-loop-policy", "drop", "--pseudoknot-policy", "drop",
                    "--out", str(out), "--manifest", str(manifest)])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["pairs"] == [[0, 11], [1, 10], [2, 9]]
    assert rows[0]["structure"] == structure
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_kind"] == "ref_plk"
    assert data["provenance"]["pair_type_policy"] == "canonical-drop"
    assert data["reject_reasons"] == {}
    assert data["raw_annotation_stats"].get("structure_crosscheck_mismatch", 0) == 0


def test_cli_ref_plk_wrong_set_is_an_error(tmp_path, capsys):
    path = _write_plk(tmp_path / "d.plk", {
        "train": [("AUGCAUGC", [(0, 7)], [0], None)]})
    rc = prep.main(["--ref-plk", str(path), "--ref-plk-set", "wrong",
                    "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2
    assert "no rows selected" in capsys.readouterr().err


def test_cli_ref_plk_structure_crosscheck_is_reported_not_fatal(tmp_path):
    """A mangled `structure` column must be counted, not allowed to reject rows.

    Measured on the real release, ``pdb_ts1`` row ``632970`` has 37 index pairs
    but only 29 bracket pairs.  Rejecting on that disagreement would discard
    precisely the pseudoknot-heavy structures the benchmark exists to measure.
    """
    # (0,5) A-U and (1,6) U-G: canonical, nested, and nothing filtered -- so the
    # cross-check really does run.  An all-dots `structure` then contradicts it.
    seq = "AUGCAUGC"
    path = _write_plk(tmp_path / "d.plk", {
        "train": [(seq, [(0, 5), (1, 6)], [0, 0], "." * 8)]})
    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "m.json"
    rc = prep.main(["--ref-plk", str(path), "--ref-plk-set", "train",
                    "--min-loop-policy", "drop", "--pseudoknot-policy", "drop",
                    "--out", str(out), "--manifest", str(manifest)])
    assert rc == 0
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["n_accepted"] == 1
    assert data["reject_reasons"] == {}
    # the disagreement was recorded rather than dropped on the floor
    assert data["raw_annotation_stats"]["structure_crosscheck_mismatch"] == 1


def test_cli_ref_plk_crosscheck_skipped_when_pairs_were_filtered(tmp_path):
    """A row whose pairs were filtered cannot be cross-checked against a column
    that still contains them; that must be counted separately, not as a
    mismatch, or the diagnostic would report a defect that is really a policy."""
    seq = "AAAAAAAAAAAA"                 # every pair non-canonical
    path = _write_plk(tmp_path / "d.plk", {
        "train": [(seq, [(0, 11), (1, 10)], [0, 0], "((......))..")]})
    manifest = tmp_path / "m.json"
    rc = prep.main(["--ref-plk", str(path), "--ref-plk-set", "train",
                    "--min-loop-policy", "drop", "--pseudoknot-policy", "drop",
                    "--out", str(tmp_path / "o.jsonl"),
                    "--manifest", str(manifest)])
    assert rc == 0
    data = json.loads(manifest.read_text(encoding="utf-8"))
    stats = data["raw_annotation_stats"]
    assert stats["structure_crosscheck_skipped_filtered"] == 1
    assert stats.get("structure_crosscheck_mismatch", 0) == 0


def test_cli_ref_plk_crosscheck_reject_mode_still_available(tmp_path):
    seq = "AUGCAUGC"
    path = _write_plk(tmp_path / "d.plk", {
        "train": [(seq, [(0, 5), (1, 6)], [0, 0], "." * 8)]})
    rc = prep.main(["--ref-plk", str(path), "--ref-plk-set", "train",
                    "--structure-crosscheck", "reject",
                    "--min-loop-policy", "drop", "--pseudoknot-policy", "drop",
                    "--out", str(tmp_path / "o.jsonl"),
                    "--manifest", str(tmp_path / "m.json")])
    assert rc == 2


# ---------------------------------------------------------------------------
# the real RiNALMo format: 1-based base_pairs, extended (pseudoknot) brackets
#
# Both were found by running the parser on the actual ArchiveII.csv, where it
# rejected 3,859 of 3,864 rows.  The second one is the dangerous one: a parser
# that only knows () silently drops every pseudoknotted structure.
# ---------------------------------------------------------------------------
def test_csv_base_pairs_are_one_based():
    """The column is 1-based; the parser must shift to 0-based."""
    path = _write_csv(_tmp(), 'r1,ACGUACGU,"((....))","[[1, 8], [2, 7]]",8\n')
    rows = list(prep.read_rinalmo_csv(path))
    assert rows[0]["pairs"] == [(0, 7), (1, 6)]


def test_csv_rejects_zero_index_in_base_pairs():
    """A 0-based index would shift to -1 and must be refused, not wrapped."""
    path = _write_csv(_tmp(), 'r1,ACGUACGU,"((....))","[[0, 7]]",8\n')
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(path))
    assert info.value.reason == "csv_bad_base_pairs"


def test_extended_dotbracket_parses_pseudoknot_levels():
    """[] {} <> are successive pseudoknot levels in the standard notation."""
    # 11 chars: two nested () pairs, then a crossing pair marked with <>
    assert prep.dotbracket_to_pairs("((..<<.>>))", extended=True) == \
        [(0, 10), (1, 9), (4, 8), (5, 7)]


def test_extended_dotbracket_handles_all_four_families():
    assert prep.dotbracket_to_pairs("([{<..>}])", extended=True) == \
        [(0, 9), (1, 8), (2, 7), (3, 6)]


def test_strict_mode_still_rejects_extended_brackets():
    """Default stays strict, so the nested-only projection contract is unchanged."""
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("<<..>>")
    assert info.value.reason == "unknown_structure_char"


def test_extended_mode_rejects_genuinely_unknown_char():
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("((..ZZ))", extended=True)
    assert info.value.reason == "unknown_structure_char"


def test_extended_mode_rejects_unbalanced_family():
    with pytest.raises(prep.RejectError) as info:
        prep.dotbracket_to_pairs("<<..>", extended=True)
    assert info.value.reason == "unbalanced_bracket"


def test_csv_with_pseudoknot_brackets_is_accepted():
    """End-to-end: a row using <> must parse, not be rejected."""
    #           0123456789012345678901
    structure = "((..<<.>>))"                    # 11 chars
    # 0-based pairs (0,10) (1,9) (4,8) (5,7) -> 1-based [[1,11],[2,10],[5,9],[6,8]]
    body = ('pk,ACGUACGUACG,"%s","[[1, 11], [2, 10], [5, 9], [6, 8]]",11\n'
            % structure)
    path = _write_csv(_tmp(), body)
    rows = list(prep.read_rinalmo_csv(path))
    assert len(rows) == 1
    assert rows[0]["pairs"] == [(0, 10), (1, 9), (4, 8), (5, 7)]


def test_csv_disagreement_message_names_the_offending_pairs():
    """The message must say *which* pairs disagree, not just how many."""
    path = _write_csv(_tmp(), 'r1,ACGUACGU,"((....))","[[1, 8]]",8\n')
    with pytest.raises(prep.RejectError) as info:
        list(prep.read_rinalmo_csv(path))
    message = str(info.value)
    assert info.value.reason == "csv_pair_column_disagrees"
    assert "only in" in message


def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp(prefix="rna_csv_"))


# ---------------------------------------------------------------------------
# projection into the model's output space
#
# The PDB test sets are DSSR-derived 2D annotations of real 3D structures and
# genuinely contain pairs spanning loops of 0-2 bases.  Rejecting those records
# would discard 27/38 of ts2 and 16/18 of ts3, so the policy is to remove the
# offending pairs and *count* them.
# ---------------------------------------------------------------------------
def test_project_keeps_legal_structure_untouched():
    kept, illegal, crossing = prep.project_to_legal(12, [(0, 11), (1, 10)])
    assert kept == [(0, 11), (1, 10)]
    assert illegal == [] and crossing == []


def test_project_drops_pair_below_minimum_loop():
    # span 3 -> a two-base loop, below the physical minimum of three.
    kept, illegal, crossing = prep.project_to_legal(10, [(0, 3), (4, 9)])
    assert kept == [(4, 9)]
    assert illegal == [(0, 3)]
    assert crossing == []


def test_project_keeps_span_four():
    kept, illegal, crossing = prep.project_to_legal(10, [(0, 4)])
    assert kept == [(0, 4)] and illegal == [] and crossing == []


def test_project_resolves_pseudoknot_deterministically():
    kept, illegal, crossing = prep.project_to_legal(12, [(0, 7), (2, 11)])
    # both pairs participate in exactly one crossing; the larger span (0,7)
    # versus (2,11) -- (2,11) has span 9, (0,7) has span 7, so (2,11) goes.
    assert kept == [(0, 7)]
    assert crossing == [(2, 11)]
    assert illegal == []


def test_project_result_is_always_nested_and_legal():
    pairs = [(0, 11), (1, 3), (2, 9), (5, 8)]
    kept, illegal, crossing = prep.project_to_legal(12, pairs)
    assert prep.find_crossing_pairs(kept) == []
    assert all(j - i > 3 for i, j in kept)
    assert len(kept) + len(illegal) + len(crossing) == len(pairs)


def test_project_rejects_out_of_range_pair():
    with pytest.raises(prep.RejectError) as info:
        prep.project_to_legal(5, [(0, 9)])
    assert info.value.reason == "pair_out_of_range"


def test_project_is_deterministic_across_input_order():
    a = prep.project_to_legal(12, [(0, 7), (2, 11), (4, 6)])
    b = prep.project_to_legal(12, [(4, 6), (2, 11), (0, 7)])
    assert a == b


# ---------------------------------------------------------------------------
# CLI policies
# ---------------------------------------------------------------------------
def test_cli_min_loop_reject_is_the_default(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    # pair (0,3) spans a two-base loop -> illegal under the strict policy
    (corpus / "tight.bpseq").write_text("1 A 4\n2 C 0\n3 G 0\n4 U 1\n",
                                        encoding="utf-8")
    out = tmp_path / "o.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out)])
    assert out.read_text(encoding="utf-8").strip() == ""


def test_cli_min_loop_drop_keeps_record_and_counts_the_removal(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "tight.bpseq").write_text("1 A 4\n2 C 0\n3 G 0\n4 U 1\n",
                                        encoding="utf-8")
    out = tmp_path / "o.jsonl"
    manifest = tmp_path / "o.manifest.json"
    rc = prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
                    "--manifest", str(manifest), "--min-loop-policy", "drop"])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["pairs"] == []
    assert records[0]["n_dropped_illegal"] == 1
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["n_dropped_illegal_pairs"] == 1
    assert meta["n_records_with_drops"] == 1
    assert meta["min_loop_policy"] == "drop"


def test_cli_pseudoknot_drop_projects_into_nested_space(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    # (0,7) and (2,11) cross
    rows = [(1, "A", 8), (2, "C", 0), (3, "G", 12), (4, "U", 0),
            (5, "A", 0), (6, "C", 0), (7, "G", 0), (8, "U", 1),
            (9, "A", 0), (10, "C", 0), (11, "G", 0), (12, "U", 3)]
    (corpus / "pk.bpseq").write_text(
        "".join(f"{i} {b} {p}\n" for i, b, p in rows), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
               "--pseudoknot-policy", "drop"])
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["is_pseudoknot"] is False
    assert records[0]["n_dropped_crossing"] == 1


def test_cli_pseudoknot_reject_discards_record(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    rows = [(1, "A", 8), (2, "C", 0), (3, "G", 12), (4, "U", 0),
            (5, "A", 0), (6, "C", 0), (7, "G", 0), (8, "U", 1),
            (9, "A", 0), (10, "C", 0), (11, "G", 0), (12, "U", 3)]
    (corpus / "pk.bpseq").write_text(
        "".join(f"{i} {b} {p}\n" for i, b, p in rows), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
               "--pseudoknot-policy", "reject"])
    assert out.read_text(encoding="utf-8").strip() == ""


def test_cli_pseudoknot_keep_flags_record(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    rows = [(1, "A", 8), (2, "C", 0), (3, "G", 12), (4, "U", 0),
            (5, "A", 0), (6, "C", 0), (7, "G", 0), (8, "U", 1),
            (9, "A", 0), (10, "C", 0), (11, "G", 0), (12, "U", 3)]
    (corpus / "pk.bpseq").write_text(
        "".join(f"{i} {b} {p}\n" for i, b, p in rows), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
               "--pseudoknot-policy", "keep"])
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["is_pseudoknot"] is True
    assert records[0]["n_pairs"] == 2


# ---------------------------------------------------------------------------
# robustness: a bad member file must not abort the whole build
# ---------------------------------------------------------------------------
def test_empty_bpseq_file_is_rejected_not_fatal(tmp_path):
    """An empty .bpseq must become a counted reject, not a crash.

    Regression: the Rfam12.3-14.10 run died here, producing a 0-byte corpus.
    """
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "a_valid.bpseq").write_text("1 A 0\n2 C 0\n", encoding="utf-8")
    (corpus / "b_empty.bpseq").write_text("", encoding="utf-8")
    (corpus / "c_valid.bpseq").write_text("1 G 5\n2 C 0\n3 A 0\n4 A 0\n5 G 1\n",
                                          encoding="utf-8")

    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "out.manifest.json"
    rc = prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
                    "--manifest", str(manifest)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # the two good files survive even though the middle one is empty
    assert {r["name"] for r in records} == {"a_valid.bpseq", "c_valid.bpseq"}

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["n_accepted"] == 2
    assert meta["n_rejected"] == 1
    assert meta["reject_reasons"] == {"empty": 1}
    assert meta["n_accepted"] + meta["n_rejected"] == meta["provenance"]["n_files"] == 3


def test_malformed_bpseq_file_does_not_abort_others(tmp_path):
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "a.bpseq").write_text("1 A 0\n", encoding="utf-8")
    (corpus / "b.bpseq").write_text("1 A\n", encoding="utf-8")          # malformed row
    (corpus / "c.bpseq").write_text("1 A 2\n2 C 0\n", encoding="utf-8")  # asymmetric
    (corpus / "d.bpseq").write_text("1 A 0\n", encoding="utf-8")

    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "out.manifest.json"
    assert prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
                      "--manifest", str(manifest)]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["name"] for r in records} == {"a.bpseq", "d.bpseq"}
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["reject_reasons"] == {"bpseq_malformed": 1, "partner_asymmetric": 1}
    assert meta["n_accepted"] + meta["n_rejected"] == 4


def test_every_file_is_accounted_for_under_mixed_failures(tmp_path):
    """Conservation: accepted + rejected must equal the number of input files."""
    corpus = tmp_path / "bpseq"
    corpus.mkdir()
    (corpus / "ok1.bpseq").write_text("1 A 0\n2 C 0\n", encoding="utf-8")
    (corpus / "empty.bpseq").write_text("", encoding="utf-8")
    (corpus / "n_base.bpseq").write_text("1 A 0\n2 N 0\n", encoding="utf-8")
    (corpus / "ok2.bpseq").write_text("1 G 5\n2 C 0\n3 A 0\n4 A 0\n5 G 1\n",
                                      encoding="utf-8")
    (corpus / "short_hairpin.bpseq").write_text("1 A 4\n2 C 0\n3 G 0\n4 U 1\n",
                                                encoding="utf-8")
    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "out.manifest.json"
    prep.main(["--bpseq-dir", str(corpus), "--out", str(out),
               "--manifest", str(manifest)])
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["provenance"]["n_files"] == 5
    assert meta["n_accepted"] + meta["n_rejected"] == 5
    # the short-hairpin file is rejected under the default strict policy
    assert meta["reject_reasons"] == {"empty": 1, "bad_alphabet": 1,
                                     "hairpin_too_short": 1}


def test_bad_csv_row_does_not_abort_the_rest(tmp_path):
    path = tmp_path / "bench.csv"
    path.write_text(
        "id,sequence,structure,base_pairs,len\n"
        'good1,ACGUACGU,"((....))","[[1, 8], [2, 7]]",8\n'
        'badlen,ACGU,"....","[]",9\n'
        'good2,ACGUACGU,"((....))","[[1, 8], [2, 7]]",8\n',
        encoding="utf-8")
    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "out.manifest.json"
    assert prep.main(["--rinalmo-csv", str(path), "--out", str(out),
                      "--manifest", str(manifest)]) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["name"] for r in records} == {"good1", "good2"}
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["n_accepted"] == 2
    assert meta["n_rejected"] == 1
    assert meta["reject_reasons"] == {"csv_len_mismatch": 1}


# ---------------------------------------------------------------------------
# length bucketing must match the protocol frozen in the benchmark decision
# ---------------------------------------------------------------------------
def test_length_histogram_matches_protocol_buckets():
    hist = prep._length_histogram([50, 100, 150, 300, 500, 800, 1500])
    assert hist == {"<=100": 2, "<=200": 1, "<=400": 1, "<=600": 1,
                    "<=1000": 1, ">1000": 1}
