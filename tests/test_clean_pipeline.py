"""End-to-end tests for the C1-C6 cleaning pipeline.

Everything here runs on **synthetic in-memory records** — no external data and no
network access.  The tests assert the audit guarantees that make the dataset
trustworthy for the paper:

* ``T -> U`` and IUPAC handling on crafted edge cases;
* every rejection carries a reason code (no silent drops);
* the attrition table satisfies the conservation identity at every level;
* bracket validation rejects unbalanced / min-hairpin / crossing structures and
  routes pseudoknots to a separate set;
* contamination detection removes crafted exact and near duplicates;
* family-level splitting produces zero family overlap.

Run:  python -m pytest tests/test_clean_pipeline.py -v
  or:  python tests/test_clean_pipeline.py
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from rnajepa.clean import (  # noqa: E402
    CleanConfig,
    ReasonCode,
    Record,
    clean_sequence,
    detect_contamination,
    family_group_split,
    family_overlap,
    normalize_sequence,
    run_pipeline,
    validate_structure,
)
from rnajepa.clean.c6_splits import freeze_family_split  # noqa: E402

# A 12-nt structured head: three nested G-C pairs with a 6-nt hairpin loop.
HEAD_SEQ = "GGGGAAAACCCC"
HEAD_STRUCT = "(((......)))"


def _tail(idx, length=28):
    """Deterministic pseudo-random tail so every synthetic record is unique."""
    rng = random.Random(idx)
    return "".join(rng.choice("ACGU") for _ in range(length))


def structured_record(rid, idx, family=None, label_source="SHAPE", source="synthetic", **meta):
    """A valid 40-nt record: structured head + unique random tail."""
    tail = _tail(idx)
    return Record(
        id=rid,
        sequence=HEAD_SEQ + tail,
        structure=HEAD_STRUCT + "." * len(tail),
        family=family,
        source=source,
        licence="cc-by-4.0",
        label_source=label_source,
        meta=dict(meta),
    )


# ---------------------------------------------------------------------------
# 1. C1 sequence layer: T->U and IUPAC handling
# ---------------------------------------------------------------------------


def test_t_to_u_normalisation():
    normalized, mask, invalid = normalize_sequence("ACGTacgt ACGU")
    assert normalized == "ACGUACGUACGU"
    assert set(mask) == {"0"}
    assert invalid == []

    record, removal = clean_sequence(Record(id="t2u", sequence="ACGTACGTACGT"))
    assert removal is None
    assert record.seq_norm == "ACGUACGUACGU"
    assert record.meta["t_to_u_converted"] == 3
    assert record.fingerprint and len(record.fingerprint) == 64


def test_iupac_ambiguity_is_masked_not_dropped():
    normalized, mask, invalid = normalize_sequence("ACGTRACGUACGU")
    assert invalid == []
    assert normalized == "ACGUNACGUACGU"
    assert mask == "0000100000000"
    assert normalized[4] == "N"

    record, removal = clean_sequence(Record(id="iupac", sequence="ACGTRACGUACGU"))
    assert removal is None
    assert record.ambiguity_mask.count("1") == 1
    assert record.meta["ambiguous_positions"] == 1


def test_all_n_is_rejected_as_ambiguous():
    record, removal = clean_sequence(Record(id="alln", sequence="N" * 50))
    assert record is None
    assert removal.reason is ReasonCode.AMBIGUOUS_FRACTION


def test_empty_sequence_is_rejected():
    for raw in ("", "   "):
        record, removal = clean_sequence(Record(id="empty", sequence=raw))
        assert record is None
        assert removal.reason is ReasonCode.EMPTY_SEQUENCE


def test_non_acgu_characters_are_rejected():
    record, removal = clean_sequence(Record(id="bad", sequence="ACGTXACGTXACGTX"))
    assert record is None
    assert removal.reason is ReasonCode.NON_IUPAC_CHARACTER
    assert "X" in removal.detail


def test_very_short_and_very_long_are_rejected_with_distinct_codes():
    _, removal = clean_sequence(Record(id="short", sequence="ACGU"))
    assert removal.reason is ReasonCode.TOO_SHORT

    _, removal = clean_sequence(Record(id="long", sequence="ACGU" * 200), max_len=100)
    assert removal.reason is ReasonCode.TOO_LONG

    # Boundary: exactly min_len passes.
    record, removal = clean_sequence(Record(id="edge", sequence="ACGUCAGUCA"))
    assert removal is None and record is not None


def test_low_complexity_hook_is_documented_stub():
    from rnajepa.clean import LOW_COMPLEXITY_IS_STUB

    assert LOW_COMPLEXITY_IS_STUB is True
    _, removal = clean_sequence(Record(id="homopolymer", sequence="A" * 60))
    assert removal.reason is ReasonCode.LOW_COMPLEXITY
    assert "stub detector" in removal.detail


# ---------------------------------------------------------------------------
# 2. C3 structure layer: bracket validation
# ---------------------------------------------------------------------------


def test_validate_valid_structure_and_non_canonical_annotation():
    check = validate_structure(
        Record(id="ok", sequence=HEAD_SEQ, structure=HEAD_STRUCT)
    )
    assert check.ok
    assert sorted(check.pairs) == [(0, 11), (1, 10), (2, 9)]
    assert check.non_canonical == []

    check = validate_structure(
        Record(id="noncanon", sequence="GAUAAAAACUC", structure="(((.....)))")
    )
    assert check.ok
    assert check.non_canonical == [(2, 8)]


def test_validate_rejects_unbalanced_structure():
    check = validate_structure(Record(id="unbalanced", sequence="GCGC", structure="(..."))
    assert not check.ok
    assert check.reason is ReasonCode.STRUCTURE_UNBALANCED


def test_validate_rejects_min_hairpin_violation():
    check = validate_structure(Record(id="hairpin", sequence="GCGC", structure="(..)"))
    assert not check.ok
    assert check.reason is ReasonCode.STRUCTURE_MIN_HAIRPIN


def test_validate_rejects_crossing_pairs():
    check = validate_structure(
        Record(id="cross", sequence="GCAAGUCCAAUG", structure="(..[...)..].")
    )
    assert not check.ok
    assert check.reason is ReasonCode.STRUCTURE_CROSSING
    assert set(check.crossing) == {(0, 7), (3, 10)}


def test_validate_rejects_length_mismatch_and_missing_structure():
    check = validate_structure(Record(id="mismatch", sequence="GCGCAAAAGC", structure="(...)"))
    assert check.reason is ReasonCode.LENGTH_MISMATCH

    check = validate_structure(Record(id="nostruct", sequence="GCGCAAAAGC", structure=None))
    assert check.reason is ReasonCode.STRUCTURE_MISSING


def test_pseudoknots_are_routed_to_a_separate_set():
    crossing = Record(
        id="cross",
        sequence="GCAAGUCCAAUG" + _tail(1),
        structure="(..[...)..]." + "." * len(_tail(1)),
        family="RF1",
        label_source="SHAPE",
    )
    good = structured_record("good", 2, family="RF2")
    config = CleanConfig(min_len=10, max_len=100, dry_run=True)
    result = run_pipeline([crossing, good], config)

    assert [r.id for r in result.pseudoknots] == ["cross"]
    assert [r.id for r in result.kept] == ["good"]
    routed = [r for r in result.removals if r.routed]
    assert len(routed) == 1 and routed[0].reason is ReasonCode.STRUCTURE_CROSSING
    # Routed records still count as removed at that level (conservation holds).
    result.attrition.assert_conservation()


# ---------------------------------------------------------------------------
# 3. C2 redundancy / contamination
# ---------------------------------------------------------------------------


def _mutate(sequence, positions):
    bases = list(sequence)
    for pos in positions:
        bases[pos] = {"A": "C", "C": "A", "G": "U", "U": "G"}[bases[pos]]
    return "".join(bases)


def test_contamination_removes_exact_and_near_duplicates():
    exact = structured_record("exact", 100)
    pretrain_source = structured_record("pretrain_src", 101)
    near = Record(
        id="near",
        sequence=_mutate(pretrain_source.sequence, [30, 31, 32]),
        structure=pretrain_source.structure,
        family="RF1",
        label_source="SHAPE",
    )
    unrelated = structured_record("unrelated", 102)

    pretrain = [exact.sequence, pretrain_source.sequence]
    kept, removals, report = detect_contamination(
        [exact, near, unrelated], pretrain, identity=0.8, dry_run=True
    )

    assert [r.id for r in kept] == ["unrelated"]
    reasons = {r.record_id: r.reason for r in removals}
    assert reasons["exact"] is ReasonCode.PRETRAIN_CONTAMINATION_EXACT
    assert reasons["near"] is ReasonCode.PRETRAIN_CONTAMINATION_NEAR
    assert report["exact_hits"] == 1 and report["near_hits"] == 1
    assert abs(report["contamination_rate"] - 2 / 3) < 1e-9
    assert report["mode"] == "dry_run_approximation"


def test_contamination_detection_integrated_in_pipeline():
    exact = structured_record("exact", 100)
    pretrain_source = structured_record("pretrain_src", 101)
    near = Record(
        id="near",
        sequence=_mutate(pretrain_source.sequence, [30, 31, 32]),
        structure=pretrain_source.structure,
        family="RF1",
        label_source="SHAPE",
    )
    unrelated = structured_record("unrelated", 102)

    # Internal dedup is disabled so that contamination is what removes the hits.
    config = CleanConfig(min_len=10, max_len=100, dedup_identities=(), dry_run=True)
    result = run_pipeline(
        [exact, near, unrelated],
        config,
        pretrain_sequences=[exact.sequence, pretrain_source.sequence],
    )
    assert [r.id for r in result.kept] == ["unrelated"]
    assert result.contamination["removed"] == 2
    result.attrition.assert_conservation()


def test_deduplication_requires_the_real_tool_unless_dry_run():
    from rnajepa.clean import ToolUnavailableError, deduplicate

    records = [structured_record("a", 1), structured_record("b", 2)]
    try:
        deduplicate(records, 0.9, tool="definitely-not-installed", dry_run=False)
    except ToolUnavailableError as exc:
        assert "dry_run=True" in str(exc)
    else:  # pragma: no cover - only if the tool happens to exist
        raise AssertionError("expected ToolUnavailableError")

    kept, removals = deduplicate(
        records, 0.9, tool="definitely-not-installed", dry_run=True
    )
    assert len(kept) + len(removals) == len(records)


# ---------------------------------------------------------------------------
# 4. C2 family-level split
# ---------------------------------------------------------------------------


def test_family_split_has_zero_family_overlap():
    records = []
    for family, size in (("RF1", 4), ("RF2", 4), ("RF3", 3), ("RF4", 3), ("RF5", 2)):
        for i in range(size):
            records.append(structured_record(f"{family}_{i}", len(records), family=family))

    splits = family_group_split(records, seed=0)
    assert family_overlap(splits) == {}
    assert sum(len(v) for v in splits.values()) == len(records)
    assert set(splits) == {"train", "dev", "test"}
    # Every split is non-empty: with >= 3 families the greedy assignment must not
    # collapse dev/test into train.
    assert all(len(v) > 0 for v in splits.values()), splits

    frozen = freeze_family_split(records, "unit", seed=0)
    assert frozen.sizes == {k: len(v) for k, v in splits.items()}
    assert frozen.digest()  # hashed, reproducible


def test_family_split_is_seed_reproducible():
    records = []
    for family in ("RF1", "RF2", "RF3", "RF4", "RF5", "RF6"):
        for i in range(3):
            records.append(structured_record(f"{family}_{i}", len(records), family=family))

    a = family_group_split(records, seed=0)
    b = family_group_split(records, seed=0)
    assert {k: [r.id for r in v] for k, v in a.items()} == {
        k: [r.id for r in v] for k, v in b.items()
    }


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline: reason codes + attrition conservation
# ---------------------------------------------------------------------------


def _mixed_records():
    records = [structured_record(f"good_{i}", i, family=f"RF{i}") for i in range(4)]

    records.append(Record(id="empty", sequence=""))
    records.append(Record(id="all_n", sequence="N" * 50))
    records.append(Record(id="badchar", sequence="ACGTXACGTXACGTX"))
    records.append(Record(id="short", sequence="ACGU"))
    records.append(Record(id="long", sequence="ACGU" * 40))

    tail_a, tail_b, tail_c, tail_d = _tail(50), _tail(51), _tail(52), _tail(53)
    records.append(
        Record(id="unbalanced", sequence=HEAD_SEQ + tail_a, structure="(((......))." + "." * len(tail_a))
    )
    records.append(
        Record(id="minhairpin", sequence=HEAD_SEQ + tail_b, structure="(..)........" + "." * len(tail_b))
    )
    records.append(
        Record(
            id="crossing",
            sequence="GCAAGUCCAAUG" + tail_c,
            structure="(..[...)..]." + "." * len(tail_c),
        )
    )
    records.append(
        Record(id="mismatch", sequence=HEAD_SEQ + tail_d, structure=HEAD_STRUCT + "." * (len(tail_d) - 1))
    )
    return records


def test_pipeline_every_rejection_has_a_reason_code():
    records = _mixed_records()
    result = run_pipeline(records, CleanConfig(min_len=10, max_len=100, dry_run=True))

    assert len(result.removals) == len(records) - len(result.kept)
    for removal in result.removals:
        assert isinstance(removal.reason, ReasonCode)
        assert removal.detail, f"{removal.record_id} has no diagnostic detail"

    reasons = {r.record_id: r.reason for r in result.removals}
    assert reasons["empty"] is ReasonCode.EMPTY_SEQUENCE
    assert reasons["all_n"] is ReasonCode.AMBIGUOUS_FRACTION
    assert reasons["badchar"] is ReasonCode.NON_IUPAC_CHARACTER
    assert reasons["short"] is ReasonCode.TOO_SHORT
    assert reasons["long"] is ReasonCode.TOO_LONG
    assert reasons["unbalanced"] is ReasonCode.STRUCTURE_UNBALANCED
    assert reasons["minhairpin"] is ReasonCode.STRUCTURE_MIN_HAIRPIN
    assert reasons["crossing"] is ReasonCode.STRUCTURE_CROSSING
    assert reasons["mismatch"] is ReasonCode.LENGTH_MISMATCH


def test_pipeline_attrition_conservation_holds_at_every_level():
    records = _mixed_records()
    result = run_pipeline(records, CleanConfig(min_len=10, max_len=100, dry_run=True))

    result.attrition.assert_conservation()

    levels = result.attrition.levels
    assert levels[0].level == "raw"
    assert levels[0].previous_count == len(records)
    for level in levels:
        assert level.previous_count == level.next_count + level.removed_count
        assert sum(level.removed_by_reason.values()) == level.removed_count
    # Levels are chained: each next_count feeds the following previous_count.
    for previous, current in zip(levels, levels[1:]):
        assert current.previous_count == previous.next_count

    assert len(result.kept) == 4
    assert [r.id for r in result.pseudoknots] == ["crossing"]
    assert levels[-1].next_count == len(result.kept)


def test_pipeline_report_and_data_card_render():
    from rnajepa.clean import render_data_card_from_result

    records = _mixed_records()
    result = run_pipeline(records, CleanConfig(min_len=10, max_len=100, dry_run=True))

    report = result.report_markdown()
    assert "## Attrition table" in report
    assert "C1_sequence" in report
    assert "C3_STRUCTURE_CROSSING" in report

    card = render_data_card_from_result(
        result, dataset_name="synthetic-unit-test", sources=["synthetic"]
    )
    assert "# Data card — synthetic-unit-test" in card
    assert "不含任何二级结构标注" in card
    assert "测试集**不参与任何超参数选择**" in card

    stats = result.stats
    assert stats["distribution"]["n"] == 4
    assert "split_distances" in stats
    assert stats["licences"]["sources_with_unknown_licence"] == []


def test_pipeline_does_not_mutate_caller_records():
    records = _mixed_records()
    before = [(r.id, r.seq_norm, r.fingerprint) for r in records]
    run_pipeline(records, CleanConfig(min_len=10, max_len=100, dry_run=True))
    after = [(r.id, r.seq_norm, r.fingerprint) for r in records]
    assert before == after


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if not failures else 'FAILURES'}: {failures} failing")
    sys.exit(1 if failures else 0)
