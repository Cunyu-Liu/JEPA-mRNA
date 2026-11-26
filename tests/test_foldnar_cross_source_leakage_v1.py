from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_foldnar_cross_source_leakage_v1 import (
    build_view,
    normalize_sequence,
    pair_table,
    parse_cluster_members,
    parse_m8,
    stable_id,
    validate_legal_source,
)


def maps():
    query = {"q": {"stable_id": "tr1c:q", "length": 100}}
    target = {"t": {"stable_id": "chanrg:t", "length": 100}}
    return query, target


def test_stable_id_and_normalization_match_registry_contract():
    row = {"id": "rna 1", "seq": "acg tu\n"}
    assert stable_id("chanrg_validation", 7, row["id"]) == "chanrg_validation:row000007:rna 1"
    assert normalize_sequence(row) == "ACG UU"


@pytest.mark.parametrize("name", ["chanrg_train_train.jsonl", "chanrg_train_validation.jsonl", "bprna_tr1c.jsonl"])
def test_legal_sources_are_explicitly_allowlisted(tmp_path, name):
    validate_legal_source(tmp_path / name, name)


@pytest.mark.parametrize("name", ["chanrg_test.jsonl", "chanrg_gena.jsonl", "chanrg_genc.jsonl", "chanrg_genf.jsonl"])
def test_external_test_sources_are_rejected(tmp_path, name):
    with pytest.raises(AssertionError):
        validate_legal_source(tmp_path / name, "chanrg_train_train.jsonl")


def test_m8_requires_identity_and_bidirectional_coverage(tmp_path):
    query_map, target_map = maps()
    path = tmp_path / "hits.m8"
    path.write_text("q\tt\t0.95\t100\t0\t0\t0\t78\t0\t99\t1e-20\t200\n")
    assert parse_m8(path, query_map, target_map) == []
    path.write_text("q\tt\t0.80\t80\t0\t0\t0\t79\t0\t79\t1e-20\t200\n")
    hits = parse_m8(path, query_map, target_map)
    assert len(hits) == 1
    assert hits[0]["query_coverage"] == 0.8
    assert hits[0]["target_coverage"] == 0.8


def test_pair_table_unions_directions_without_double_counting():
    forward = [{"query_id": "tr1c:q", "target_id": "chanrg:t", "identity": 0.9,
                "query_coverage": 0.9, "target_coverage": 0.8, "evalue": 1e-8}]
    reverse = [{"target_id": "tr1c:q", "query_id": "chanrg:t", "identity": 0.91,
                "query_coverage": 0.8, "target_coverage": 0.9, "evalue": 1e-9}]
    rows = pair_table(forward, reverse)
    assert len(rows) == 1
    assert rows[0]["forward_alignments"] == 1
    assert rows[0]["reverse_alignments"] == 1
    assert rows[0]["max_identity"] == 0.91
    assert rows[0]["best_evalue"] == 1e-9


def cluster_fixture():
    validation = ["chanrg_validation:row000000:a", "chanrg_validation:row000001:b",
                  "chanrg_validation:row000002:c", "chanrg_validation:row000003:d"]
    stable_cluster = {validation[0]: "r0", validation[1]: "r0", validation[2]: "r1", validation[3]: "r2"}
    clusters = {
        "r0": {validation[0], validation[1], "chanrg_train:row000000:x"},
        "r1": {validation[2], "chanrg_train:row000001:y"},
        "r2": {validation[3]},
    }
    return validation, stable_cluster, clusters


def test_cluster_closure_excludes_nonhit_cluster_mate_but_remains_unknown():
    validation, stable_cluster, clusters = cluster_fixture()
    pairs = [{"tr1c_id": "tr1c:q", "chanrg_validation_id": validation[0]}]
    result = build_view("raw", validation, [], pairs, stable_cluster, clusters)
    assert result["direct_validation"] == {validation[0]}
    assert result["excluded"] == {validation[0], validation[1]}
    assert result["remaining"] == validation[2:]
    assert result["train_members"] == {"chanrg_train:row000000:x"}


def test_matched_view_is_subset_and_no_hit_is_not_route_eligibility():
    validation, stable_cluster, clusters = cluster_fixture()
    result = build_view("matched_clean", validation, validation[2:], [], stable_cluster, clusters)
    assert result["selected_order"] == validation[2:]
    assert result["direct_validation"] == set()
    assert result["excluded"] == set()
    assert result["remaining"] == validation[2:]


def test_cluster_parser_maps_all_train_validation_members(tmp_path):
    chanrg = {
        "chanrg_train": [{"stable_id": "chanrg_train:row000000:x", "source_row_index": 0}],
        "chanrg_validation": [{"stable_id": "chanrg_validation:row000000:a", "source_row_index": 0}],
    }
    path = tmp_path / "cluster.tsv"
    path.write_text("c0\tc00000000_chanrg_train\nc0\tc00000000_chanrg_validation\n")
    stable_cluster, clusters = parse_cluster_members(path, chanrg)
    assert stable_cluster["chanrg_train:row000000:x"] == "c0"
    assert stable_cluster["chanrg_validation:row000000:a"] == "c0"
    assert len(clusters["c0"]) == 2
