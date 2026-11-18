from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rnajepa.fold_structure import (  # noqa: E402
    StructureError,
    build_helix_tree,
    canonical_pairs,
    decompose_loops,
    extract_stems,
    pair_prf,
    structure_metrics,
    topology_ged,
    topology_signature,
)


def test_known_pair_and_stem_values():
    truth = [(0, 9), (1, 8), (2, 7), (12, 15)]
    pred = [(0, 9), (1, 8), (12, 15), (4, 5)]
    pair = pair_prf(pred, truth)
    assert (pair["tp"], pair["fp"], pair["fn"]) == (3, 1, 1)
    assert math.isclose(pair["f1"], 0.75)
    metrics = structure_metrics(pred, truth, length=16)
    assert metrics["n_truth_stems"] == 2
    assert metrics["stem"]["tp"] == 1
    assert metrics["stem"]["fp"] == 2
    assert metrics["stem"]["fn"] == 1


def test_loop_types_and_topology_not_coordinates():
    hairpin = [(0, 9), (1, 8), (2, 7)]
    internal = [(0, 9), (3, 6)]
    bulge = [(0, 9), (1, 6)]
    multiloop = [(0, 13), (2, 5), (7, 10)]
    assert [x.kind for x in decompose_loops(hairpin)] == ["EXTERNAL", "HAIRPIN"]
    assert [x.kind for x in decompose_loops(internal)] == ["EXTERNAL", "INTERNAL", "HAIRPIN"]
    assert [x.kind for x in decompose_loops(bulge)] == ["EXTERNAL", "BULGE", "HAIRPIN"]
    assert [x.kind for x in decompose_loops(multiloop)] == ["EXTERNAL", "MULTILOOP", "HAIRPIN", "HAIRPIN"]
    shifted = [(20, 29), (21, 28), (22, 27)]
    assert topology_signature(build_helix_tree(hairpin)) == topology_signature(build_helix_tree(shifted))
    assert topology_ged(build_helix_tree(hairpin), build_helix_tree(shifted)) == 0
    assert structure_metrics(hairpin, shifted)["stem"]["f1"] == 0.0


def test_wiring_counts_all_helix_parent_child_edges():
    truth = [(0, 13), (2, 5), (7, 10)]
    pred = [(0, 13), (2, 5)]
    metrics = structure_metrics(pred, truth)
    assert (metrics["wiring"]["tp"], metrics["wiring"]["fp"], metrics["wiring"]["fn"]) == (2, 0, 1)
    assert metrics["junction"]["fn"] == 2


def test_known_tree_edit_distances():
    empty = build_helix_tree([])
    hairpin = build_helix_tree([(0, 9), (1, 8)])
    internal = build_helix_tree([(0, 9), (3, 6)])
    two = build_helix_tree([(0, 5), (8, 13)])
    assert topology_ged(empty, hairpin) == 1
    assert topology_ged(hairpin, internal) == 1
    assert topology_ged(hairpin, two) == 1
    assert topology_ged(two, hairpin) == 1
    assert topology_ged(hairpin, hairpin) == 0


def test_invalid_structures_are_not_silently_repaired():
    with pytest.raises(StructureError):
        canonical_pairs([(0, 5), (0, 6)])
    with pytest.raises(StructureError):
        canonical_pairs([(0, 5), (2, 7)])
    with pytest.raises(StructureError):
        canonical_pairs([(0, 0)])
    with pytest.raises(StructureError):
        canonical_pairs([(0, 4)], length=4)


def _noncrossing_matchings(length: int):
    pairs = [(i, j) for i in range(length) for j in range(i + 2, length)]

    def valid(selected):
        used = set()
        for i, j in selected:
            if i in used or j in used:
                return False
            used.update((i, j))
        for a, b in itertools.combinations(selected, 2):
            i, j = sorted(a)
            k, l = sorted(b)
            if i < k < j < l or k < i < l < j:
                return False
        return True

    # Exhaustive subsets are practical through length six (15 candidate pairs).
    for mask in range(1 << len(pairs)):
        selected = [pairs[k] for k in range(len(pairs)) if mask & (1 << k)]
        if valid(selected):
            yield selected


def test_small_exhaustive_pair_and_structure_invariants():
    rng = random.Random(20261118)
    structures = list(_noncrossing_matchings(6))
    assert len(structures) == 17
    for truth in structures:
        identity = structure_metrics(truth, truth, length=6)
        assert identity["pair"]["f1"] == 1.0
        assert identity["stem"]["f1"] == 1.0
        assert identity["loop"]["f1"] == 1.0
        assert identity["wiring"]["f1"] == 1.0
        assert identity["topology"]["ged"] == 0
        assert identity["topology"]["exact"] is True
        assert sum(identity["truth_loop_counts"].values()) == identity["n_truth_stems"] + 1
    for _ in range(150):
        a, b = rng.choice(structures), rng.choice(structures)
        ab = structure_metrics(a, b, length=6)
        ba = structure_metrics(b, a, length=6)
        assert ab["topology"]["ged"] == ba["topology"]["ged"]
        assert 0.0 <= ab["topology"]["f1"] <= 1.0
        assert 0.0 <= ab["pair"]["f1"] <= 1.0
        assert 0.0 <= ab["stem"]["f1"] <= 1.0


def test_topology_ged_matches_independent_zss_implementation():
    from zss import Node, simple_distance

    def convert(node):
        return Node(node.loop_kind, [convert(child) for child in node.children])

    def independent(roots_a, roots_b):
        return simple_distance(
            Node("EXTERNAL", [convert(node) for node in roots_a]),
            Node("EXTERNAL", [convert(node) for node in roots_b]),
        )

    rng = random.Random(718)
    structures = list(_noncrossing_matchings(6))
    for _ in range(150):
        a, b = rng.choice(structures), rng.choice(structures)
        roots_a, roots_b = build_helix_tree(a), build_helix_tree(b)
        assert topology_ged(roots_a, roots_b) == independent(roots_a, roots_b)
