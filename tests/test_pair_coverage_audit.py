import random

from tools.pair_coverage_audit import (RangeTree, helix_bin, keys_for, normalize_pairs,
                                       pair_geometry, prf, span_bin)


def naive_crossing(pairs):
    ordered = sorted(pairs)
    result = {p: False for p in ordered}
    for a, (i, j) in enumerate(ordered):
        for x, y in ordered[a + 1:]:
            if i < x < j < y:
                result[(i, j)] = result[(x, y)] = True
    return result


def test_normalize_pairs_and_prf():
    pairs, duplicate, invalid = normalize_pairs([[3, 1], [1, 3], [-1, 2], [0, 0], "bad", [0, 3]], 5)
    assert pairs == {(1, 3), (0, 3)}
    assert duplicate == 1
    assert invalid == 3
    assert prf({(1, 3)}, {(1, 3), (0, 3)}) == {
        "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}


def test_mask_crossing_conflict_and_helix_features():
    seq = "AUAUAUAUAUAUAUAUAUAU"
    cross = pair_geometry(seq, {(0, 9), (4, 14)})
    assert cross[(0, 9)]["crossing"] and cross[(4, 14)]["crossing"]
    pairs = {(0, 9), (1, 8), (2, 7)}
    f = pair_geometry(seq, pairs)
    assert not any(f[p]["crossing"] for p in pairs)
    assert f[(0, 9)]["inner_step"] and f[(0, 9)]["outer_step"] is False
    assert f[(1, 8)]["helix_length"] == 3
    assert f[(2, 7)]["terminal_hairpin"]
    assert f[(2, 7)]["hairpin_loop_length"] == 4
    assert f[(2, 7)]["span"] == 5
    conflict = pair_geometry(seq, {(0, 9), (0, 10)})
    assert conflict[(0, 9)]["partner_conflict"] and conflict[(0, 10)]["partner_conflict"]
    assert span_bin(3) == "0_3" and span_bin(4) == "4_9" and span_bin(200) == "200_plus"
    assert helix_bin(1) == "1" and helix_bin(13) == "13_plus"


def test_range_tree_against_naive():
    rng = random.Random(7)
    for n in range(1, 20):
        values = [rng.randrange(100) for _ in range(n)]
        tree = RangeTree(values)
        for left in range(n + 1):
            for right in range(left, n + 1):
                expected_min = min(values[left:right]) if left < right else float("inf")
                expected_max = max(values[left:right]) if left < right else -float("inf")
                assert tree.query(left, right) == expected_min
                assert tree.query(left, right, maximum=True) == expected_max


def test_crossing_flags_match_brute_force_on_random_pair_sets():
    rng = random.Random(19)
    for n in range(2, 15):
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for _ in range(30):
            pairs = {p for p in candidates if rng.random() < .16}
            if not pairs:
                continue
            got = pair_geometry("A" * n, pairs)
            expected = naive_crossing(pairs)
            assert all(got[p]["crossing"] == expected[p] for p in pairs)


def test_environment_key_is_stable_and_typed():
    seq = "GCAUAUAAAAAAAAAGCAU"
    pairs = {(0, 9), (1, 8)}
    f = pair_geometry(seq, pairs)[(1, 8)]
    k1, k2, k3, k4 = keys_for(f)
    assert k1 == f["pair_type"]
    assert k2.startswith(k1 + "|span=")
    assert "|helix=" in k3 and "|reg=" in k3 and "|loop=" in k3
    assert f"|span={f['span']}|helix={f['helix_length']}" in k4
