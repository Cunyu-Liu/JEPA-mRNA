import numpy as np

from tools.pair_decode_audit import (apply_external, entries_for_file, make_ranking_summary,
                                     pr_thresholds, selection_row, tune_vl0)


def make_raw_file(path, seed=1):
    rng = np.random.default_rng(seed)
    seq = "AUGCUAGCUAGC"
    n = len(seq)
    scores = np.triu(rng.normal(size=(n, n)), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, name=np.array(f"seq{seed}"), seq=np.array(seq),
                        sequence_sha256=np.array(f"hash{seed}"), selection_sha256=np.array(f"sel{seed}"),
                        gt_pairs=np.array([[0, 5], [1, 4]], dtype=np.int32), r2d_on=scores)


def test_pr_thresholds_and_selection_counts():
    labels = np.array([True, True, False, False])
    scores = np.array([.9, .8, .7, .1])
    cfg = pr_thresholds(labels, scores)
    assert cfg["max_f1"]["recall"] == 1.0
    assert cfg["max_f1"]["precision"] >= 2 / 3
    entry = {"dataset": "d", "name": "n", "labels": labels, "n_gt": 3, "n_legal": 2}
    row = selection_row(entry, "v", "INDEPENDENT_RAW_MAXF1", 1.0, scores, .75)
    assert (row["tp"], row["fp"], row["fn"], row["n_pred"]) == (2, 0, 1, 2)


def test_raw_and_posterior_entries_are_not_duplicated(tmp_path):
    path = tmp_path / "bprna_vl0" / "a.npz"
    make_raw_file(path)
    dp, ranks, entries = entries_for_file(path, "bprna_vl0", (0.5, 1.0, 2.0, 4.0), True)
    assert len(entries) == 5
    assert sum(e["kind"] == "RAW_BASE" for e in entries) == 1
    assert sum(e["kind"] == "POSTERIOR" for e in entries) == 4
    assert len(ranks) == 5
    assert len(dp) == 1 + 4 * 6
    summary = make_ranking_summary(entries, ranks, "bprna_vl0")
    raw = next(r for r in summary if r["score_kind"] == "RAW")
    assert raw["n_legal_candidates"] == len(entries[0]["labels"])


def test_external_uses_frozen_vl0_configuration(tmp_path):
    make_raw_file(tmp_path / "raw/bprna_vl0/a.npz", 1)
    make_raw_file(tmp_path / "raw/bprna_vl0/b.npz", 2)
    make_raw_file(tmp_path / "raw/bprna_new/c.npz", 3)
    dp, ranks, entries, config = tune_vl0(tmp_path / "raw")
    assert config["r2d_on"]["marginal_independent"]["temperature"] in (0.5, 1.0, 2.0, 4.0)
    out_dp, out_ranks, out_entries = apply_external(tmp_path / "raw", "bprna_new", config)
    assert sum(r["decoder"] == "MEA" for r in out_dp) == 1
    assert sum(r["decoder"] == "CENTROID" for r in out_dp) == 1
    assert sum(r["decoder"] == "MAP" for r in out_dp) == 1
    assert sum(r["decoder"] == "INDEPENDENT_POST_MAXF1" for r in out_dp) == 1
    selected_t = config["r2d_on"]["marginal_independent"]["temperature"]
    assert next(r for r in out_dp if r["decoder"] == "INDEPENDENT_POST_MAXF1")["temperature"] == selected_t
