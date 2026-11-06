from append_round38_ownclean_tables import all_values, cell_values, load_source


def test_all_104_cells_and_empty_semantics():
    values = all_values(load_source())
    assert len(values) == 104
    empty = [value for value in values if value["status"] == "EMPTY_AFTER_DELEAK"]
    assert len(empty) == 2
    assert {(value["system"], value["split"]) for value in empty} == {
        ("RNAformer-intra", "ref_pdb_ts3"), ("SPOT-RNA", "ref_pdb_ts3")}
    for value in empty:
        assert value["clean_n"] == "0"
        assert all(value[key] == "—" for key in ["f1", "tp", "fp", "fn"])


def test_cell_mapping_preserves_counts_and_f1():
    cell = {"status": "PASS", "raw_n": 100, "clean_n": 80, "removed_exact": 5,
            "removed_homology_only": 15, "f1_values": [0.75] * 3, "f1_mean": 0.75,
            "tp": 50, "fp": 10, "fn": 20}
    assert cell_values("xens2", "bprna_ts0", cell) == {
        "system": "xens2", "split": "bprna_ts0", "clean_n": "80", "f1": "0.7500",
        "tp": "50", "fp": "10", "fn": "20", "removed_exact": "5",
        "removed_homology": "15", "status": "PASS"}
