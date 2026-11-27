import importlib.util
from pathlib import Path


def load(name):
    path = Path(__file__).parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


incremental = load("build_rnaformer_pdb_incremental_20261127.py")
builder = load("rebuild_rnaformer_pdb_labels_20261127.py")


def test_scheme_repair_excludes_only_out_of_entity_endpoints():
    chain = {
        "auth_chain": "A",
        "scheme_rows": [{"auth_seq": "1", "insertion": ".", "seq_id": 1},
                        {"auth_seq": "2", "insertion": ".", "seq_id": 2}],
        "observed_mask": [True, True],
    }
    dssr = {"pairs": [{"nt1": "A.C1", "nt2": "A.G2"},
                      {"nt1": "A.C1", "nt2": "A.GTP101"}]}
    result = incremental.filtered_pairs(builder, dssr, chain)
    assert result["full_pairs"] == [[0, 1]]
    assert result["observed_pairs"] == [[0, 1]]
    assert len(result["excluded_pair_records"]) == 1
    assert result["excluded_pair_records"][0]["reason"] == "ENDPOINT_OUTSIDE_SELECTED_POLYMER_ENTITY"


def test_unobserved_residue_cannot_be_repaired_into_a_pair():
    chain = {"auth_chain": "A",
             "scheme_rows": [{"auth_seq": "1", "insertion": ".", "seq_id": 1},
                             {"auth_seq": "2", "insertion": ".", "seq_id": 2}],
             "observed_mask": [True, False]}
    result = incremental.filtered_pairs(builder, {"pairs": [{"nt1": "A.C1", "nt2": "A.G2"}]}, chain)
    assert result["full_pairs"] == []
    assert len(result["excluded_pair_records"]) == 1
