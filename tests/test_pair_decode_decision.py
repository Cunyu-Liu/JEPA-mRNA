import pandas as pd
from tools.make_pair_decode_decision import decide


def route_row(dataset, good=True):
    delta = .03 if good else -.01
    return {"dataset_id": dataset, "score_variant": "r2d_on",
            "map_micro_f1": .5, "map_macro_f1": .48,
            "map_micro_precision": .6, "map_micro_recall": .45,
            "mea_micro_f1": .5 + delta, "mea_macro_f1": .48 + delta,
            "mea_micro_precision": .59 if good else .4, "mea_micro_recall": .46,
            "mea_delta_micro_f1_vs_map": delta}


def test_frozen_route_rule_requires_vl0_and_six_of_eight():
    dev = pd.DataFrame([route_row("bprna_vl0")])
    ext = pd.DataFrame([route_row(f"s{i}", good=i < 6) for i in range(8)])
    decision, details = decide(dev, ext)
    row = decision.iloc[0]
    assert row.n_external_both_f1_improve == 6
    assert row["n_external_micro_precision_drop_le_0.05"] == 6
    assert bool(row.decode_main_effect_evidence)
    assert len(details) == 8


def test_frozen_route_rule_rejects_small_vl0_effect_and_incomplete_panel():
    dev = pd.DataFrame([route_row("bprna_vl0")])
    dev.loc[0, "mea_delta_micro_f1_vs_map"] = .01
    ext = pd.DataFrame([route_row(f"s{i}", good=True) for i in range(8)])
    decision, _ = decide(dev, ext)
    assert not bool(decision.iloc[0].decode_main_effect_evidence)
    dev.loc[0, "mea_delta_micro_f1_vs_map"] = .03
    decision, _ = decide(dev, ext.iloc[:7])
    assert not bool(decision.iloc[0].decode_main_effect_evidence)
