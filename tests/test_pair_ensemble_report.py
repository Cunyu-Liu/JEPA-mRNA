import pandas as pd
from tools.make_pair_ensemble_report import analyze, ATOMIC, METRICS


def row(model, dataset, recall):
    out = {"dataset_id": dataset, "score_variant": f"{model}_on"}
    for metric in METRICS:
        out[metric] = recall
    return out


def test_ensemble_recall_complement_is_descriptive_per_prior():
    models = (*ATOMIC, "plana_mean", "xens2")
    recalls = {"plana0": .50, "plana1": .60, "r2d": .55, "plana_mean": .525, "xens2": .45}
    vl0 = pd.DataFrame([row(m, "bprna_vl0", recalls[m]) for m in models])
    ext = pd.DataFrame([row(m, f"s{i}", recalls[m]) for i in range(8) for m in models])
    details, summary = analyze(vl0, ext)
    assert len(details) == 9
    assert details.micro_recall_below_all_atomic.all()
    assert summary["n_external_micro_recall_below_all_atomic"] == 8
    assert summary["by_prior"]["on"]["n_micro_recall_below_best_atomic"] == 8
    r = details.query("dataset_role == 'VL0'").iloc[0]
    assert abs(r["xens2_delta_map_micro_recall_vs_best_atomic"] - (-.15)) < 1e-12
    assert abs(r["xens2_delta_map_micro_recall_vs_plana_mean"] - (-.075)) < 1e-12
