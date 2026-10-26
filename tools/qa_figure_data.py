#!/usr/bin/env python3
"""最终 PPT 图源数据的独立一致性检查。"""
from pathlib import Path
import json
import os
import numpy as np

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03"))
D = json.loads((ROOT / "figures" / "source_data.json").read_text(encoding="utf-8"))

m = D["main_benchmark"]
assert len({len(m[k]) for k in ["splits", "single", "xens2", "clean_reference", "full_clean_reference", "reference_names"]}) == 1
assert sum(v > r for v, r in zip(m["single"], m["clean_reference"])) == m["single_clean_wins"] == 4
assert sum(v > r for v, r in zip(m["xens2"], m["clean_reference"])) == m["xens2_clean_wins"] == 6
assert sum(v > r for v, r in zip(m["xens2"], m["full_clean_reference"])) == m["xens2_full_clean_wins"] == 5
assert np.isclose(m["single"][-1], 0.7166987385075904) and np.isclose(m["xens2"][-1], 0.8370)
assert np.isclose(m["xens2"][1], 0.5532)

s = D["speed"]
tp, fp, fn = s["tp_fp_fn"]
assert np.isclose(2*tp/(2*tp+fp+fn), s["f1"], atol=2e-6)
assert max(x["throughput"] for x in s["systems"]) == 124.78
assert s["ladder"][5]["throughput"] == 124.78

e = D["ece"]
assert all(np.less(e["recalibrated"][:4], e["raw"][:4]))
assert e["recalibrated"][4] < e["raw"][4] and e["recalibrated"][7] < e["raw"][7]
assert e["recalibrated"][5] > e["raw"][5] and e["recalibrated"][6] > e["raw"][6]
assert len(e["spearman"]) == 8 and min(e["spearman"]) > 0

models = D["accuracy_speed_a100"]["models"]
frontier = []
for i, model in enumerate(models):
    if not any(j != i and other["throughput"] >= model["throughput"] and other["f1"] >= model["f1"] and
               (other["throughput"] > model["throughput"] or other["f1"] > model["f1"])
               for j, other in enumerate(models)):
        frontier.append(model["name"])
assert set(frontier) == {"RNAformer", "UFold"}, frontier

nb = D["nb_transfer"]
delta = np.array(nb["jev_on_nb"]) - np.array(nb["nb_ft"])
assert (delta > 0).all() and np.isclose(delta.mean(), 0.134, atol=0.0006)
assert np.isclose(delta.min(), 0.055, atol=0.0006) and np.isclose(delta.max(), 0.234, atol=0.0006)

for m in D["chanrg"]["models"]:
    assert np.isclose(np.mean(m["values"][1:]), m["ood_mean"], atol=6e-5)
assert len(D["chanrg"]["models"]) == 4

for key in ["ts0_n", "archiveii_n", "buckets"]:
    assert len(D["length"][key]) == 4
for group in ["ts0", "archiveii"]:
    assert all(len(values) == 4 for values in D["length"][group].values())

cur = D["matched_30k_current"]
stats = json.loads((ROOT / "figures" / "matched_30k_statistics.json").read_text(encoding="utf-8"))
assert stats["status"] == "PASS" and stats["n_arms"] == 6 and stats["n_cells"] == 24
order = ["chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
for arm, record in stats["arms"].items():
    short = arm.replace("chanrg_clean_30k_", "clean_") if arm.startswith("chanrg_clean") else arm.replace("mixed_30k_", "mixed_")
    values = [record["metrics"][split] for split in order]
    assert np.array_equal(cur[short], values), short
    assert np.isclose(np.mean(values[1:]), record["ood_mean"], atol=1e-12), short
assert set(cur["statistics"]["verdicts"]) >= {"genf_mix_hypothesis_pass", "genf_sota_pass", "ood_mean_sota_pass", "ofoldna_ood_mean_pass"}
print("FIGURE_SOURCE_DATA_QA_PASS")
print("main_benchmark single=4/8 xens2=6/8 full_clean=5/8")
print("pareto", frontier)
print("nb_delta", delta.min(), delta.mean(), delta.max())
print("matched 6/6 arms and 24/24 cells synchronized; OOD means", {key: np.mean(cur[key][1:]) for key in ["clean_s0", "clean_s1", "clean_s2", "mixed_s0", "mixed_s1", "mixed_s2"]})
