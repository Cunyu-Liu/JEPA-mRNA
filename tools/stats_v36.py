#!/usr/bin/env python
"""v3.6 refresh: add new seeds (big_s3, bigsum, bigtr1_s1, ff_tr1_s1) to the
definitive statistics, recompute all aggregates exactly."""
import json
import statistics as st
from pathlib import Path

ART = Path("/mnt/cunyuliu/rna-jepa")


def micro(prefix, arm, step, split):
    return json.load(open(ART / "eval_decision" / f"{prefix}_rinalmo_{arm}_step{step}_{split}" / "result.json"))["pair_level"]["micro"]["f1"]


FF = [json.load(open(ART / "eval_decision/ff20000_bprna_ts0/result.json"))["pair_level"]["micro"]["f1"]] + [micro("arms", "ff_b4_s1", 20000, "ts0"), micro("arms", "ff_b4_s2", 20000, "ts0")] + [micro("ow", f"ff_b4_s{s}", 20000, "bprna_ts0") for s in [4, 5, 3, 6, 7]]
# order s0..s7: s0, s1, s2, s3, s4, s5, s6, s7
FF = [FF[0], FF[1], FF[2]] + [micro("ow", "ff_b4_s3", 20000, "bprna_ts0"), micro("ow", "ff_b4_s4", 20000, "bprna_ts0"), micro("ow", "ff_b4_s5", 20000, "bprna_ts0"), micro("ow", "ff_b4_s6", 20000, "bprna_ts0"), micro("ow", "ff_b4_s7", 20000, "bprna_ts0")]

BIG = [micro("arms", "big_b4_s0", 20000, "ts0")] + [micro("ow", f"big_b4_s{s}", 20000, "bprna_ts0") for s in [1, 2, 3]]
BIGSUM = micro("ow", "bigsum_b4_s0", 20000, "bprna_ts0")

FF_S = {0: FF[0], 1: FF[1], 2: FF[2]}
BIG_S = {0: BIG[0], 1: BIG[1], 2: BIG[2]}

print("8-seed ff:", [round(v, 4) for v in FF])
print("mean", round(st.mean(FF), 5), "std", round(st.stdev(FF), 5))
print("4-seed big:", [round(v, 4) for v in BIG])
print("big mean", round(st.mean(BIG), 5), "std", round(st.stdev(BIG), 5))
print("bigsum (sum-norm replication):", round(BIGSUM, 4))
diffs = [BIG[i] - FF[0] for i in range(1)]  # only s0 pairs directly
print("paired capacity diffs (s0-s2):", [round(BIG_S[i] - FF_S[i], 4) for i in range(3)])

TR1_NEW = [json.load(open(ART / "eval_decision/tr1_ff_step20000_bprna_new/result.json"))["pair_level"]["micro"]["f1"], micro("ow", "ff_tr1_b4_s1", 20000, "bprna_new")]
TR1_TS0 = [json.load(open(ART / "eval_decision/tr1_ff_step20000_bprna_ts0/result.json"))["pair_level"]["micro"]["f1"], micro("ow", "ff_tr1_b4_s1", 20000, "bprna_ts0")]
BIGTR1_NEW = [micro("ow", "bigtr1_b4_s0", 20000, "bprna_new"), micro("ow", "bigtr1_b4_s1", 20000, "bprna_new")]
BIGTR1_TS0 = [micro("ow", "bigtr1_b4_s0", 20000, "bprna_ts0"), micro("ow", "bigtr1_b4_s1", 20000, "bprna_ts0")]
FF_NEW = json.load(open(ART / "eval_decision/ff20000_bprna_new/result.json"))["pair_level"]["micro"]["f1"]

print("TR1 new 2-seed:", [round(v, 4) for v in TR1_NEW], "mean", round(st.mean(TR1_NEW), 4))
print("TR1 ts0 2-seed:", [round(v, 4) for v in TR1_TS0], "mean", round(st.mean(TR1_TS0), 4))
print("bigtr1 new 2-seed:", [round(v, 4) for v in BIGTR1_NEW], "mean", round(st.mean(BIGTR1_NEW), 4))
print("bigtr1 ts0 2-seed:", [round(v, 4) for v in BIGTR1_TS0], "mean", round(st.mean(BIGTR1_TS0), 4))
print("combo vs ff (new):", [round(v - FF_NEW, 4) for v in BIGTR1_NEW])
print("combo vs tr1-20k (new):", [round(v - TR1_NEW[0], 4) for v in BIGTR1_NEW], [round(v - TR1_NEW[1], 4) for v in BIGTR1_NEW])

out = {
    "ff_8seed": [round(v, 4) for v in FF],
    "ff_mean": round(st.mean(FF), 5), "ff_std": round(st.stdev(FF), 5),
    "big_4seed": [round(v, 4) for v in BIG],
    "big_mean": round(st.mean(BIG), 5), "big_std": round(st.stdev(BIG), 5),
    "bigsum_sumnorm_replication": round(BIGSUM, 4),
    "tr1_new_2seed": [round(v, 4) for v in TR1_NEW],
    "tr1_ts0_2seed": [round(v, 4) for v in TR1_TS0],
    "bigtr1_new_2seed": [round(v, 4) for v in BIGTR1_NEW],
    "bigtr1_ts0_2seed": [round(v, 4) for v in BIGTR1_TS0],
}
(ART / "tables/stats_v36.json").write_text(json.dumps(out, indent=2))
print("wrote tables/stats_v36.json")
