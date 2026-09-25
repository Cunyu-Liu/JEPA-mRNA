#!/usr/bin/env python3
"""v3.4 consistency audit: every number in the draft vs the authoritative artifacts."""
import json
import re
import sys
from pathlib import Path

DRAFT = Path("/home/cunyuliu/rna-jepa/paper/preprint_draft.md")
text = DRAFT.read_text()

checks = []


def ck(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


def micro_of(path):
    return json.load(open(path))["pair_level"]["micro"]["f1"]


E = "/mnt/cunyuliu/rna-jepa/eval_decision"

V = {
    "ff_s0_ts0": micro_of(f"{E}/ff20000_bprna_ts0/result.json"),
    "ff_s0_new": micro_of(f"{E}/ff20000_bprna_new/result.json"),
    "ff_s1": micro_of(f"{E}/arms_rinalmo_ff_b4_s1_step20000_ts0/result.json"),
    "ff_s2": micro_of(f"{E}/arms_rinalmo_ff_b4_s2_step20000_ts0/result.json"),
    "ff_s3": micro_of(f"{E}/ow_rinalmo_ff_b4_s3_step20000_bprna_ts0/result.json"),
    "ff_s4": micro_of(f"{E}/ow_rinalmo_ff_b4_s4_step20000_bprna_ts0/result.json"),
    "ff_s5": micro_of(f"{E}/ow_rinalmo_ff_b4_s5_step20000_bprna_ts0/result.json"),
    "ff_s6": micro_of(f"{E}/ow_rinalmo_ff_b4_s6_step20000_bprna_ts0/result.json"),
    "ff_s7": micro_of(f"{E}/ow_rinalmo_ff_b4_s7_step20000_bprna_ts0/result.json"),
    "big_s0": micro_of(f"{E}/arms_rinalmo_big_b4_s0_step20000_ts0/result.json"),
    "big_s1": micro_of(f"{E}/ow_rinalmo_big_b4_s1_step20000_bprna_ts0/result.json"),
    "big_s2": micro_of(f"{E}/ow_rinalmo_big_b4_s2_step20000_bprna_ts0/result.json"),
    "big_new": micro_of(f"{E}/ow_rinalmo_big_b4_s0_step20000_bprna_new/result.json"),
    "tr1_20k_new": micro_of(f"{E}/tr1_ff_step20000_bprna_new/result.json"),
    "tr1_20k_ts0": micro_of(f"{E}/tr1_ff_step20000_bprna_ts0/result.json"),
    "tr1_40k_ts0": micro_of(f"{E}/ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_ts0/result.json"),
    "tr1_40k_new": micro_of(f"{E}/ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_new/result.json"),
    "bigtr1_ts0": micro_of(f"{E}/ow_rinalmo_bigtr1_b4_s0_step20000_bprna_ts0/result.json"),
    "bigtr1_new": micro_of(f"{E}/ow_rinalmo_bigtr1_b4_s0_step20000_bprna_new/result.json"),
    "pw_s0": micro_of(f"{E}/arms_rinalmo_pw_b4_s0_step20000_ts0/result.json"),
    "pw_s1": micro_of(f"{E}/ow_rinalmo_pw_b4_s1_step20000_bprna_ts0/result.json"),
    "cascR": micro_of(f"{E}/ow_rinalmo_cascR_b4_s0_step20000_bprna_ts0/result.json"),
    "casc": micro_of(f"{E}/arch_rinalmo_casc_b4_s0_step2000_ts0/result.json"),
}

import statistics as st
seed8 = [V["ff_s0_ts0"]] + [V[f"ff_s{i}"] for i in range(1, 8)]
mean8 = st.mean(seed8)
std8 = st.stdev(seed8)
big3 = [V["big_s0"], V["big_s1"], V["big_s2"]]
capdiffs = [V["big_s0"] - V["ff_s0_ts0"], V["big_s1"] - V["ff_s1"], V["big_s2"] - V["ff_s2"]]

d = json.load(open("/mnt/cunyuliu/rna-jepa/tables/stats_definitive.json"))

ck("8-seed mean 0.5937 in draft", f"{mean8:.4f}" == "0.5937", f"{mean8:.5f}")
ck("8-seed std 0.0033", f"{std8:.4f}" == "0.0033", f"{std8:.5f}")
ck("capacity paired +0.0394", f"{st.mean(capdiffs):+.4f}" == "+0.0394", f"{st.mean(capdiffs):+.5f}")
ck("cap std 0.0088", f"{st.stdev(capdiffs):.4f}" == "0.0088", f"{st.stdev(capdiffs):.4f}")
ck("ff s0 new = 0.4870 (draft)", f"{V['ff_s0_new']:.4f}" == "0.4870" and "0.4870" in text, f"{V['ff_s0_new']:.4f}")
ck("big new 0.4641", f"{V['big_new']:.4f}" == "0.4641")
ck("bigtr1 new 0.4558", f"{V['bigtr1_new']:.4f}" == "0.4558")
ck("tr1 20k new 0.5162", f"{V['tr1_20k_new']:.4f}" == "0.5162")
ck("tr1 40k new 0.4999", f"{V['tr1_40k_new']:.4f}" == "0.4999")
ck("no stale 0.3536 as baseline (only audit notes)", text.count("0.3536") <= 2, f"count={text.count('0.3536')}")
ck("no '+0.163' anywhere", "+0.163" not in text)
ck("no '+0.111' anywhere", "+0.111" not in text)
ck("no stale mean 0.5950 outside audit note", text.count("0.5950") <= 1, f"count={text.count('0.5950')}")
ck("capacity new negative stated", "−0.023" in text or "-0.023" in text)
ck("data gain +0.029 stated", "+0.029" in text)
ck("wilcoxon section 4.3c present", "4.3c" in text and "Wilcoxon" in text)
ck("holm correction mentioned", "Holm" in text)
ck("8-seed range 0.5893-0.5990", "0.5893" in text and "0.5990" in text)
ck("big 3-seed values listed", all(f"{v:.4f}" in text for v in big3))
ck("combo p 4e-147 present", "4e-147" in text)
ck("cap s1 per-seq -0.021 present", "−0.021" in text or "-0.021" in text)
ck("conclusion: interaction negative", "interaction is negative" in text)
ck("deficit 0.19 stated", "0.19" in text)
ck("capacity vs baseline combo -0.031", "−0.031" in text or "-0.031" in text)

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
