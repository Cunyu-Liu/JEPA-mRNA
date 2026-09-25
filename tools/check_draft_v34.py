#!/usr/bin/env python3
"""v3.6 consistency audit: every number in the draft vs the authoritative artifacts.

v3.4 base (24 checks) + v3.6 additions: capacity 4-seed, second seeds
(bigtr1 s1 / ff_tr1 s1, both splits), artifact-drift canon (s7 0.5974,
big_s2 0.6337 re-produced by watch self-heal, protocol-identical).
"""
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
    "big_s3": micro_of(f"{E}/ow_rinalmo_big_b4_s3_step20000_bprna_ts0/result.json"),
    "bigsum_ts0": micro_of(f"{E}/ow_rinalmo_bigsum_b4_s0_step20000_bprna_ts0/result.json"),
    "bigsum_new": micro_of(f"{E}/ow_rinalmo_bigsum_b4_s0_step20000_bprna_new/result.json"),
    "bigtr1_s1_ts0": micro_of(f"{E}/ow_rinalmo_bigtr1_b4_s1_step20000_bprna_ts0/result.json"),
    "bigtr1_s1_new": micro_of(f"{E}/ow_rinalmo_bigtr1_b4_s1_step20000_bprna_new/result.json"),
    "tr1_s1_20k_ts0": micro_of(f"{E}/ow_rinalmo_ff_tr1_b4_s1_step20000_bprna_ts0/result.json"),
    "tr1_s1_20k_new": micro_of(f"{E}/ow_rinalmo_ff_tr1_b4_s1_step20000_bprna_new/result.json"),
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
big4 = [V["big_s0"], V["big_s1"], V["big_s2"], V["big_s3"]]
capdiffs = [V["big_s0"] - V["ff_s0_ts0"], V["big_s1"] - V["ff_s1"], V["big_s2"] - V["ff_s2"], V["big_s3"] - V["ff_s3"]]

d = json.load(open("/mnt/cunyuliu/rna-jepa/tables/stats_definitive.json"))

ck("8-seed mean 0.5938 in draft", f"{mean8:.4f}" == "0.5938" and "0.5938" in text, f"{mean8:.5f}")
ck("8-seed std 0.0034", f"{std8:.4f}" == "0.0034", f"{std8:.5f}")
ck("capacity paired +0.0395", f"{st.mean(capdiffs):+.4f}" == "+0.0395", f"{st.mean(capdiffs):+.5f}")
ck("cap std 0.0072", f"{st.stdev(capdiffs):.4f}" == "0.0072", f"{st.stdev(capdiffs):.5f}")
ck("ff s0 new = 0.4870 (draft)", f"{V['ff_s0_new']:.4f}" == "0.4870" and "0.4870" in text, f"{V['ff_s0_new']:.4f}")
ck("big new 0.4641", f"{V['big_new']:.4f}" == "0.4641")
ck("bigtr1 new 0.4558", f"{V['bigtr1_new']:.4f}" == "0.4558")
ck("tr1 20k new 0.5162", f"{V['tr1_20k_new']:.4f}" == "0.5162")
ck("tr1 40k new 0.4999", f"{V['tr1_40k_new']:.4f}" == "0.4999")
ck("no stale 0.3536 as baseline (only audit notes)", text.count("0.3536") <= 2, f"count={text.count('0.3536')}")
ck("no '+0.163' anywhere", "+0.163" not in text)
ck("no '+0.111' anywhere", "+0.111" not in text)
ck("no stale mean 0.5950 outside audit note", text.count("0.5950") <= 1, f"count={text.count('0.5950')}")
ck("no stale mean 0.5937 outside v3.5 banner", text.count("0.5937") <= 2, f"count={text.count('0.5937')}")
ck("capacity new negative stated", "−0.023" in text or "-0.023" in text)
ck("data gain +0.029 stated", "+0.029" in text)
ck("wilcoxon section 4.3c present", "4.3c" in text and "Wilcoxon" in text)
ck("holm correction mentioned", "Holm" in text)
ck("8-seed range 0.5893-0.5990", "0.5893" in text and "0.5990" in text)
ck("big 4-seed values listed", all(f"{v:.4f}" in text for v in big4))
ck("combo p 4e-147 present", "4e-147" in text)
ck("cap s1 per-seq -0.021 present", "−0.021" in text or "-0.021" in text)
ck("conclusion: interaction negative", "interaction is negative" in text)
ck("deficit 0.19 stated", "0.19" in text)
ck("capacity vs baseline combo -0.031", "−0.031" in text or "-0.031" in text)

ck("v3.6: bigtr1 s1 ts0 0.6282", f"{V['bigtr1_s1_ts0']:.4f}" == "0.6282" and "0.6282" in text, f"{V['bigtr1_s1_ts0']:.4f}")
ck("v3.6: bigtr1 s1 new 0.4088", f"{V['bigtr1_s1_new']:.4f}" == "0.4088" and "0.4088" in text, f"{V['bigtr1_s1_new']:.4f}")
ck("v3.6: ff_tr1 s1 ts0 0.5842", f"{V['tr1_s1_20k_ts0']:.4f}" == "0.5842" and "0.5842" in text, f"{V['tr1_s1_20k_ts0']:.4f}")
ck("v3.6: ff_tr1 s1 new 0.4954", f"{V['tr1_s1_20k_new']:.4f}" == "0.4954" and "0.4954" in text, f"{V['tr1_s1_20k_new']:.4f}")
ck("v3.6: big 4-seed mean 0.6324", f"{st.mean(big4):.4f}" == "0.6324", f"{st.mean(big4):.5f}")
ck("v3.6: bigtr1 2-seed ts0 mean 0.6364", f"{st.mean([V['bigtr1_ts0'], V['bigtr1_s1_ts0']]):.4f}" == "0.6364" and "0.6364" in text)
ck("v3.6: bigtr1 2-seed new mean 0.4323", f"{st.mean([V['bigtr1_new'], V['bigtr1_s1_new']]):.4f}" == "0.4323" and "0.4323" in text)
ck("v3.6: ff_tr1 2-seed new mean 0.5058", f"{st.mean([V['tr1_20k_new'], V['tr1_s1_20k_new']]):.4f}" == "0.5058" and "0.5058" in text)
ck("v3.6: combo s1 vs tr1 s1 -0.087 stated", "-0.087" in text or "−0.087" in text or "−0.09" in text)
ck("v3.6: combo s1 new vs ff -0.078 stated", "-0.078" in text or "−0.078" in text)
ck("v3.6: drift canon s7 0.5974", f"{V['ff_s7']:.4f}" == "0.5974", f"{V['ff_s7']:.5f}")
ck("v3.6: drift canon big_s2 0.6337", f"{V['big_s2']:.4f}" == "0.6337", f"{V['big_s2']:.5f}")

ck("v3.8: bigsum ts0 0.6302", f"{V['bigsum_ts0']:.4f}" == "0.6302" and "0.6302" in text, f"{V['bigsum_ts0']:.5f}")
ck("v3.8: bigsum new 0.4789", f"{V['bigsum_new']:.4f}" == "0.4789" and "0.4789" in text, f"{V['bigsum_new']:.5f}")
ck("v3.8: bigsum new vs ff -0.008 stated", f"{V['bigsum_new'] - V['ff_s0_new']:+.4f}" == "-0.0080" and ("-0.008" in text or "−0.008" in text), f"{V['bigsum_new'] - V['ff_s0_new']:+.5f}")
ck("v3.8: bigsum new vs big +0.015 stated", f"{V['bigsum_new'] - V['big_new']:+.4f}" == "+0.0149" and ("+0.015" in text or "0.0149" in text), f"{V['bigsum_new'] - V['big_new']:+.5f}")
ck("v3.8: 18-test family stated", "18-test family" in text)
ck("v3.8: bigsum both splits in appendix", "ow_rinalmo_bigsum_b4_s0_step20000_{bprna_ts0,bprna_new}" in text)


ENS = "/mnt/cunyuliu/rna-jepa/eval_decision"
V["ens8_ts0"] = micro_of(f"{E}/ensemble8_ts0/result.json")
V["ens8_new"] = micro_of(f"{E}/ensemble8_new/result.json")
V["ens_tr1_new"] = micro_of(f"{E}/ensemble_tr1_2seed_new/result.json")
sd = json.load(open("/mnt/cunyuliu/rna-jepa/tables/stats_definitive.json"))
ens = sd.get("ensembles", {})

ck("v3.9: ens8 ts0 0.6105", f"{V['ens8_ts0']:.4f}" == "0.6105" and "0.6105" in text, f"{V['ens8_ts0']:.4f}")
ck("v3.9: ens8 new 0.5106", f"{V['ens8_new']:.4f}" == "0.5106" and "0.5106" in text, f"{V['ens8_new']:.4f}")
ck("v3.9: ens tr1 2seed new 0.5229", f"{V['ens_tr1_new']:.4f}" == "0.5229" and "0.5229" in text, f"{V['ens_tr1_new']:.4f}")
ck("v3.9: ensemble gain +0.0166 stated", "+0.0166" in text)
ck("v3.9: ensemble OOD gain +0.0236 stated", "+0.0236" in text)
ck("v3.9: ensemble section 4.3f present", "4.3f" in text and "seed ensemble" in text.lower())
ck("v3.9: stats json ensembles block matches", ens.get("ens8_ts0", {}).get("micro") == round(V["ens8_ts0"], 4) and ens.get("ens8_new", {}).get("micro") == round(V["ens8_new"], 4), f"{ens}")

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
