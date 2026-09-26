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
ck("v3.8: test-family count tracks current family (22)", "22-test family" in text and "18-test family" not in text and "20-test family" not in text)
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

V["rnafm_ts0"] = micro_of(f"{E}/ow_rnafm_ff_b4_s0_step20000_bprna_ts0/result.json")
V["rnafm_new"] = micro_of(f"{E}/ow_rnafm_ff_b4_s0_step20000_bprna_new/result.json")
sd6 = json.load(open("/mnt/cunyuliu/rna-jepa/tables/stats_definitive.json"))
bb = sd6.get("backbone", {})

ck("v3.10: rnafm ts0 0.4199", f"{V['rnafm_ts0']:.4f}" == "0.4199" and "0.4199" in text, f"{V['rnafm_ts0']:.4f}")
ck("v3.10: rnafm new 0.3005", f"{V['rnafm_new']:.4f}" == "0.3005" and "0.3005" in text, f"{V['rnafm_new']:.4f}")
ck("v3.10: backbone ts0 delta -0.176", f"{V['rnafm_ts0'] - V['ff_s0_ts0']:+.4f}" == "-0.1758" and "−0.176" in text, f"{V['rnafm_ts0'] - V['ff_s0_ts0']:+.5f}")
ck("v3.10: backbone new delta -0.187", f"{V['rnafm_new'] - V['ff_s0_new']:+.4f}" == "-0.1865" and "−0.187" in text, f"{V['rnafm_new'] - V['ff_s0_new']:+.5f}")
ck("v3.10: ratios 70.5%/61.7% stated", "70.5%" in text and "61.7%" in text and f"{V['rnafm_ts0']/V['ff_s0_ts0']*100:.1f}" == "70.5" and f"{V['rnafm_new']/V['ff_s0_new']*100:.1f}" == "61.7")
ck("v3.10: 22-test family stated", "22-test family" in text and "20-test family" not in text)
ck("v3.10: stats json backbone block matches", bb.get("ts0_micro") == round(V["rnafm_ts0"], 4) and bb.get("new_micro") == round(V["rnafm_new"], 4) and bb.get("ts0_ratio_pct") == 70.5 and bb.get("new_ratio_pct") == 61.7, f"{bb}")
ck("v3.10: drift cells match 22-family re-run (7.2e-146, 1.5e-198)", "7.2e-146" in text and "1.5e-198" in text and "6.0e-146" not in text and "1.3e-198" not in text and "6.4e-146" not in text)
ck("v3.10: backbone negative framing present", "backbone-dimension axis as a negative result" in text)
ck("v3.10: appendix backbone artifact row", "ow_rnafm_ff_b4_s0_step20000_{bprna_ts0,bprna_new}" in text)

V["r2d_ts0"] = micro_of(f"{E}/ow_rinalmo_r2d_b4_s0_step20000_bprna_ts0/result.json")
V["r2d_new"] = micro_of(f"{E}/ow_rinalmo_r2d_b4_s0_step20000_bprna_new/result.json")
sc = sd.get("scorer", {})

ck("v3.12: r2d ts0 0.6629", f"{V['r2d_ts0']:.4f}" == "0.6629" and "0.6629" in text, f"{V['r2d_ts0']:.4f}")
ck("v3.12: r2d new 0.5010", f"{V['r2d_new']:.4f}" == "0.5010" and "0.5010" in text, f"{V['r2d_new']:.4f}")
ck("v3.12: scorer ts0 delta +0.067", f"{V['r2d_ts0'] - V['ff_s0_ts0']:+.4f}" == "+0.0673" and ("+0.067" in text or "0.0673" in text), f"{V['r2d_ts0'] - V['ff_s0_ts0']:+.5f}")
ck("v3.12: scorer new delta +0.014", f"{V['r2d_new'] - V['ff_s0_new']:+.4f}" == "+0.0141" and ("+0.014" in text or "0.0141" in text), f"{V['r2d_new'] - V['ff_s0_new']:+.5f}")
ck("v3.12: per-seq splits +0.047/-0.023 stated", "+0.047" in text and "−0.023" in text)
ck("v3.12: Holm cells 1.6e-22 / 4.8e-09 stated", "1.6e-22" in text and "4.8e-09" in text)
ck("v3.12: 22-test family stated", "22-test family" in text and "20-test family" not in text and "18-test family" not in text)
ck("v3.12: stats json scorer block matches", sc.get("ts0_micro") == round(V["r2d_ts0"], 4) and sc.get("new_micro") == round(V["r2d_new"], 4) and sc.get("ts0_vs_ff") == 0.0673 and sc.get("new_vs_ff") == 0.0141, f"{sc}")
ck("v3.12: re-drifted Holm cells updated (7.2e-146, 1.5e-198, 1.5e-115, 1.1e-81, 1.1e-09, 1.5e-13, 1.3e-44, 6.0e-61, 3.6e-11)", "7.2e-146" in text and "1.5e-198" in text and "1.5e-115" in text and "1.1e-81" in text and "6.0e-61" in text and "3.6e-11" in text)
ck("v3.12: stale Holm values purged", "6.4e-146" not in text and "1.4e-198" not in text and "1.3e-115" not in text and "9.9e-82" not in text and "5.2e-61" not in text and "3.3e-11" not in text)
ck("v3.12: aggregation divergence point 3 present", "joins the aggregation-divergence family" in text)
ck("v3.12: 4.3g (d) block present", "Plan-B scorer arm (2D-context scorer on the frozen backbone" in text and "architecture half of (b)'s residual" in text)
ck("v3.12: appendix scorer artifact row", "ow_rinalmo_r2d_b4_s0_step20000_{bprna_ts0,bprna_new}" in text)
ck("v3.12: structRFM near-equality stated", "0.6638 vs 0.6629" in text)
ck("v3.12: version banner v3.12", "v3.12, 2026-09-26" in text and "v3.11 change note" in text)
ck("v3.12: ledger range extends to 14.74", "§14.1–§14.74" in text)

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
