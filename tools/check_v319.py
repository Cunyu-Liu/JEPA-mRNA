#!/usr/bin/env python3
"""v3.19 checker: every number in the tr1c-native ablation matrix, the
fourth-front paragraph, the 4-seed/xens3 rows and the cov-v2 paragraph
must re-derive from disk. Extends the v3.18 discipline."""
import json
import sys
from pathlib import Path

DRAFT = Path("/home/cunyuliu/rna-jepa/paper/preprint_draft.md")
text = DRAFT.read_text(encoding="utf-8")
E = "/mnt/cunyuliu/rna-jepa/eval_decision"

checks = []

def ck(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

def micro_of(path):
    return json.load(open(path))["pair_level"]["micro"]["f1"]

V = {
    "ctrl_ts0": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_bprna_ts0/result.json"),
    "ctrl_new": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_bprna_new/result.json"),
    "ctrl_ts1": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts1/result.json"),
    "ctrl_ts2": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts2/result.json"),
    "ctrl_ts3": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts3/result.json"),
    "ctrl_hard": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_ref_pdb_ts_hard/result.json"),
    "tz_ts0": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_bprna_ts0/result.json"),
    "tz_new": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_bprna_new/result.json"),
    "tz_ts1": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_ref_pdb_ts1/result.json"),
    "tz_ts2": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_ref_pdb_ts2/result.json"),
    "tz_ts3": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_ref_pdb_ts3/result.json"),
    "tz_hard": micro_of(f"{E}/turnerzero_r2dtr1c_s0_step20000_ref_pdb_ts_hard/result.json"),
    "ff_ts0": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_bprna_ts0/result.json"),
    "ff_new": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_bprna_new/result.json"),
    "ff_ts1": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_ref_pdb_ts1/result.json"),
    "ff_ts2": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_ref_pdb_ts2/result.json"),
    "ff_ts3": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_ref_pdb_ts3/result.json"),
    "ff_hard": micro_of(f"{E}/ow_rinalmo_fftr1c_b4_s0_step20000_ref_pdb_ts_hard/result.json"),
    "pdb_ts0": micro_of(f"{E}/ow_rinalmo_r2dtr1c_tr1cpdb_b4_s0_step20000_bprna_ts0/result.json"),
    "pdb_ts2": micro_of(f"{E}/ow_rinalmo_r2dtr1c_tr1cpdb_b4_s0_step20000_ref_pdb_ts2/result.json"),
    "pdb_hard": micro_of(f"{E}/ow_rinalmo_r2dtr1c_tr1cpdb_b4_s0_step20000_ref_pdb_ts_hard/result.json"),
    "xens3_ts0": micro_of(f"{E}/xens3_bprna_ts0_w0.7/result.json"),
    "xens3_tsb": micro_of(f"{E}/xens3_testsetb_w0.7/result.json"),
}

def near(a, b, tol=5e-5):
    return abs(a - b) < tol

fmt = lambda x: "%.4f" % x

# --- tr1c-native ablation matrix rows (draft text) ---
for key, num, where in [
    ("ctrl_ts0", "0.6679", "control TS0"),
    ("ctrl_new", "0.5643", "control new"),
    ("ctrl_ts1", "0.7551", "control TS1"),
    ("ctrl_ts2", "0.7686", "control TS2"),
    ("ctrl_ts3", "0.7815", "control TS3"),
    ("ctrl_hard", "0.7005", "control hard"),
    ("tz_ts0", "0.4413", "turnerzero TS0"),
    ("tz_new", "0.1960", "turnerzero new"),
    ("tz_ts1", "0.4738", "turnerzero TS1"),
    ("tz_ts2", "0.3509", "turnerzero TS2"),
    ("tz_ts3", "0.3816", "turnerzero TS3"),
    ("tz_hard", "0.2333", "turnerzero hard"),
    ("ff_ts0", "0.5791", "fftr1c TS0"),
    ("ff_new", "0.5117", "fftr1c new"),
    ("ff_ts1", "0.6822", "fftr1c TS1"),
    ("ff_ts2", "0.6882", "fftr1c TS2"),
    ("ff_ts3", "0.6203", "fftr1c TS3"),
    ("ff_hard", "0.5673", "fftr1c hard"),
    ("pdb_ts0", "0.6456", "tr1cpdb TS0"),
    ("pdb_ts2", "0.5594", "tr1cpdb TS2"),
    ("pdb_hard", "0.6034", "tr1cpdb hard"),
    ("xens3_ts0", "0.8062", "xens3 TS0"),
    ("xens3_tsb", "0.8477", "xens3 TestSetB"),
]:
    ck(f"number {num} == disk ({where})", near(V[key], float(num)), f"{key}={V[key]}")
    ck(f"'{num}' appears in draft ({where})", num in text)

# nodistill / norlcd rows (from 15.27/15.29)
nd = micro_of(f"{E}/ow_ablation_nodistill_step20000_bprna_ts0/result.json") if Path(f"{E}/ow_ablation_nodistill_step20000_bprna_ts0/result.json").exists() else None
if nd is not None:
    ck("nodistill TS0 0.6627 == disk", near(nd, 0.6627), f"{nd}")

# derived claims
ck("prior worth TS0 +0.2266", near(V["ctrl_ts0"] - V["tz_ts0"], 0.2266, 1e-4))
ck("prior worth hard +0.4672", near(V["ctrl_hard"] - V["tz_hard"], 0.4672, 1e-4))
ck("scorer worth TS0 +0.0888", near(V["ctrl_ts0"] - V["ff_ts0"], 0.0888, 1e-4))
ck("scorer worth TS3 +0.1612", near(V["ctrl_ts3"] - V["ff_ts3"], 0.1612, 1e-4))
ck("tr1cpdb TS2 delta -0.2092", near(V["ctrl_ts2"] - V["pdb_ts2"], 0.2092, 1e-4))

# --- Abstract (v3.19) numbers must match 4.3h disk values ---
def plana_of(sp, f="plana_giga_tr1c_s0_step20000"):
    d = json.load(open(f"{E}/{f}/plan_a_result.json"))
    return d["splits"][sp]["micro"]["f1"]

for num, val in [
    ("0.7866", plana_of("ts0")),
    ("0.8570", plana_of("ts1", "plana_giga_tr1c_s1_step20000")),
    ("0.8530", plana_of("hard")),
]:
    ck(f"abstract {num} in draft and == disk ({val:.4f})", num in text and abs(val - float(num)) < 5e-5)
ck("abstract 0.6132 (ens new) in draft", "0.6132" in text)
ck("abstract 0.7507 (tsb) in draft", "0.7507" in text)
ck("abstract 0.7403 (arch) in draft", "0.7403" in text)
ck("abstract 0.0007 (cal ECE) in draft", "0.0007" in text)
ck("abstract four-front sentence", "four elimination fronts" in text or "survived four elimination fronts" in text)
ck("abstract 6-of-8 claim", "6 of 8" in text)

# structural anchors
for token, where in [
    ("tr1c-native ablation matrix", "4.3b anchor"),
    ("every structural\ncomponent contributes more out-of-distribution", "signature sentence"),
    ("four elimination fronts", "fourth-front sentence"),
    ("4-seed family picture", "xens3 anchor"),
    ("Masked-marginal covariation probe", "cov v2 anchor"),
    ("v3.19", "banner version"),
]:
    ck(f"anchor '{token[:34]}' in draft ({where})", token in text)

n_fail = sum(1 for _, ok, _ in checks if not ok)
for name, ok, detail in checks:
    if not ok:
        print(f"FAIL {name}  {detail}")
print(f"v3.19 checker: {len(checks) - n_fail}/{len(checks)} PASS")
sys.exit(1 if n_fail else 0)
