#!/usr/bin/env python3
"""v3.18 checker: every number in the new 4.3h table + banner claims must
re-derive from disk. Extends the v3.4 discipline to the turnaround chapter."""
import json
import sys
from pathlib import Path

DRAFT = Path("/home/cunyuliu/rna-jepa/paper/preprint_draft.md")
text = DRAFT.read_text(encoding="utf-8")
E = "/mnt/cunyuliu/rna-jepa/eval_decision"
T = "/home/cunyuliu/rna-jepa/tables/sota_vs_ours.md"

checks = []

def ck(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

def micro_of(path):
    return json.load(open(path))["pair_level"]["micro"]["f1"]

def plana_of(sp, f="plana_giga_tr1c_s0_step20000"):
    d = json.load(open(f"{E}/{f}/plan_a_result.json"))
    return d["splits"][sp]["micro"]["f1"]

V = {
    "plana_s0_ts0": plana_of("ts0"),
    "plana_s0_hard": plana_of("hard"),
    "plana_s0_ts3": plana_of("ts3"),
    "plana_s1_ts1": plana_of("ts1", "plana_giga_tr1c_s1_step20000"),
    "plana_s1_ts2": plana_of("ts2", "plana_giga_tr1c_s1_step20000"),
    "r2d_s0_ts0": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_bprna_ts0/result.json"),
    "r2d_s0_new": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step8000_bprna_new/result.json"),
    "r2d_s0_arch": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_archiveii_embok_clean/result.json"),
    "r2d_s0_tsb": micro_of(f"{E}/ow_rinalmo_r2dtr1c_b4_s0_step20000_testsetb/result.json"),
    "xens2_ts0": micro_of(f"{E}/xens2_bprna_ts0_w0.7/result.json"),
    "xens2_ts1": micro_of(f"{E}/xens2_ref_pdb_ts1_w0.7/result.json"),
    "xens2_hard": micro_of(f"{E}/xens2_ref_pdb_ts_hard_w0.7/result.json"),
    "xens2_ts2": micro_of(f"{E}/xens2_ref_pdb_ts2_w0.7/result.json"),
    "xens2_ts3": micro_of(f"{E}/xens2_ref_pdb_ts3_w0.7/result.json"),
    "xens2_tsb": micro_of(f"{E}/xens2_testsetb_w0.7/result.json"),
    "xens2_arch": micro_of(f"{E}/xens2_archiveii_embok_clean_w0.7/result.json"),
    "ens2seed_new": micro_of(f"{E}/ens_r2dtr1_2seed_bprna_new/result.json"),
}

# --- 4.3h table numbers (draft text) ---
for key, num, where in [
    ("plana_s0_ts0", "0.7866", "4.3h TS0 single"),
    ("plana_s1_ts1", "0.8570", "4.3h TS1 single"),
    ("plana_s0_hard", "0.8530", "4.3h TS-hard single"),
    ("plana_s0_ts3", "0.9049", "4.3h TS3 single"),
    ("plana_s1_ts2", "0.8393", "4.3h TS2 single"),
    ("r2d_s0_tsb", "0.7507", "4.3h TestSetB single"),
    ("r2d_s0_arch", "0.7403", "4.3h ArchiveII single"),
    ("r2d_s0_new", "0.6067", "(OOD peak cell @8k)"),
    ("xens2_ts0", "0.8039", "4.3h TS0 ens"),
    ("xens2_ts1", "0.8755", "4.3h TS1 ens"),
    ("xens2_hard", "0.8732", "4.3h TS-hard ens"),
    ("xens2_ts2", "0.8588", "4.3h TS2 ens"),
    ("xens2_ts3", "0.8965", "4.3h TS3 ens"),
    ("xens2_tsb", "0.8448", "4.3h TestSetB ens"),
    ("xens2_arch", "0.7760", "4.3h ArchiveII ens"),
    ("ens2seed_new", "0.6132", "4.3h bpRNA-new ens"),
]:
    ck(f"disk {where}: {V[key]:.4f} == {num}",
       f"{V[key]:.4f}" == num, f"disk={V[key]:.4f} draft={num}")

# --- structural anchors ---
ck("4.3h section present", "### 4.3h Decontamination turned the leaderboard" in text)
ck("banner v3.18", "v3.18, 2026-10-06" in text)
ck("v3.18 change note", "v3.18 change note" in text)
ck("retraction disclosure in 4.3h", "1,087 TS0 sequences" in text and "247 of the 428" in text)
ck("TS2/TS3 negative result stated", "real model gap" in text or "real discrimination gap" in text)
ck("bpRNA-new honesty (ensemble-only)", "ensemble-only" in text)
ck("limitation 12 (retraction)", "Two of our own earlier numbers were retracted" in text)
ck("limitation 13 (bpRNA-new)", "ensemble-only pass" in text)
ck("conclusion refreshed", "decontaminated eight-benchmark board" in text)

# --- cross-check against the sota table (source of truth) ---
sota = Path(T).read_text(encoding="utf-8")
for num in ["0.7866", "0.8570", "0.8530", "0.9049", "0.8393", "0.7507", "0.7403",
            "0.8039", "0.8755", "0.8732", "0.8588", "0.8965", "0.8448", "0.7760", "0.6132"]:
    ck(f"sota table contains {num}", num in sota)

fails = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
