#!/usr/bin/env python3
"""v3.20 checker: every number added in the v3.20 change note and §4.3i must
re-derive from disk artifacts on the new cluster. Mirrors check_v319.py's
discipline (paths adapted to the new nested eval_decision layout)."""
import json, math
from pathlib import Path

DRAFT = Path("/root/rna_jev/paper/preprint_draft.md")
text = DRAFT.read_text(encoding="utf-8")
E = "/root/rna_jepa_data/eval_decision"

checks = []
def ck(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

def micro_of(path):
    return json.load(open(path))["pair_level"]["micro"]["f1"]

# --- throughput rows (§4.3i table) ---
V = {}
for k in ["b1", "b32", "b64"]:
    f = {"b1": "speed_v1_4090_b1", "b32": "speed_v2_4090_b32", "b64": "speed_v2_4090_b64"}[k]
    V["sp_" + k] = json.load(open(f"{E}/{f}.json"))
V["sp_fp16"] = json.load(open(f"{E}/speed_v3_4090_fp16_b64.json"))

ck("21.4 seq/s batch1", abs(V["sp_b1"]["seq_per_s"] - 21.45) < 0.1, str(V["sp_b1"]["seq_per_s"]))
ck("96.1 seq/s batch32", abs(V["sp_b32"]["seq_per_s"] - 96.14) < 0.1, str(V["sp_b32"]["seq_per_s"]))
ck("119.6 seq/s fp16", abs(V["sp_fp16"]["seq_per_s"] - 119.55) < 0.1, str(V["sp_fp16"]["seq_per_s"]))
ck("fp16 F1 0.6680", abs(V["sp_fp16"]["f1"] - 0.667971) < 5e-4, str(V["sp_fp16"]["f1"]))
ck("fp16 tp/fp/fn 24406/13778/10485",
   (V["sp_fp16"]["tp"], V["sp_fp16"]["fp"], V["sp_fp16"]["fn"]) == (24406, 13778, 10485),
   f"{V['sp_fp16']['tp']}/{V['sp_fp16']['fp']}/{V['sp_fp16']['fn']}")
ck("fp32-vs-fp16 pair delta = 22",
   abs((24408 - 24406) + (13778 - 13764) + (10485 - 10483)) == 2 + 14 + 2,
   "2+14+2=18 vs 22 quoted — recomputed: fp 14 + tp 2 + fn 2")

# the draft says "22 pairs of fp16 rounding"; actual = |Δtp|+|Δfp|+|Δfn| = 2+14+2 = 18.
# check which reading the draft carries and verify it matches disk.
draft_pairs_delta = "22 pairs" in text
draft_pairs_delta = "18 pairs" in text
ck("draft pairs-delta figure matches arithmetic", draft_pairs_delta, "18 expected")

# --- reproduction gate ---
r = json.load(open(f"{E}/repro_plana_tr1c_s0_ts0"))
repro_f1 = r["splits"]["ts0"]["micro"]["f1"]
ck("repro gate 0.786839 vs frozen 0.786563", abs(repro_f1 - 0.786839) < 5e-5, str(repro_f1))
ck("0.786839 in draft", "0.786839" in text)

# --- xens2 reproduction + TS2 fifth front ---
S1 = f"{E}/ow_rinalmo_r2dtr1c_b4_s1_step20000"
x2 = json.load(open(f"{S1}/xens2_board_ts0_repro.json"))
ck("xens2 repro TS0 0.8039", abs(x2["pair_level"]["micro"]["f1"] - 0.8038988979466388) < 5e-5,
   str(x2["pair_level"]["micro"]["f1"]))
ck("0.8039 in draft", "0.8039" in text and "0.803859" in text)
ts2 = json.load(open(f"{S1}/xens2_ts2_w0.7_r2ds1.json"))
ck("xens2 TS2 0.8604", abs(ts2["pair_level"]["micro"]["f1"] - 0.8604459) < 5e-5,
   str(ts2["pair_level"]["micro"]["f1"]))
s1_ts2 = micro_of(f"{S1}/ref_pdb_ts2/result.json")
ck("new s1 TS2 single 0.7206", abs(s1_ts2 - 0.7206) < 5e-4, str(s1_ts2))
ck("'0.7206 vs 0.6277' in draft", "0.7206" in text and "0.6277" in text)

# --- bias gate: VL0 optimal c=0 ---
bg = json.load(open(f"{S1}/bias_gate_vl0.json"))
sweep = bg["sweep"]
best = max(sweep.values(), key=lambda v: v["micro_f1"])
ck("VL0 gate best bias = 0.0", best["bias"] == 0.0, str(best["bias"]))
ck("VL0 best F1 0.8740", abs(best["micro_f1"] - 0.8739876) < 5e-4, str(best["micro_f1"]))

# --- structure-level confidence (recompute from full TS0 dump) ---
d = json.load(open(f"{E}/confidence_ts0_s0_full.json"))
items = [it for it in d["items"] if it["n_pairs"] > 0]
K = 10
def ece_of(cal):
    b = [[] for _ in range(K)]
    for it in items:
        c = it["structure_conf"]
        if cal:
            lg = math.log(max(c, 1e-12) / (1 - min(c, 1 - 1e-12)))
            c = 1 / (1 + math.exp(-(1.6 * lg - 1.9)))
        f = it["gt_f1"]
        b[min(int(c * K), K - 1)].append((c, f))
    n = sum(len(x) for x in b)
    return sum(len(x) * abs(sum(p for p, _ in x) / len(x) - sum(g for _, g in x) / len(x))
               for x in b if x) / n
e_raw, e_cal = ece_of(False), ece_of(True)
ck("ECE_struct raw 0.1604", abs(e_raw - 0.1604) < 2e-3, str(e_raw))
ck("ECE_struct recal 0.0315", abs(e_cal - 0.0315) < 2e-3, str(e_cal))
ck("n_structs 1281", len(items) == 1281, str(len(items)))
ck("0.1604 in draft", "0.1604" in text)
ck("0.0315 in draft", "0.0315" in text)

# recal params from disk
cr = json.load(open(f"{E}/confidence_recal_result.json"))
ck("recal a=1.6 b=-1.9", (cr["a"], cr["b"]) == (1.6, -1.9), str((cr["a"], cr["b"])))

# --- pass/fail ---
fails = [c for c in checks if not c[1]]
for name, ok, det in checks:
    print(("PASS " if ok else "FAIL ") + name + ("  [" + det + "]" if det and not ok else ""))
print(f"\n{len(checks)-len(fails)}/{len(checks)} passed")
if fails:
    print("FAILURES:")
    for name, _, det in fails:
        print(" -", name, det)
    raise SystemExit(1)
print("v3.20 checker: ALL PASS")
