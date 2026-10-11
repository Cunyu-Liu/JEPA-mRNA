#!/usr/bin/env python
"""S15.47 verification: xens2 2-seed (board protocol) 3-run stability.

Checks (>=3 discipline), TS0 full 1288:
  1. micro tp/fp/fn bit-identical across r1/r2/r3
  2. micro F1/P/R identical to 15dp; macro identical to 12dp
  3. per-sequence pred_pairs identical (1288/1288, all pairs)
  4. board-protocol reproduction: F1 == xens2_bprna_ts0_w0.7 board value
"""
import json
import sys

BASE = "/mnt/cunyuliu/rna-jepa/eval_decision/xens2_2seed_ts0_stability_r%d/result.json"
BOARD = "/mnt/cunyuliu/rna-jepa/eval_decision/xens2_bprna_ts0_w0.7/result.json"

runs = [json.load(open(BASE % r)) for r in (1, 2, 3)]
board = json.load(open(BOARD))

checks = []

m1, m2, m3 = [r["pair_level"]["micro"] for r in runs]
for k in ("tp", "fp", "fn"):
    checks.append(("micro.%s r1==r2==r3" % k, (m1[k], m2[k], m3[k]),
                   m1[k] == m2[k] == m3[k]))
for k in ("f1", "precision", "recall"):
    same = abs(m1[k] - m2[k]) < 1e-15 and abs(m2[k] - m3[k]) < 1e-15
    checks.append(("micro.%s (1e-15) r1==r2==r3" % k, round(m1[k], 15), same))
spread = max(m1["f1"], m2["f1"], m3["f1"]) - min(m1["f1"], m2["f1"], m3["f1"])
checks.append(("F1 spread r1..r3 == 0", spread, spread == 0.0))

fa = [r["pair_level"]["macro"]["f1"] for r in runs]
checks.append(("macro.f1 (1e-12) r1==r2==r3", round(fa[0], 12),
               max(fa) - min(fa) < 1e-12))

ps = [r["per_sequence"] for r in runs]
n = len(ps[0])
assert n == len(ps[1]) == len(ps[2]) == 1288, (n, len(ps[1]), len(ps[2]))
n_diff = 0
first = None
for i in range(n):
    a, b, c = ps[0][i], ps[1][i], ps[2][i]
    if a["name"] != b["name"] or a["name"] != c["name"]:
        n_diff += 1
        first = first or (i, "name mismatch")
        continue
    if a["pred_pairs"] != b["pred_pairs"] or a["pred_pairs"] != c["pred_pairs"]:
        n_diff += 1
        first = first or (i, a["name"], "pred_pairs differ",
                          a["pred_pairs"][:5], b["pred_pairs"][:5], c["pred_pairs"][:5])
checks.append(("per_seq pred_pairs 1288x3 identical", 1288 - n_diff, n_diff == 0))

mb = board["pair_level"]["micro"]
checks.append(("board reproduce: F1 == xens2_bprna_ts0_w0.7 (1e-15)",
               (round(m1["f1"], 15), round(mb["f1"], 15)),
               abs(m1["f1"] - mb["f1"]) < 1e-15))
for k in ("tp", "fp", "fn"):
    checks.append(("board reproduce: %s == board" % k, (m1[k], mb[k]),
                   m1[k] == mb[k]))

prot = []
for r in runs:
    prot.append((r["plana_ckpt"].split("/")[-1], r["plana_ckpt2"].split("/")[-1],
                 r["r2d_ckpt"].split("/")[-1], r["w_plana"], r["plana_s1_weight"],
                 r["n_sequences"], r["n_skipped"]))
checks.append(("protocol identical across runs (ckpt/w/s1w/n)", prot[0],
               len(set(prot)) == 1))

n_pass = 0
for name, val, ok in checks:
    print("%-52s %-30s %s" % (name, str(val)[:30], "PASS" if ok else "FAIL"))
    n_pass += ok
if first:
    print("first difference:", first)
print("== %d/%d checks PASS ==" % (n_pass, len(checks)))
sys.exit(0 if n_pass == len(checks) else 1)
