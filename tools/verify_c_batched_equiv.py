#!/bin/env python3
"""T-A52 verification: c_batched decode in evaluate_decision.py vs numpy exact.

Three check families, each on the SAME checkpoint + SAME split (bprna_ts0):
  1. micro pair metrics (tp/fp/fn) bit-identical + P/R/F1 to 15dp
  2. macro pair metrics (f1/inf) identical to 12dp
  3. per-sequence pred_pairs lists IDENTICAL (1288/1288, not just aggregate)

Run twice: (a) same-GPU (both GPU0), (b) cross-GPU (c_batched on GPU3 vs exact GPU0).
"""
import json, sys

A = sys.argv[1]  # c_batched result.json
B = sys.argv[2]  # exact (numpy) result.json

da = json.load(open(A))
db = json.load(open(B))

checks = []

ma, mb = da["pair_level"]["micro"], db["pair_level"]["micro"]
for k in ("tp", "fp", "fn"):
    checks.append((f"micro.{k}", ma[k], mb[k], ma[k] == mb[k]))
for k in ("precision", "recall", "f1"):
    checks.append((f"micro.{k} (1e-15)", round(ma[k], 15), round(mb[k], 15),
                   abs(ma[k] - mb[k]) < 1e-15))

fa, fb = da["pair_level"]["macro"], db["pair_level"]["macro"]
for k in ("f1", "inf"):
    checks.append((f"macro.{k} (1e-12)", round(fa[k], 12), round(fb[k], 12),
                   abs(fa[k] - fb[k]) < 1e-12))

pa, pb = da["per_sequence"], db["per_sequence"]
assert len(pa) == len(pb) == 1288, (len(pa), len(pb))
n_diff = 0
first_diff = None
for i, (ra, rb) in enumerate(zip(pa, pb)):
    if ra["name"] != rb["name"] or ra["length"] != rb["length"]:
        n_diff += 1
        first_diff = first_diff or (i, "name/length mismatch")
        continue
    if ra["pred_pairs"] != rb["pred_pairs"]:
        n_diff += 1
        first_diff = first_diff or (i, ra["name"], ra["length"],
                                    "pred_pairs differ",
                                    ra["pred_pairs"][:5], rb["pred_pairs"][:5])
checks.append(("per_seq pred_pairs 1288/1288 identical", 1288 - n_diff, 1288,
               n_diff == 0))

for k in ("n_illegal", "n_hairpin_violations"):
    checks.append((f"legality.{k}", da["legality"][k], db["legality"][k],
                   da["legality"][k] == db["legality"][k]))

print(f"== {da['tag']} vs {db['tag']} ==")
print(f"{'check':<40} {'c_batched':<20} {'exact(numpy)':<20} pass")
n_pass = 0
for name, va, vb, ok in checks:
    print(f"{name:<40} {str(va):<20} {str(vb):<20} {'PASS' if ok else 'FAIL'}")
    n_pass += ok
if first_diff:
    print("first difference:", first_diff)
print(f"== {n_pass}/{len(checks)} checks PASS ==\n")
sys.exit(0 if n_pass == len(checks) else 1)
