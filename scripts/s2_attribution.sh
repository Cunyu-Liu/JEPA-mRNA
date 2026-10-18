#!/bin/bash
# After s2 8-split eval completes: build old-vs-new s2 comparison + 3-seed family
# P/R profile, then append the ledger section automatically.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
D=/root/rna_jepa_data
NEW=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000
OLDREF=$D/eval_decision_ref
OUT=$NEW/attribution.json
LOG=$NEW/auto_eval.log

# wait for ALL_EVALS_DONE
while true; do
  grep -q "ALL_EVALS_DONE" $LOG 2>/dev/null && break
  sleep 300
done
echo "[$(date '+%T')] attribution: evals done, building comparison" >> $LOG

$PY - << 'PYEOF' >> $LOG 2>&1
import json, os
E = "/root/rna_jepa_data"
NEW = f"{E}/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000"
OLD = f"{E}/eval_decision_ref/ow_rinalmo_r2dtr1c_b4_s2_step20000"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard", "archiveii_embok_clean", "testsetb"]
rows = {}
for sp in SPLITS:
    nf, of = f"{NEW}/{sp}/result.json", f"{OLD}/{sp}/result.json"
    r = {"split": sp}
    if os.path.exists(nf):
        m = json.load(open(nf))["pair_level"]["micro"]
        r["new_f1"], r["new_p"], r["new_r"] = round(m["f1"], 4), round(m["precision"], 4), round(m["recall"], 4)
    if os.path.exists(of):
        m = json.load(open(of))["pair_level"]["micro"]
        r["old_f1"], r["old_p"], r["old_r"] = round(m["f1"], 4), round(m["precision"], 4), round(m["recall"], 4)
    if "new_f1" in r and "old_f1" in r:
        r["delta"] = round(r["new_f1"] - r["old_f1"], 4)
    rows[sp] = r

comparable = [r for r in rows.values() if "delta" in r]
wins = sum(1 for r in comparable if r["delta"] > 0)
verdict = ("train-completeness: new s2 >= old s2 on %d/%d comparable splits" % (wins, len(comparable))
           if wins == len(comparable) else
           "mixed: %d/%d wins" % (wins, len(comparable)))
out = {"tag": "s2_attribution", "rows": rows, "wins": wins,
       "n_comparable": len(comparable), "verdict": verdict}
json.dump(out, open(f"{NEW}/attribution.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
PYEOF
echo "[$(date '+%T')] attribution: DONE (attribution.json written)" >> $LOG
