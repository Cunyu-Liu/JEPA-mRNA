#!/usr/bin/env python
"""Independent four-arm recheck (CPU-only, read-only).

Recomputes every arm metric directly from the raw exported Plan-A score
matrices (inside_outside + MEA + independent raw metrics), WITHOUT reading
the official aggregate numbers, then cross-checks the recomputation against
the official evaluation_v3/pilot_verdict.json. This is an independence layer
on top of verify_raw_score_matrix.py: it confirms the official aggregates are
reproducible from raw scores and that the per-arm direction gates and the
overall pilot decision agree. It never promotes, never changes thresholds.
"""
import sys, json, math
from pathlib import Path
import numpy as np
REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO/"tools")); sys.path.insert(0, str(REPO/"src"))
import verify_raw_score_matrix as V

STAGE = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/raw_score_mechanism_4arm_v1/seed0")
ARMS = ["full_control", "frozen", "limited4", "marginal"]
KEYS = ["micro_pair_f1","macro_pair_f1","micro_auprc","micro_pair_precision",
        "macro_pair_precision","micro_stem_f1","macro_stem_f1"]

def recompute():
    contract = V.load_raw_score_contract(REPO/"configs/raw_score_mechanism_4arm_v3.json")
    csha = V.sha256_file(REPO/"configs/raw_score_mechanism_4arm_v3.json")
    records = V.load_records(contract["training"]["evaluation_data"])
    out = {}
    for arm in ARMS:
        ed = V.export_dir(contract, arm, 0, STAGE)
        meta = json.loads((ed/"export_metadata.json").read_text())
        ckpt = V.checkpoint_path(contract, arm, 0, STAGE, 9600)
        cks = V.sha256_file(ckpt)
        assert meta["checkpoint"]["sha256"] == cks, (arm, "ckpt meta mismatch")
        rows=[]; sp=[]; lp=[]
        for sh, seq, truth in records:
            with np.load(ed/"bprna_vl0"/(sh+".npz"), allow_pickle=False) as d:
                assert str(d["seq"])==seq and str(d["sequence_sha256"])==sh
                assert d["checkpoint_sha256"].item()==cks, (arm, sh)
                scores = np.asarray(d["plana_off"], dtype=np.float64)
                nt = {tuple(sorted(map(int,x))) for x in d["gt_pairs"].tolist()}
            assert scores.shape==(len(seq),len(seq)) and nt==truth
            mask = V.valid_pair_mask(seq)
            assert np.isfinite(scores[mask]).all()
            lz, post = V.inside_outside(scores, mask)
            assert math.isfinite(lz) and np.isfinite(post).all()
            pred = set(V.decode_mea(post, mask, 1.0))
            row = {"arm":arm,"sequence_sha256":sh,"n_truth_pairs":len(truth),"n_predicted_pairs":len(pred)}
            row.update(V.pair_and_stem_counts(pred, truth))
            row["raw"] = V.independent_raw_metrics(scores, mask, truth)
            rows.append(row)
            rr,cc = np.where(np.triu(mask,k=1)); o=np.lexsort((cc,rr)); rr,cc=rr[o],cc[o]
            sp.append(scores[rr,cc])
            lp.append(np.fromiter(((int(i),int(j)) in truth for i,j in zip(rr,cc)),dtype=bool,count=len(rr)))
        out[arm] = V.aggregate(rows, sp, lp, cks)
        print("[recheck] %s microF1=%.6f macroF1=%.6f AUPRC=%.6f r@P0.8=%s"%(
            arm, out[arm]["micro_pair_f1"], out[arm]["macro_pair_f1"],
            out[arm]["micro_auprc"], str(out[arm]["micro_recall_at_precision_0.8"])), flush=True)
    return contract, csha, len(records), out

def main():
    contract, csha, nseq, mine = recompute()
    rules = contract["pilot"]["per_arm_direction_gate"]
    my_comp = {}
    for arm in ("frozen","limited4","marginal"):
        d,c,p = V.direction_gate(mine[arm], mine["full_control"], rules)
        my_comp[arm] = {"delta_vs_full_control":d,"checks":c,"direction_passed":p}
    my_decision = "PILOT_DIRECTION_CONFIRMED" if any(x["direction_passed"] for x in my_comp.values()) else "STOP_RAW_SCORE_MATRIX_NO_DIRECTION"
    official = json.loads((STAGE/"evaluation_v3"/"pilot_verdict.json").read_text())
    off_arms = official["arms"]; off_comp = official["comparisons"]
    max_diff = {}; gate_agree = {}
    for arm in ARMS:
        diffs = {k: abs(float(mine[arm][k]) - float(off_arms[arm][k])) for k in KEYS}
        for k in ("micro_recall_at_precision_0.8",):
            a,b = mine[arm][k], off_arms[arm][k]
            diffs[k] = 0.0 if (a is None and b is None) else (abs(float(a)-float(b)) if (a is not None and b is not None) else 1.0)
        max_diff[arm] = diffs
    for arm in ("frozen","limited4","marginal"):
        gate_agree[arm] = bool(my_comp[arm]["direction_passed"]) == bool(off_comp[arm]["direction_passed"])
    metrics_ok = all(v <= 1e-6 for arm in max_diff for v in max_diff[arm].values())
    gates_ok = all(gate_agree.values())
    decision_ok = my_decision == official["decision"]
    overall = "PASS" if (metrics_ok and gates_ok and decision_ok) else "FAIL"
    result = {"scope":"INDEPENDENT_FOURARM_RECHECK_FROM_RAW_SCORES",
        "contract_sha256":csha,"seed":0,"n_sequences":nseq,
        "recomputed_arms":mine,"recomputed_comparisons":my_comp,"recomputed_decision":my_decision,
        "official_decision":official["decision"],"official_promoted_arms":official.get("promoted_arms"),
        "max_abs_diff_vs_official":max_diff,"direction_gate_agreement":gate_agree,
        "metrics_match_1e-6":metrics_ok,"gates_match":gates_ok,"decision_match":decision_ok,
        "overall":overall}
    (STAGE/"final_independent_check_4arm.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("=== INDEPENDENT RECHECK %s ==="%overall)
    print("metrics_match=%s gates_match=%s decision_match=%s"%(metrics_ok,gates_ok,decision_ok))
    print("my_decision=%s official_decision=%s"%(my_decision,official["decision"]))
    print(json.dumps(my_comp,indent=2))
    return 0 if overall=="PASS" else 1

if __name__=="__main__":
    sys.exit(main())
