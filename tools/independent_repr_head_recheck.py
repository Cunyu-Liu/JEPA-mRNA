#!/usr/bin/env python
"""Independent representation/head four-arm recheck (CPU-only, read-only).

Recomputes every arm metric directly from the raw exported Plan-A score matrices
(inside_outside + MEA + independent raw metrics) WITHOUT reading the official
aggregate numbers, then cross-checks against the official pilot_verdict.json.
An independence layer on top of verify_repr_head_matrix.py: confirms the official
aggregates are reproducible from raw scores and that the per-arm direction gates
(vs the retrained control_matched) and the overall pilot decision agree. Never
promotes, never changes thresholds. Reuses the audited RAW_SCORE primitives.
"""
import argparse
import sys
import json
import math
from pathlib import Path
import numpy as np
REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import verify_raw_score_matrix as V
from rnajepa.repr_head_matrix import (CONTROL_ARM, STOP_DECISION, CONFIRM_DECISION,
                                      active_repr_head_arms, load_repr_head_contract)

KEYS = ["micro_pair_f1", "macro_pair_f1", "micro_auprc", "micro_pair_precision",
        "macro_pair_precision", "micro_stem_f1", "macro_stem_f1"]


def recompute(contract_path, stage, seed, steps):
    contract = load_repr_head_contract(contract_path)
    csha = V.sha256_file(contract_path)
    records = V.load_records(contract["training"]["evaluation_data"])
    out = {}
    arms = active_repr_head_arms(contract, seed)
    for arm in arms:
        ed = stage / arm / "export_train_off"
        meta = json.loads((ed / "export_metadata.json").read_text())
        ckpt = stage / arm / "snapshots" / f"{arm}_step{steps}.pt"
        cks = V.sha256_file(ckpt)
        assert meta["checkpoint"]["sha256"] == cks, (arm, "ckpt meta mismatch")
        rows = []; sp = []; lp = []
        for sh, seq, truth in records:
            with np.load(ed / "bprna_vl0" / (sh + ".npz"), allow_pickle=False) as d:
                assert str(d["seq"]) == seq and str(d["sequence_sha256"]) == sh
                assert d["checkpoint_sha256"].item() == cks, (arm, sh)
                scores = np.asarray(d["plana_off"], dtype=np.float64)
                nt = {tuple(sorted(map(int, x))) for x in d["gt_pairs"].tolist()}
            assert scores.shape == (len(seq), len(seq)) and nt == truth
            mask = V.valid_pair_mask(seq)
            assert np.isfinite(scores[mask]).all()
            lz, post = V.inside_outside(scores, mask)
            assert math.isfinite(lz) and np.isfinite(post).all()
            pred = set(V.decode_mea(post, mask, 1.0))
            row = {"arm": arm, "sequence_sha256": sh, "n_truth_pairs": len(truth), "n_predicted_pairs": len(pred)}
            row.update(V.pair_and_stem_counts(pred, truth))
            row["raw"] = V.independent_raw_metrics(scores, mask, truth)
            rows.append(row)
            rr, cc = np.where(np.triu(mask, k=1)); o = np.lexsort((cc, rr)); rr, cc = rr[o], cc[o]
            sp.append(scores[rr, cc])
            lp.append(np.fromiter(((int(i), int(j)) in truth for i, j in zip(rr, cc)), dtype=bool, count=len(rr)))
        out[arm] = V.aggregate(rows, sp, lp, cks)
        print("[recheck] %s microF1=%.6f macroF1=%.6f AUPRC=%.6f r@P0.8=%s" % (
            arm, out[arm]["micro_pair_f1"], out[arm]["macro_pair_f1"],
            out[arm]["micro_auprc"], str(out[arm]["micro_recall_at_precision_0.8"])), flush=True)
    return contract, csha, len(records), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=REPO / "configs/representation_head_mechanism_4arm_v1.json")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=9600)
    args = parser.parse_args()
    contract, csha, nseq, mine = recompute(args.contract, args.stage, args.seed, args.steps)
    rules = contract["pilot"]["per_arm_direction_gate"]
    arms = active_repr_head_arms(contract, args.seed)
    variant_arms = arms[1:]
    my_comp = {}
    for arm in variant_arms:
        d, c, p = V.direction_gate(mine[arm], mine[CONTROL_ARM], rules)
        my_comp[arm] = {"delta_vs_control_matched": d, "checks": c, "direction_passed": p}
    my_decision = CONFIRM_DECISION if any(x["direction_passed"] for x in my_comp.values()) else STOP_DECISION
    official = json.loads((args.evaluation / "pilot_verdict.json").read_text())
    off_arms = official["arms"]; off_comp = official["comparisons"]
    max_diff = {}; gate_agree = {}
    for arm in arms:
        diffs = {k: abs(float(mine[arm][k]) - float(off_arms[arm][k])) for k in KEYS}
        for k in ("micro_recall_at_precision_0.8",):
            a, b = mine[arm][k], off_arms[arm][k]
            diffs[k] = 0.0 if (a is None and b is None) else (abs(float(a) - float(b)) if (a is not None and b is not None) else 1.0)
        max_diff[arm] = diffs
    for arm in variant_arms:
        gate_agree[arm] = bool(my_comp[arm]["direction_passed"]) == bool(off_comp[arm]["direction_passed"])
    metrics_ok = all(v <= 1e-6 for arm in max_diff for v in max_diff[arm].values())
    gates_ok = all(gate_agree.values())
    decision_ok = my_decision == official["decision"]
    overall = "PASS" if (metrics_ok and gates_ok and decision_ok) else "FAIL"
    result = {"scope": "INDEPENDENT_REPRESENTATION_HEAD_RECHECK_FROM_RAW_SCORES",
              "contract_sha256": csha, "seed": args.seed, "n_sequences": nseq,
              "control_arm": CONTROL_ARM,
              "recomputed_arms": mine, "recomputed_comparisons": my_comp, "recomputed_decision": my_decision,
              "official_decision": official["decision"], "official_promoted_arms": official.get("promoted_arms"),
              "max_abs_diff_vs_official": max_diff, "direction_gate_agreement": gate_agree,
              "metrics_match_1e-6": metrics_ok, "gates_match": gates_ok, "decision_match": decision_ok,
              "overall": overall}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("=== INDEPENDENT RECHECK %s ===" % overall)
    print("metrics_match=%s gates_match=%s decision_match=%s" % (metrics_ok, gates_ok, decision_ok))
    print("my_decision=%s official_decision=%s" % (my_decision, official["decision"]))
    print(json.dumps(my_comp, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())