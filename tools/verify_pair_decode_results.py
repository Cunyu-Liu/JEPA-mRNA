#!/usr/bin/env python3
"""Independent verification of raw/decode diagnostics.

The brute-force reference is implemented here independently; the production
implementation is imported only as the system under comparison.
"""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))


def all_structures(seq):
    n = len(seq)
    canonical = {"AU", "UA", "GC", "CG", "GU", "UG"}
    # De-duplicating exhaustive recursion; no production partition or traceback.
    structures = set()
    def enumerate_from(i, chosen):
        if i >= n:
            structures.add(frozenset(chosen))
            return
        enumerate_from(i + 1, chosen)
        for j in range(i + 4, n):
            if seq[i] + seq[j] in canonical and all(i not in (a, b) and j not in (a, b) and not (a < i < b < j or i < a < j < b) for a, b in chosen):
                chosen.append((i, j))
                enumerate_from(i + 1, chosen)
                chosen.pop()
    enumerate_from(0, [])
    return sorted(structures, key=lambda x: sorted(x))


def brute_reference(seq, scores, temperature, gamma, centroid=False):
    structures = all_structures(seq)
    log_weights = np.array([sum(scores[i, j] / temperature for i, j in s) for s in structures], dtype=float)
    log_z = float(logsumexp(log_weights))
    probabilities = np.exp(log_weights - log_z)
    n = len(seq)
    posterior = np.zeros((n, n), dtype=float)
    for structure, probability in zip(structures, probabilities):
        for i, j in structure:
            posterior[i, j] += probability
    posterior = posterior + posterior.T
    map_value = float(np.max(log_weights))
    q = np.clip(1 - posterior.sum(axis=1), 0, None)
    def objective(structure):
        if centroid:
            return float(sum(2 * posterior[i, j] - 1 for i, j in structure))
        return float(2 * gamma * posterior[tuple(zip(*structure))].sum() + sum(q[i] for i in range(n) if all(i not in p for p in structure))) if structure else float(q.sum())
    best_objective = max(objective(s) for s in structures)
    return log_z, posterior, map_value, best_objective


def brute_force_checks():
    from pair_decode_common import decode_centroid, decode_mea
    from rnajepa.harness import inside_outside, nussinov_map, valid_pair_mask
    rng = np.random.default_rng(20261119)
    failures = []
    for case in range(12):
        n = 6 + case % 3
        seq = "".join(rng.choice(list("ACGU"), size=n))
        scores = rng.normal(0, 2, size=(n, n))
        scores = (scores + scores.T) / 2
        np.fill_diagonal(scores, -1e4)
        mask = valid_pair_mask(seq)
        temperature = float([0.5, 1, 2, 4][case % 4])
        gamma = float([0.5, 1, 2, 4, 8][case % 5])
        log_z, posterior, map_expected, mea_expected = brute_reference(seq, scores, temperature, gamma)
        log_z_actual, posterior_actual = inside_outside(scores / temperature, mask)
        map_actual = nussinov_map(scores / temperature, mask)
        mea_actual = decode_mea(posterior_actual, mask, gamma)
        centroid_expected = brute_reference(seq, scores, temperature, 0.5, centroid=True)[3]
        centroid_actual = decode_centroid(posterior_actual, mask)
        q = np.clip(1 - posterior.sum(axis=1), 0, None)
        def objective(structure, g):
            return float(2 * g * sum(posterior[i, j] for i, j in structure) + sum(q[i] for i in range(n) if all(i not in pair for pair in structure)))
        checks = {
            "log_z": math.isclose(log_z, log_z_actual, rel_tol=1e-9, abs_tol=1e-9),
            "posterior": np.allclose(posterior, posterior_actual, atol=1e-10, rtol=1e-9),
            "map": math.isclose(map_expected, sum(scores[i, j] / temperature for i, j in map_actual), rel_tol=1e-9, abs_tol=1e-9),
            "mea": math.isclose(mea_expected, objective(mea_actual, gamma), rel_tol=1e-9, abs_tol=1e-9),
            "centroid": math.isclose(centroid_expected, sum(2 * posterior[i, j] - 1 for i, j in centroid_actual), rel_tol=1e-9, abs_tol=1e-9),
        }
        if not all(checks.values()):
            failures.append({"case": case, "seq": seq, "checks": checks})
    return {"n_cases": 12, "passed": not failures, "failures": failures}


def choose(rows, decoder):
    candidates = [r for r in rows if r.decoder == decoder]
    return max(candidates, key=lambda r: (r.micro_f1, r.macro_f1, r.micro_precision,
                                          -r.temperature, -(0 if pd.isna(r.gamma) else r.gamma)))


def verify_decode(decode_dir):
    decode_dir = Path(decode_dir)
    config = json.loads((decode_dir / "frozen_decode_config.json").read_text())["config"]
    per_seq = pd.read_parquet(decode_dir / "decode_per_sequence.parquet")
    metrics = pd.read_parquet(decode_dir / "decode_summary.parquet")
    failures = []
    # Independently recompute every aggregate from per-sequence TP/FP/FN.
    group_columns = ["dataset_id", "score_variant", "decoder", "temperature", "gamma"]
    records = per_seq.to_dict("records")
    recomputed = []
    for keys, group in per_seq.groupby(group_columns, dropna=False):
        tp, fp, fn = (int(group[k].sum()) for k in ("tp", "fp", "fn"))
        micro_p = tp / (tp + fp) if tp + fp else 0.0
        micro_r = tp / (tp + fn) if tp + fn else 0.0
        row = dict(zip(group_columns, keys))
        row.update({"micro_precision": micro_p, "micro_recall": micro_r,
                    "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
                    "macro_precision": float(group.precision.mean()), "macro_recall": float(group.recall.mean()),
                    "macro_f1": float(group.f1.mean()), "n_sequences": len(group)})
        recomputed.append(row)
    rec = pd.DataFrame(recomputed)
    compare = metrics.merge(rec, on=group_columns, how="outer", suffixes=("_reported", "_independent"), indicator=True)
    numeric = ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1", "n_sequences"]
    bad = compare[compare._merge != "both"]
    for column in numeric:
        bad = pd.concat([bad, compare[~np.isclose(compare[f"{column}_reported"], compare[f"{column}_independent"], atol=1e-12, rtol=1e-10)]])
    if not bad.empty:
        failures.append({"check": "aggregate_recalculation", "examples": bad.head(5).to_dict("records")})
    # Re-select VL0 parameters from metrics using the frozen tie-breaking rule.
    dev = metrics[metrics.dataset_id == "bprna_vl0"]
    for variant in sorted(dev.score_variant.unique()):
        rows = list(dev[dev.score_variant == variant].itertuples(index=False))
        for decoder, key in (("MEA", "mea"), ("CENTROID", "centroid")):
            best = choose(rows, decoder)
            expected = config[variant][key]
            if float(best.temperature) != float(expected["temperature"]):
                failures.append({"check": "selection_temperature", "variant": variant, "decoder": decoder})
            if decoder == "MEA" and float(best.gamma) != float(expected["gamma"]):
                failures.append({"check": "selection_gamma", "variant": variant, "gamma": float(best.gamma), "expected": expected["gamma"]})
        post = [r for r in rows if r.decoder == "INDEPENDENT_POST_MAXF1"]
        best_post = choose(post, "INDEPENDENT_POST_MAXF1")
        if float(best_post.temperature) != float(config[variant]["marginal_independent"]["temperature"]):
            failures.append({"check": "selection_posterior_temperature", "variant": variant})
    # External rows may only use the frozen configuration, never a re-tuned one.
    external = metrics[metrics.dataset_id != "bprna_vl0"]
    for row in external.itertuples(index=False):
        cfg = config[row.score_variant]
        allowed = row.decoder == "MAP"
        if row.decoder == "MEA":
            allowed = float(row.temperature) == float(cfg["mea"]["temperature"]) and float(row.gamma) == float(cfg["mea"]["gamma"])
        elif row.decoder == "CENTROID":
            allowed = float(row.temperature) == float(cfg["centroid"]["temperature"])
        elif row.decoder.startswith("INDEPENDENT_POST"):
            allowed = float(row.temperature) == float(cfg["marginal_independent"]["temperature"])
        elif row.decoder.startswith("INDEPENDENT_RAW"):
            allowed = float(row.temperature) == 1.0
        if not allowed:
            failures.append({"check": "external_frozen_application", "row": row._asdict()})
    result = {"status": "PASS" if not failures else "FAIL", "n_decode_rows": len(metrics),
              "n_per_sequence_rows": len(per_seq), "failures": failures,
              "bruteforce": brute_force_checks()}
    if not result["bruteforce"]["passed"]:
        result["status"] = "FAIL"
    (decode_dir / "independent_decode_verification.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-dir", default=str(ROOT / "decode_r2d"))
    args = parser.parse_args()
    print(json.dumps(verify_decode(args.decode_dir), indent=2, default=str))
    if (Path(args.decode_dir) / "independent_decode_verification.json").read_text().find('"status": "FAIL"') >= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
