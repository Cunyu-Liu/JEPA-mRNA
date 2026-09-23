"""CDPFold calibration check -- the project-blocking item (spec §0.9.4 / §7.2.1, gate S7).

Why this is the single highest-priority item
-------------------------------------------
CDPFold is a conditional diffusion model that **already outputs an ``L x L``
pair-probability matrix with no DP at all**.  That directly threatens the
novelty of contribution C1 ("DP-free **calibration**", spec §0.8 problem 6):
"no DP but a probability matrix" is not new; the claimed gap is "no DP *and* the
probabilities are calibrated".  If CDPFold's calibration is as good as ours,
C1 does not hold and the paper skeleton must fall back to **C2 + C4**.

What this harness does
----------------------
1. Loads CDPFold predictions (a probability matrix) -- or a user-supplied one.
2. Measures its calibration: ECE, reliability diagram, marginal calibration,
   Brier and NLL.
3. Compares head-to-head against our System-1 head's calibration and against the
   exact partition-function marginals (the System-2 reference).
4. Emits an explicit verdict using the S7 threshold ``|ΔECE| <= 0.02``:
   * CDPFold within the threshold of ours  -> **C1 不成立**, fallback ``C2 + C4``;
   * CDPFold better than ours (ΔECE < -tol) -> **C1 不成立**, fallback ``C2 + C4``;
   * CDPFold meaningfully worse (ΔECE > tol) -> **C1 成立**, skeleton ``C2 + C1``.
5. Writes a decision record to ``records/`` in the style of the existing files.

Everything is testable on **synthetic** probability matrices, so the verdict
logic can be exercised without the real CDPFold model (which is not available
here, and whose citation itself is marked 待核验 in ``spec/citation_register.csv``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

# Importable both as a top-level module (``import cdpfold_check``) and as part of
# the ``ss`` package (``import ss.cdpfold_check``).
try:
    from baselines import calibration_metrics  # noqa: E402
except ImportError:  # pragma: no cover - package import path
    from .baselines import calibration_metrics  # type: ignore  # noqa: E402

#: S7 threshold: the DP-free student's ECE must be within 0.02 of the exact
#: marginals' ECE (spec §8.3).  Reused here as the CDPFold-vs-ours threshold.
S7_TOL = 0.02

RECORDS_DIR = os.path.join(_ROOT, "records")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_prob_matrix(path: str) -> np.ndarray:
    """Load an ``L x L`` probability matrix from ``.npy`` or a delimited text file."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    with open(path) as fh:
        rows = [[float(x) for x in line.replace(",", " ").split()]
                for line in fh if line.strip()]
    m = np.asarray(rows, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {m.shape}")
    return m


def load_labels(path: str) -> np.ndarray:
    return load_prob_matrix(path)


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def c1_verdict(ece_cdp: float, ece_ours: float, ece_exact: float,
               *, tol: float = S7_TOL) -> Dict[str, object]:
    """Decide whether contribution C1 holds, from the three ECEs.

    * ``within_tol`` : ``|ece_cdp - ece_ours| <= tol`` -> CDPFold is calibrated
      as well as ours -> C1 does NOT hold.
    * ``delta < -tol`` : CDPFold is *better* than ours -> C1 does NOT hold.
    * ``delta > tol``  : CDPFold is meaningfully worse -> C1 holds.
    """
    delta = ece_cdp - ece_ours
    within_tol = abs(delta) <= tol
    if within_tol:
        c1_holds = False
        reason = (f"CDPFold 校准与本文相当（|ΔECE|={abs(delta):.4f} ≤ {tol}）"
                  f" -> C1 不成立")
    elif delta < 0:
        c1_holds = False
        reason = f"CDPFold 校准优于本文（ΔECE={delta:.4f} < -{tol}）-> C1 不成立"
    else:
        c1_holds = True
        reason = f"CDPFold 校准显著更差（ΔECE={delta:.4f} > {tol}）-> C1 成立"
    return {
        "delta_ece_cdp_minus_ours": float(delta),
        "tol": float(tol),
        "within_tol": bool(within_tol),
        "c1_holds": bool(c1_holds),
        "reason": reason,
        "verdict": "C1 成立（免 DP 的校准成立）" if c1_holds else "C1 不成立",
        "fallback": "C2 + C1" if c1_holds else "C2 + C4",
        "s7_ours_vs_exact": bool(abs(ece_ours - ece_exact) <= tol),
    }


def compare_cdpfold(cdp_probs, ours_probs, exact_marginals, labels, *,
                    mask=None, tol: float = S7_TOL, n_bins: int = 10) -> Dict[str, object]:
    """Head-to-head calibration comparison.  All inputs are ``L x L`` matrices."""
    cdp = calibration_metrics(cdp_probs, labels, mask, n_bins)
    ours = calibration_metrics(ours_probs, labels, mask, n_bins)
    exact = calibration_metrics(exact_marginals, labels, mask, n_bins)
    verdict = c1_verdict(cdp["ece"], ours["ece"], exact["ece"], tol=tol)
    return {
        "cdpfold": cdp,
        "ours_system1": ours,
        "exact_marginals": exact,
        "verdict": verdict,
        "tol": float(tol),
    }


# ---------------------------------------------------------------------------
# synthetic cases (so the logic is testable without the real model)
# ---------------------------------------------------------------------------
def synthetic_case(kind: str, *, L: int = 12, seed: int = 0) -> Dict[str, object]:
    """Build a controlled ``(cdp, ours, exact, labels)`` case.

    ``kind`` is one of ``c1_holds`` / ``c1_fails_within_tol`` /
    ``c1_fails_cdp_better``.  Probabilities are crafted so the ECEs are known.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(i + 3, L):
            v = 1.0 if rng.random() < 0.25 else 0.0
            labels[i, j] = labels[j, i] = v

    def sharp(a: float, b: float) -> np.ndarray:
        m = np.where(labels > 0.5, a, b)
        np.fill_diagonal(m, 0.0)
        return m

    if kind == "c1_holds":
        ours = sharp(0.99, 0.01)          # ECE ~ 0.01
        exact = ours.copy()
        cdp = np.full((L, L), 0.5)        # ECE = |0.5 - base_rate|, large
        np.fill_diagonal(cdp, 0.0)
    elif kind == "c1_fails_within_tol":
        ours = sharp(0.99, 0.01)
        exact = ours.copy()
        cdp = ours.copy()                 # identical -> delta 0 -> within tol
    elif kind == "c1_fails_cdp_better":
        ours = sharp(0.90, 0.10)          # ECE ~ 0.10
        exact = sharp(0.99, 0.01)         # ECE ~ 0.01
        cdp = sharp(0.999, 0.001)         # ECE ~ 0.001 -> better than ours
    else:
        raise ValueError(f"unknown synthetic case: {kind!r}")
    return {"kind": kind, "L": L, "labels": labels,
            "cdpfold": cdp, "ours": ours, "exact": exact, "mask": None}


# ---------------------------------------------------------------------------
# decision record
# ---------------------------------------------------------------------------
def render_decision_record(result: Dict[str, object], *, synthetic: bool = False,
                           inputs: Optional[Dict[str, str]] = None) -> str:
    v = result["verdict"]  # type: ignore[index]
    cdp = result["cdpfold"]  # type: ignore[index]
    ours = result["ours_system1"]  # type: ignore[index]
    exact = result["exact_marginals"]  # type: ignore[index]
    lines: List[str] = []
    lines.append("# CDPFold calibration verdict (gate S7 / contribution C1)")
    lines.append("")
    if synthetic:
        lines.append("**SYNTHETIC SELF-TEST — NOT a real CDPFold verdict.**")
        lines.append("")
        lines.append("This record was produced from crafted probability matrices to exercise the")
        lines.append("verdict logic. It is a PLACEHOLDER: the real verdict must be re-generated once")
        lines.append("CDPFold predictions (or a published probability matrix) are available. The")
        lines.append("CDPFold citation itself is marked 待核验 in `spec/citation_register.csv`.")
    else:
        lines.append("Produced from supplied probability matrices.")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat()}Z")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    if inputs:
        for k, val in inputs.items():
            lines.append(f"- {k}: `{val}`")
    else:
        lines.append("- (in-memory synthetic matrices)")
    lines.append("")
    lines.append("## Calibration metrics")
    lines.append("")
    lines.append("| model | ECE | marginal calib. | Brier | NLL | mean pred | mean empirical |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, m in (("CDPFold", cdp), ("ours (System-1)", ours), ("exact marginals", exact)):
        lines.append("| {n} | {e:.4f} | {mc:.4f} | {b:.4f} | {nl:.4f} | {mp:.4f} | {me:.4f} |".format(
            n=label, e=m["ece"], mc=m["marginal_calibration"], b=m["brier"], nl=m["nll"],
            mp=m["mean_pred"], me=m["mean_empirical"]))
    lines.append("")
    lines.append("## Verdict (S7 threshold |ΔECE| ≤ {tol})".format(tol=v["tol"]))
    lines.append("")
    lines.append(f"- ΔECE (CDPFold − ours) = **{v['delta_ece_cdp_minus_ours']:.4f}**")
    lines.append(f"- within threshold: **{v['within_tol']}**")
    lines.append(f"- **C1 holds: {v['c1_holds']}**")
    lines.append(f"- reason: {v['reason']}")
    lines.append(f"- ours vs exact (S7 for our own head): {v['s7_ours_vs_exact']}")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if v["c1_holds"]:
        lines.append("Paper skeleton stays **C2 + C1**: CDPFold's DP-free probabilities are")
        lines.append("meaningfully less calibrated than ours, so \"DP-free *and* calibrated\" remains")
        lines.append("a defensible novelty.")
    else:
        lines.append("**Fallback engaged: paper skeleton becomes C2 + C4.** CDPFold's calibration is")
        lines.append("not measurably worse than ours, so C1 (\"DP-free calibration\") does not hold as")
        lines.append("a standalone contribution. The skeleton falls back to C2 (hierarchical cascade)")
        lines.append("plus C4 (calibration-first evaluation protocol), per spec §0.9.4 / §2.1.")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("Calibration is data- and length-dependent. A single matrix is not a verdict; the")
    lines.append("real decision requires CDPFold evaluated on the frozen benchmarks (ArchiveII")
    lines.append("de-redundified + bpRNA-new) with the same binning and mask as our own numbers.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_decision_record(path: str, result: Dict[str, object], *, synthetic: bool = False,
                          inputs: Optional[Dict[str, str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = render_decision_record(result, synthetic=synthetic, inputs=inputs)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CDPFold calibration check (gate S7 / C1)")
    ap.add_argument("--cdpfold", default="", help="CDPFold probability matrix (.npy/.txt/.csv)")
    ap.add_argument("--ours", default="", help="our System-1 probability matrix")
    ap.add_argument("--exact", default="", help="exact partition-function marginals matrix")
    ap.add_argument("--labels", default="", help="binary pair-indicator matrix (ground truth)")
    ap.add_argument("--synthetic", default="", choices=["", "c1_holds", "c1_fails_within_tol",
                                                        "c1_fails_cdp_better"],
                    help="run a crafted synthetic case instead of real matrices")
    ap.add_argument("--tol", type=float, default=S7_TOL)
    ap.add_argument("--record", default="", help="decision-record path (default: records/...)")
    ap.add_argument("--json", default="", help="write the full result JSON here")
    args = ap.parse_args(argv)

    if args.synthetic:
        case = synthetic_case(args.synthetic)
        result = compare_cdpfold(case["cdpfold"], case["ours"], case["exact"],
                                 case["labels"], mask=case["mask"], tol=args.tol)
        synthetic = True
        inputs = {"synthetic_case": args.synthetic}
    else:
        if not (args.cdpfold and args.ours and args.exact and args.labels):
            print("[cdpfold] need --cdpfold --ours --exact --labels (or --synthetic)", file=sys.stderr)
            return 2
        result = compare_cdpfold(load_prob_matrix(args.cdpfold), load_prob_matrix(args.ours),
                                 load_prob_matrix(args.exact), load_labels(args.labels),
                                 tol=args.tol)
        synthetic = False
        inputs = {"cdpfold": args.cdpfold, "ours": args.ours, "exact": args.exact,
                  "labels": args.labels}

    record = args.record or os.path.join(
        RECORDS_DIR, "CDPFold_calibration_verdict.PLACEHOLDER.md" if synthetic
        else "CDPFold_calibration_verdict.md")
    write_decision_record(record, result, synthetic=synthetic, inputs=inputs)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

    v = result["verdict"]
    print(f"[cdpfold] ECE cdp={result['cdpfold']['ece']:.4f} "  # type: ignore[index]
          f"ours={result['ours_system1']['ece']:.4f} "  # type: ignore[index]
          f"exact={result['exact_marginals']['ece']:.4f} "  # type: ignore[index]
          f"| Δ={v['delta_ece_cdp_minus_ours']:.4f} | C1_holds={v['c1_holds']} "
          f"| skeleton={v['fallback']}")
    print(f"[cdpfold] record -> {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
