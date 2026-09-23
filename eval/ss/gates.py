"""Quantitative gate verification (spec §8).

Verifies every gate: hard **G1-G5**, performance **P1-P7**, speed **S1-S9** and
science **SC1-SC4**.  The verifier emits a gate-verification table, lists the
gates that are **not** met, and refuses to declare ``submission_ready`` when any
hard gate (G1-G5) fails -- and, when a plan is supplied, when the analysis plan is
not frozen (spec §7.4).

S6 and S7 have dedicated implementations because they are the stated criteria for
whether contributions C1 and C2 hold:

* **S7** -- the DP-free calibration cost: ``|ECE(System-1) - ECE(exact marginals)|
  <= 0.02`` (C1).  Uses the frozen :func:`rnajepa.distill.calibration_cost` for
  matrix inputs, or scalar ECE values from the evidence.
* **S6** -- the compute-adaptive benefit at **matched average FLOPs** (C2): the
  gated model's average FLOPs must be no greater (within tolerance) than the
  heavier of the two pure routes, while its F1 is at least as high as both.

Every gate returns one of ``pass`` / ``fail`` / ``not_measured``.  ``not_measured``
is *not* a pass: an unmeasured hard gate blocks submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from rnajepa.distill import calibration_cost

from .ablations import ABLATION_KEYS
from .stats import AnalysisPlan, load_analysis_plan, require_frozen

PASS = "pass"
FAIL = "fail"
NOT_MEASURED = "not_measured"

#: S7 tolerance from spec §8.3.
S7_TOLERANCE = 0.02
#: G3 numerical-exactness tolerance (spec asks for bitwise agreement at L <= 12).
G3_TOLERANCE = 1e-12
#: Default tolerance for "matched" average FLOPs in S6.
S6_DEFAULT_FLOP_TOL = 0.10

HARD_GATES = ("G1", "G2", "G3", "G4", "G5")

HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")


@dataclass
class GateRow:
    gid: str
    group: str
    requirement: str
    status: str
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return {"gate": self.gid, "group": self.group, "requirement": self.requirement,
                "status": self.status, "detail": self.detail}


# ---------------------------------------------------------------------------
# evidence accessors
# ---------------------------------------------------------------------------
def _num(ev: Dict[str, object], key: str) -> Optional[float]:
    v = ev.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _flag(ev: Dict[str, object], key: str) -> Optional[bool]:
    v = ev.get(key)
    return None if v is None else bool(v)


def _result(passed: Optional[bool], ok: str, bad: str, missing: str
            ) -> Tuple[str, str]:
    if passed is None:
        return NOT_MEASURED, missing
    return (PASS, ok) if passed else (FAIL, bad)


# ---------------------------------------------------------------------------
# individual gate checks -> (status, detail)
# ---------------------------------------------------------------------------
def _g1(ev):
    v = _num(ev, "g1_illegal_structure_rate")
    return _result(None if v is None else v == 0.0,
                   f"illegal-structure rate = {v}", f"illegal-structure rate = {v} (must be 0)",
                   "g1_illegal_structure_rate not provided")


def _g2(ev):
    v = _num(ev, "g2_min_hairpin_violation_rate")
    return _result(None if v is None else v == 0.0,
                   f"min-hairpin violation rate = {v}",
                   f"min-hairpin violation rate = {v} (must be 0)",
                   "g2_min_hairpin_violation_rate not provided")


def _g3(ev):
    v = _num(ev, "g3_max_abs_err")
    if v is None:
        return NOT_MEASURED, "g3_max_abs_err not provided"
    ok = v <= G3_TOLERANCE
    return (PASS if ok else FAIL,
            f"max |Z/p_hat error| at L<=12 = {v:.3e} (tol {G3_TOLERANCE:.0e})")


def _g4(ev):
    hom = _num(ev, "g4_cross_split_homology_rate")
    cont = _num(ev, "g4_contamination_rate")
    disc = _flag(ev, "g4_contamination_disclosed")
    if hom is None or cont is None:
        return NOT_MEASURED, "g4_cross_split_homology_rate / g4_contamination_rate not provided"
    ok = hom == 0.0 and (cont == 0.0 or disc is True)
    return (PASS if ok else FAIL,
            f"cross-split homology = {hom}; contamination = {cont} "
            f"(disclosed={disc})")


def _g5(ev):
    v = _num(ev, "g5_missing_provenance")
    if v is None:
        return NOT_MEASURED, "g5_missing_provenance not provided"
    return (PASS if v == 0.0 else FAIL,
            f"missing baseline provenance entries = {int(v)}")


def _threshold_gate(ev, key, threshold, cmp_ge=True, label=None):
    v = _num(ev, key)
    label = label or key
    if v is None:
        return NOT_MEASURED, f"{label} not provided"
    ok = (v >= threshold) if cmp_ge else (v <= threshold)
    rel = ">=" if cmp_ge else "<="
    return (PASS if ok else FAIL, f"{label} = {v} (target {rel} {threshold})")


def _p1(ev):
    m, b = _num(ev, "p1_f1"), _num(ev, "p1_best_baseline")
    if m is None or b is None:
        return NOT_MEASURED, "p1_f1 / p1_best_baseline not provided"
    return (PASS if m >= b else FAIL, f"ArchiveII F1 {m} vs best baseline {b}")


def _p2(ev):
    m, b = _num(ev, "p2_f1"), _num(ev, "p2_best_baseline")
    sig = _flag(ev, "p2_significant_vs_e2efold")
    if m is None or b is None or sig is None:
        return NOT_MEASURED, "p2_f1 / p2_best_baseline / p2_significant_vs_e2efold not provided"
    ok = m >= b and sig
    return (PASS if ok else FAIL,
            f"bpRNA-new F1 {m} vs best baseline {b}; significant vs E2Efold = {sig}")


def _p3(ev):
    m, b = _num(ev, "p3_inf"), _num(ev, "p3_best_baseline")
    if m is None or b is None:
        return NOT_MEASURED, "p3_inf / p3_best_baseline not provided"
    return (PASS if m >= b else FAIL, f"INF {m} vs best baseline {b}")


def _p4(ev):
    return _threshold_gate(ev, "p4_ece", 0.05, cmp_ge=False, label="pair ECE")


def _p5(ev):
    m, t = _num(ev, "p5_marginal_cal_error"), _num(ev, "p5_teacher_marginal_cal_error")
    tol = _num(ev, "p5_tol") or 0.0
    if m is None or t is None:
        return NOT_MEASURED, "p5_marginal_cal_error / p5_teacher_marginal_cal_error not provided"
    ok = m <= t + tol
    return (PASS if ok else FAIL, f"marginal-cal error {m} vs McCaskill teacher {t} (tol {tol})")


def _p6(ev):
    return _threshold_gate(ev, "p6_l0_helix_recall", 0.98, cmp_ge=True, label="L0 helix recall")


def _p7(ev):
    v = _num(ev, "p7_structure_calibration_ece")
    if v is None:
        return NOT_MEASURED, "p7_structure_calibration_ece not reported"
    return PASS, f"structure-level calibration reported (ECE = {v})"


def _s1(ev):
    return _threshold_gate(ev, "s1_speedup_vs_mccaskill", 50.0, label="speedup vs McCaskill")


def _s2(ev):
    return _threshold_gate(ev, "s2_speedup_vs_e2efold", 5.0, label="speedup vs E2Efold")


def _s3(ev):
    return _threshold_gate(ev, "s3_latency_ms_l512", 10.0, cmp_ge=False,
                           label="latency (L=512, System-1, ms)")


def _s4(ev):
    run = _flag(ev, "s4_runnable_l2048")
    decay = _num(ev, "s4_f1_decay")
    if run is None or decay is None:
        return NOT_MEASURED, "s4_runnable_l2048 / s4_f1_decay not provided"
    ok = run and decay <= 0.15
    return (PASS if ok else FAIL,
            f"L=2048 runnable = {run}; F1 decay vs L<=512 = {decay} (target <= 0.15)")


def _s5(ev):
    v = _flag(ev, "s5_pareto_dominant")
    return _result(v, "on the latency-accuracy Pareto frontier",
                   "not Pareto-dominant", "s5_pareto_dominant not provided")


def check_s6(ev: Dict[str, object]) -> Tuple[str, str]:
    """S6: compute-adaptive benefit at **matched average FLOPs** (spec §8.3, C2).

    Passes iff the gated model's average FLOPs do not exceed the heavier of the two
    pure routes by more than ``s6_flops_tol`` (i.e. compute really is matched) and
    its F1 is at least as high as both pure routes.
    """
    f1g = _num(ev, "s6_f1_gated")
    f1a = _num(ev, "s6_f1_system1")
    f1b = _num(ev, "s6_f1_system2")
    fg = _num(ev, "s6_avg_flops_gated")
    fa = _num(ev, "s6_avg_flops_system1")
    fb = _num(ev, "s6_avg_flops_system2")
    tol = _num(ev, "s6_flops_tol")
    tol = S6_DEFAULT_FLOP_TOL if tol is None else tol
    if any(v is None for v in (f1g, f1a, f1b, fg, fa, fb)):
        return NOT_MEASURED, "s6 F1 / average-FLOPs for gated and both pure routes required"
    ref = max(fa, fb)
    matched = fg <= ref * (1.0 + tol)
    better = f1g >= max(f1a, f1b)
    ok = matched and better
    return (PASS if ok else FAIL,
            f"F1 gated {f1g} vs system1 {f1a} / system2 {f1b}; "
            f"avg FLOPs gated {fg:.3g} vs matched reference {ref:.3g} (tol {tol})")


def check_s7(ev: Dict[str, object]) -> Tuple[str, str]:
    """S7: DP-free calibration cost, ``|ECE(System-1) - ECE(exact)| <= 0.02`` (C1).

    Prefers matrix inputs (``s7_student_probs`` / ``s7_exact_marginals`` /
    ``s7_labels``) so the frozen :func:`rnajepa.distill.calibration_cost` is used;
    falls back to scalar ECE values or an explicit delta.
    """
    if all(k in ev for k in ("s7_student_probs", "s7_exact_marginals", "s7_labels")):
        res = calibration_cost(ev["s7_student_probs"], ev["s7_exact_marginals"],
                               ev["s7_labels"], ev.get("s7_mask"), tol=S7_TOLERANCE)
        ok = bool(res["passes_s7"])
        return (PASS if ok else FAIL,
                f"ECE student {res['ece_student']:.4f} vs exact {res['ece_exact']:.4f} "
                f"(|delta| {res['delta']:.4f} <= {S7_TOLERANCE})")
    delta = _num(ev, "s7_delta")
    if delta is None:
        a, b = _num(ev, "s7_ece_system1"), _num(ev, "s7_ece_exact")
        if a is not None and b is not None:
            delta = abs(a - b)
    if delta is None:
        return NOT_MEASURED, "s7_delta / (s7_ece_system1, s7_ece_exact) / matrices not provided"
    ok = delta <= S7_TOLERANCE
    return (PASS if ok else FAIL,
            f"|ECE(System-1) - ECE(exact marginals)| = {delta:.4f} (target <= {S7_TOLERANCE})")


def _s8(ev):
    flat_f = _num(ev, "s8_flat_flops")
    casc_f = _num(ev, "s8_cascade_flops")
    flat_f1 = _num(ev, "s8_flat_f1")
    casc_f1 = _num(ev, "s8_cascade_f1")
    if any(v is None for v in (flat_f, casc_f, flat_f1, casc_f1)):
        return NOT_MEASURED, "s8 flat/cascade FLOPs and F1 (at L>=1024) required"
    ok = casc_f < flat_f and casc_f1 >= flat_f1
    return (PASS if ok else FAIL,
            f"cascade FLOPs {casc_f:.3g} < flat {flat_f:.3g}; "
            f"cascade F1 {casc_f1} >= flat {flat_f1}")


def _s9(ev):
    casc = _num(ev, "s9_cascade_longrange_f1")
    flat = _num(ev, "s9_flat_longrange_f1")
    band = _num(ev, "s9_banded_longrange_f1")
    if any(v is None for v in (casc, flat, band)):
        return NOT_MEASURED, "s9 cascade/flat/banded long-range F1 required"
    ok = casc >= flat and casc > band
    return (PASS if ok else FAIL,
            f"long-range F1: cascade {casc} >= flat {flat} and > banded {band}")


def _sc1(ev):
    concl = ev.get("sc1_hypotheses")
    if not isinstance(concl, dict):
        return NOT_MEASURED, "sc1_hypotheses (dict H1..H7 -> conclusion) not provided"
    missing = [h for h in HYPOTHESES if not str(concl.get(h, "")).strip()]
    if missing:
        return FAIL, f"hypotheses without a conclusion: {missing}"
    return PASS, "H1-H7 all have a conclusion (supported or falsified)"


def _sc2(ev):
    falsified = ev.get("sc2_falsified")
    mechanisms = ev.get("sc2_mechanisms")
    if falsified is None or not isinstance(mechanisms, dict):
        return NOT_MEASURED, "sc2_falsified / sc2_mechanisms not provided"
    missing = [h for h in falsified if not str(mechanisms.get(h, "")).strip()]
    if missing:
        return FAIL, f"falsified hypotheses without a mechanism explanation: {missing}"
    return PASS, f"mechanism given for every falsified hypothesis ({list(falsified)})"


def _sc3(ev):
    done = ev.get("sc3_completed_ablations")
    if done is None:
        return NOT_MEASURED, "sc3_completed_ablations not provided"
    done_set = set(done)
    missing = [k for k in ABLATION_KEYS if k not in done_set]
    if missing:
        return FAIL, f"incomplete ablations ({len(missing)}): {missing}"
    return PASS, f"all {len(ABLATION_KEYS)} spec §7.5 ablations completed"


def _sc4(ev):
    reported = _flag(ev, "sc4_negative_results_reported")
    selective = _flag(ev, "sc4_selective_reporting")
    if reported is None:
        return NOT_MEASURED, "sc4_negative_results_reported not provided"
    ok = reported and selective is not True
    return (PASS if ok else FAIL,
            f"negative results reported = {reported}; selective reporting = {selective}")


#: (gate id, group, requirement text, checker) in spec order.
GATE_SPECS: Tuple[Tuple[str, str, str, Callable[[Dict[str, object]], Tuple[str, str]]], ...] = (
    ("G1", "hard", "illegal-structure rate = 0", _g1),
    ("G2", "hard", "min-hairpin violation rate = 0", _g2),
    ("G3", "hard", "Z(x)/p_hat exact vs brute force at L<=12", _g3),
    ("G4", "hard", "cross-split homology = 0; contamination = 0 or disclosed", _g4),
    ("G5", "hard", "baseline commit + weight hash recorded, 0 missing", _g5),
    ("P1", "performance", "ArchiveII F1 >= strongest baseline", _p1),
    ("P2", "performance", "bpRNA-new F1 >= strongest baseline and significantly > E2Efold", _p2),
    ("P3", "performance", "INF >= strongest baseline", _p3),
    ("P4", "performance", "pair ECE <= 0.05", _p4),
    ("P5", "performance", "marginal calibration comparable/better than McCaskill teacher", _p5),
    ("P6", "performance", "L0 helix recall >= 0.98", _p6),
    ("P7", "performance", "structure-level calibration reported", _p7),
    ("S1", "speed", "speedup vs McCaskill >= 50x (L=512, System-1)", _s1),
    ("S2", "speed", "speedup vs E2Efold >= 5x (L=512)", _s2),
    ("S3", "speed", "latency <= 10 ms (L=512, GPU, batch=1, System-1)", _s3),
    ("S4", "speed", "runnable at L=2048, F1 decay <= 15%", _s4),
    ("S5", "speed", "latency-accuracy Pareto dominance", _s5),
    ("S6", "speed", "compute-adaptive benefit at matched average FLOPs (C2)", check_s6),
    ("S7", "speed", "DP-free calibration cost |dECE| <= 0.02 (C1)", check_s7),
    ("S8", "speed", "cascade FLOPs << flat head at L>=1024, F1 not worse (C2)", _s8),
    ("S9", "speed", "cascade keeps long-range pairs, beats banding", _s9),
    ("SC1", "science", "H1-H7 all concluded", _sc1),
    ("SC2", "science", "falsified hypotheses have a mechanism explanation", _sc2),
    ("SC3", "science", "all 15 spec §7.5 ablations completed", _sc3),
    ("SC4", "science", "negative results reported, no selective reporting", _sc4),
)


@dataclass
class GateReport:
    rows: List[GateRow]
    plan_frozen: Optional[bool] = None
    plan_note: str = ""

    @property
    def unmet(self) -> List[GateRow]:
        return [r for r in self.rows if r.status != PASS]

    @property
    def failed(self) -> List[GateRow]:
        return [r for r in self.rows if r.status == FAIL]

    @property
    def hard_failures(self) -> List[GateRow]:
        return [r for r in self.rows if r.group == "hard" and r.status != PASS]

    @property
    def submission_ready(self) -> bool:
        """True only if every hard gate passes **and** the plan is frozen."""
        if self.hard_failures:
            return False
        if self.plan_frozen is False:
            return False
        return True

    def blockers(self) -> List[str]:
        out = [f"{r.gid}: {r.detail}" for r in self.hard_failures]
        if self.plan_frozen is False:
            out.append(f"analysis plan not frozen: {self.plan_note}")
        return out

    def render_table(self) -> str:
        lines = [f"{'gate':<5} {'group':<12} {'status':<13} detail",
                 "-" * 96]
        for r in self.rows:
            lines.append(f"{r.gid:<5} {r.group:<12} {r.status:<13} {r.detail}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "submission_ready": self.submission_ready,
            "plan_frozen": self.plan_frozen,
            "plan_note": self.plan_note,
            "rows": [r.as_dict() for r in self.rows],
            "unmet": [r.gid for r in self.unmet],
            "hard_failures": [r.gid for r in self.hard_failures],
            "blockers": self.blockers(),
        }


def verify_gates(evidence: Dict[str, object],
                 plan: Optional[AnalysisPlan] = None) -> GateReport:
    """Check every gate against ``evidence`` and return a :class:`GateReport`.

    ``evidence`` is a flat dict keyed by the evidence names documented on each
    checker.  Missing keys yield ``not_measured`` (never a silent pass).
    """
    rows: List[GateRow] = []
    for gid, group, requirement, checker in GATE_SPECS:
        status, detail = checker(evidence)
        rows.append(GateRow(gid=gid, group=group, requirement=requirement,
                            status=status, detail=detail))

    plan_frozen: Optional[bool] = None
    plan_note = ""
    if plan is not None:
        try:
            require_frozen(plan)
            plan_frozen = True
            plan_note = f"plan {plan.path} is frozen ({plan.freeze_timestamp_raw})"
        except Exception as exc:  # PlanNotFrozenError and parse failures both block
            plan_frozen = False
            plan_note = str(exc)
    return GateReport(rows=rows, plan_frozen=plan_frozen, plan_note=plan_note)


def verify_all(evidence: Dict[str, object], plan_path: Optional[str] = None) -> GateReport:
    """Convenience: load the analysis plan from ``plan_path`` and verify all gates."""
    plan = load_analysis_plan(plan_path) if plan_path else None
    return verify_gates(evidence, plan)


__all__ = [
    "FAIL",
    "GATE_SPECS",
    "HARD_GATES",
    "HYPOTHESES",
    "NOT_MEASURED",
    "PASS",
    "S6_DEFAULT_FLOP_TOL",
    "S7_TOLERANCE",
    "GateReport",
    "GateRow",
    "check_s6",
    "check_s7",
    "verify_all",
    "verify_gates",
]
