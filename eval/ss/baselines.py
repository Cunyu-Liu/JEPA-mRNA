"""Baseline reproduction matrix for RNA secondary-structure prediction (spec §7.2).

Two things this module is built to make impossible:

1. **Cherry-picking the comparison.**  Every route in spec §7.2.2 is registered,
   and the four *closest prior work* models (spec §7.2.1) -- CONTRAfold,
   CDPFold, LinearPartition, E2Efold -- are flagged ``closest_prior=True`` and
   cannot be quietly dropped.  LinearPartition's speedup is reported as its own
   field so the "fast probability" route is never reduced to a single
   favourable number.
2. **Missing reproducibility provenance.**  Gate G5 requires *every* baseline to
   carry a repo commit, a weight hash and an evaluation-script version, with
   zero missing entries.  :func:`assert_reproducible` enforces this and raises
   with the exact list of gaps.  Because nothing is installed here and there is
   no network, the registry currently fails that gate *on purpose* -- commits and
   weight hashes are marked ``待核验`` rather than invented.

Every runner emits F1, INF **and** the calibration metrics (ECE, marginal
calibration, Brier, NLL), because the CONTRAfold / CDPFold comparison hinges on
calibration, not just F1 (spec §7.1 / §7.2.1).

No tool is installed and there is no network, so :func:`run_baseline` raises an
actionable :class:`ToolUnavailableError` naming the tool, the required version
lock and the install hint.  It does **not** fabricate outputs.  A ``--dry-run``
mode prints what would be run.  If a caller supplies real predictions
(``probs`` / ``pred_pairs``) the metrics are computed for real from them.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from rnajepa.distill import expected_calibration_error  # noqa: E402

#: Version string of this evaluation script, recorded per baseline (gate G5).
EVAL_SCRIPT_VERSION = "eval/ss/baselines.py@v1"

#: Metrics every runner must emit (spec §7.1): pairing, structure, calibration.
REQUIRED_METRICS = ("f1", "inf", "ece", "marginal_calibration", "brier", "nll")

HASH_UNKNOWN = "待核验"

# routes (spec §7.2.2)
ROUTE_THERMO = "thermodynamic_dp"
ROUTE_LINEAR = "linear_dp"
ROUTE_DISCRIMINATIVE = "discriminative_deep"
ROUTE_FOUNDATION = "rna_foundation"
ROUTE_AUTOREG = "autoregressive_generative"
ROUTE_3D = "3d_end_to_end"

ROUTES = (ROUTE_THERMO, ROUTE_LINEAR, ROUTE_DISCRIMINATIVE, ROUTE_FOUNDATION,
          ROUTE_AUTOREG, ROUTE_3D)


class ToolUnavailableError(RuntimeError):
    """Raised when a baseline tool is required but not installed here.

    Carries the tool name, the required version lock and an install hint so the
    failure is actionable rather than a bare traceback.
    """

    def __init__(self, message: str, *, key: str, tool: str,
                 version_lock: str, install_hint: str) -> None:
        super().__init__(message)
        self.key = key
        self.tool = tool
        self.version_lock = version_lock
        self.install_hint = install_hint


class ReproducibilityGateError(RuntimeError):
    """Raised when the G5 provenance gate fails (missing commit / weight hash)."""


@dataclass
class Baseline:
    key: str
    name: str
    route: str
    spec_section: str
    closest_prior: bool = False
    needs_dp: bool = False
    report_speedup_separately: bool = False
    repo: str = ""
    commit: str = ""
    weight_hash: str = ""
    eval_script_version: str = EVAL_SCRIPT_VERSION
    version_lock: str = ""
    install_hint: str = ""
    licence: str = HASH_UNKNOWN
    emits: Tuple[str, ...] = REQUIRED_METRICS
    notes: str = ""
    to_select: bool = False


def _b(**kw) -> Baseline:
    return Baseline(**kw)


def registry() -> List[Baseline]:
    """All baselines from spec §7.2.2 (plus the §7.2.1 closest prior work)."""
    return [
        # ---- §7.2.2 thermodynamic DP ----
        _b(key="rnafold", name="RNAfold (ViennaRNA 2.x)", route=ROUTE_THERMO, spec_section="§7.2.2",
           needs_dp=True, repo="https://github.com/ViennaRNA/ViennaRNA",
           version_lock="ViennaRNA 2.x（Turner 参数随版本变化，须锁定精确 minor）",
           install_hint="conda install -c bioconda viennarna=2.6.*（或源码编译）",
           notes="热力学 DP 基线；同时是 System-2 教师候选"),
        _b(key="rnastructure", name="RNAstructure", route=ROUTE_THERMO, spec_section="§7.2.2",
           needs_dp=True, repo="https://github.com/RNAstructure/RNAstructure",
           version_lock="RNAstructure（锁定版本）",
           install_hint="conda install -c bioconda rnastructure（或官网编译）"),
        _b(key="contrafold", name="CONTRAfold", route=ROUTE_THERMO, spec_section="§7.2.1",
           closest_prior=True, needs_dp=True, repo="http://contra.stanford.edu/contrafold/",
           version_lock="CONTRAfold 2.02",
           install_hint="从官方站点下载源码编译（无 pip 包）",
           notes="**本方案的直接前身**：同结构空间对数线性模型 + DP 精确配分函数 + 条件似然训练；"
                 "必须对比 F1 / ECE / 边际校准 / 延迟 / 是否需 DP"),
        # ---- §7.2.2 linear-time DP ----
        _b(key="linearfold", name="LinearFold", route=ROUTE_LINEAR, spec_section="§7.2.2",
           needs_dp=True, report_speedup_separately=True,
           repo="https://github.com/LinearFold/LinearFold",
           version_lock="LinearFold（锁定 commit）",
           install_hint="git clone + make（无 pip 包）"),
        _b(key="linearpartition", name="LinearPartition", route=ROUTE_LINEAR, spec_section="§7.2.1",
           closest_prior=True, needs_dp=True, report_speedup_separately=True,
           repo="https://github.com/LinearFold/LinearPartition",
           version_lock="LinearPartition（锁定 commit）",
           install_hint="git clone + make（无 pip 包）",
           notes="线性时间配分函数；**速度增益必须单独报告**（report_speedup_separately），"
                 "不得只挑有利的一方"),
        # ---- §7.2.1 closest prior: DP-free probability matrix ----
        _b(key="cdpfold", name="CDPFold (conditional diffusion)", route=ROUTE_DISCRIMINATIVE,
           spec_section="§7.2.1", closest_prior=True, needs_dp=False, repo="",
           version_lock="CDPFold（版本未确认，待核验）",
           install_hint="官方仓库/权重未确认（citation_register.csv 标为待核验）",
           notes="**阻塞项**：同为『免 DP 直接输出配对概率矩阵』，直接威胁 C1 新颖性；"
                 "必须证明我们的校准优于它（eval/ss/cdpfold_check.py）"),
        # ---- §7.2.2 discriminative deep ----
        _b(key="ufold", name="UFold", route=ROUTE_DISCRIMINATIVE, spec_section="§7.2.2",
           repo="https://github.com/uci-cbcl/UFold", version_lock="UFold（锁定 commit + 权重）",
           install_hint="git clone + 下载官方权重（torch）"),
        _b(key="spotrna", name="SPOT-RNA", route=ROUTE_DISCRIMINATIVE, spec_section="§7.2.2",
           repo="https://github.com/jaswindersingh2/SPOT-RNA",
           version_lock="SPOT-RNA（锁定 commit + 权重）", install_hint="git clone + 官方权重"),
        _b(key="spotrna2", name="SPOT-RNA2", route=ROUTE_DISCRIMINATIVE, spec_section="§7.2.2",
           repo="https://github.com/jaswindersingh2/SPOT-RNA2",
           version_lock="SPOT-RNA2（锁定 commit + 权重）", install_hint="git clone + 官方权重"),
        _b(key="mxfold2", name="MXfold2", route=ROUTE_DISCRIMINATIVE, spec_section="§7.2.2",
           needs_dp=True, repo="https://github.com/mxfold/mxfold2",
           version_lock="MXfold2（锁定 commit + 权重）", install_hint="pip install mxfold2（或源码）"),
        _b(key="e2efold", name="E2Efold", route=ROUTE_DISCRIMINATIVE, spec_section="§7.2.1",
           closest_prior=True, needs_dp=True, repo="https://github.com/ml4bio/e2efold",
           version_lock="E2Efold（锁定 commit + 权重）",
           install_hint="git clone + 官方权重（需 GPU；展开式可微 DP）",
           notes="未见家族上 F≈0.0361（MXfold2 论文报告）——跨家族泛化的关键对照"),
        # ---- §7.2.2 RNA foundation models ----
        _b(key="rnafm", name="RNA-FM", route=ROUTE_FOUNDATION, spec_section="§7.2.2",
           repo="https://github.com/ml4bio/RNA-FM", version_lock="RNA-FM（锁定权重哈希）",
           install_hint="pip install rna-fm + 官方权重"),
        _b(key="rinalmo", name="RiNALMo (650M)", route=ROUTE_FOUNDATION, spec_section="§7.2.2",
           repo="https://github.com/lbcb-sci/RiNALMo", version_lock="RiNALMo 650M（锁定权重哈希）",
           install_hint="pip install rinalmo + 官方权重（650M 规模）"),
        _b(key="mrnabert", name="mRNABERT", route=ROUTE_FOUNDATION, spec_section="§7.2.2",
           repo="https://huggingface.co/YYLY66/mRNABERT",
           version_lock="mRNABERT（本机已有权重，哈希待核验）",
           install_hint="本地权重已在 /mnt/cunyuliu；需记录权重哈希"),
        _b(key="rnamsm", name="RNA-MSM", route=ROUTE_FOUNDATION, spec_section="§7.2.2",
           repo="https://github.com/yikunpku/RNA-MSM", version_lock="RNA-MSM（锁定权重哈希）",
           install_hint="git clone + 官方权重（MSA-based）"),
        # ---- §7.2.2 autoregressive / generative (key de-noising control) ----
        _b(key="autoreg_ss_decoder", name="Autoregressive / generative structure decoder",
           route=ROUTE_AUTOREG, spec_section="§7.2.2", to_select=True, repo="",
           version_lock="待核验：spec §7.2.2 未点名具体模型",
           install_hint="待核验：须先按文献选定一个自回归/生成式结构模型并锁定版本",
           notes="**关键对照**：验证『去解码』收益（O(L) 次前向、结构幻觉）；"
                 "spec §7.2.2 只给出这一槽位、未命名模型 —— 未选定前不得填任何 commit/权重"),
        # ---- §7.2.2 3D end-to-end (supplemental) ----
        _b(key="rhofold", name="RhoFold", route=ROUTE_3D, spec_section="§7.2.2",
           repo="https://github.com/ml4bio/RhoFold", version_lock="RhoFold（锁定权重哈希）",
           install_hint="git clone + 官方权重", notes="3D 端到端（补充）"),
        _b(key="trrosetta", name="trRosettaRNA", route=ROUTE_3D, spec_section="§7.2.2",
           repo="https://github.com/wangchulab/trRosettaRNA", version_lock="trRosettaRNA（锁定权重哈希）",
           install_hint="git clone + 官方权重", notes="3D 端到端（补充）"),
        _b(key="alphafold3", name="AlphaFold3", route=ROUTE_3D, spec_section="§7.2.2",
           repo="https://github.com/google-deepmind/alphafold3",
           version_lock="AlphaFold3（权重需申请）", install_hint="需申请权重；补充对比",
           notes="3D 端到端（补充）"),
    ]


def get_baseline(key: str) -> Baseline:
    for b in registry():
        if b.key == key:
            return b
    raise KeyError(f"unknown baseline key: {key!r}")


def list_baselines(*, route: Optional[str] = None, closest_only: bool = False) -> List[Baseline]:
    out = registry()
    if route:
        out = [b for b in out if b.route == route]
    if closest_only:
        out = [b for b in out if b.closest_prior]
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _as_pairs(pairs) -> List[Tuple[int, int]]:
    norm = []
    for i, j in pairs:
        norm.append((i, j) if i < j else (j, i))
    return norm


def pair_metrics(pred_pairs, gt_pairs) -> Dict[str, float]:
    """Base-pair precision / recall / F1 / MCC and binary INF.

    INF here is the **base-pair-level** Interaction Network Fidelity, i.e. the
    geometric mean of PPV and sensitivity: ``TP / sqrt((TP+FP)(TP+FN))``.  The
    full 3D weighted INF of Parisien et al. (2009) is out of scope.
    """
    p = set(_as_pairs(pred_pairs))
    g = set(_as_pairs(gt_pairs))
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn))
    inf = tp / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "inf": inf,
            "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def reliability_diagram(probs, labels, mask=None, n_bins: int = 10) -> List[Dict[str, float]]:
    """Binned reliability data (mean predicted vs mean empirical per bin)."""
    import numpy as np
    from rnajepa.distill import pair_selector
    p = np.asarray(probs, dtype=float)
    a = np.asarray(labels, dtype=float)
    sel = pair_selector(mask, p.shape[0]).numpy()
    p = np.clip(p[sel], 0.0, 1.0)
    a = a[sel]
    if p.size == 0:
        return []
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        out.append({"bin": float(b), "bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins,
                    "count": float(m.sum()), "mean_pred": float(p[m].mean()),
                    "mean_empirical": float(a[m].mean())})
    return out


def calibration_metrics(probs, labels, mask=None, n_bins: int = 10) -> Dict[str, object]:
    """ECE + marginal calibration + Brier + NLL on masked strict-upper pairs.

    * ``ece`` -- standard hard-binned Expected Calibration Error (reused from
      ``rnajepa.distill`` so the number is identical to the training-time metric);
    * ``marginal_calibration`` -- calibration-in-the-large, ``|E[p] - E[y]|``
      (a *marginal*, not confidence-conditional, measure; distinct from ECE);
    * ``brier`` / ``nll`` -- proper scoring rules.
    """
    import numpy as np
    from rnajepa.distill import pair_selector
    p = np.asarray(probs, dtype=float)
    a = np.asarray(labels, dtype=float)
    sel = pair_selector(mask, p.shape[0]).numpy()
    p = np.clip(p[sel], 0.0, 1.0)
    a = a[sel]
    if p.size == 0:
        return {"ece": 0.0, "marginal_calibration": 0.0, "brier": 0.0, "nll": 0.0,
                "mean_pred": 0.0, "mean_empirical": 0.0, "reliability": []}
    eps = 1e-7
    p_safe = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - a) ** 2))
    nll = float(-np.mean(a * np.log(p_safe) + (1 - a) * np.log(1 - p_safe)))
    return {
        "ece": float(expected_calibration_error(probs, labels, mask, n_bins)),
        "marginal_calibration": float(abs(p.mean() - a.mean())),
        "brier": brier,
        "nll": nll,
        "mean_pred": float(p.mean()),
        "mean_empirical": float(a.mean()),
        "reliability": reliability_diagram(probs, labels, mask, n_bins),
    }


def evaluate_predictions(*, probs=None, labels=None, mask=None, pred_pairs=None,
                         gt_pairs=None, threshold: float = 0.5, n_bins: int = 10) -> Dict[str, object]:
    """Compute the full required metric set from supplied predictions.

    Either pass a probability matrix (``probs`` + ``labels`` [+ ``mask``]) or
    explicit pair sets (``pred_pairs`` + ``gt_pairs``).  This is real
    computation on caller-supplied predictions -- never a fabricated output.
    """
    metrics: Dict[str, object] = {}
    if probs is not None and labels is not None:
        metrics.update(calibration_metrics(probs, labels, mask, n_bins))
        if pred_pairs is None:
            import numpy as np
            from rnajepa.distill import pair_selector
            p = np.asarray(probs, dtype=float)
            sel = pair_selector(mask, p.shape[0]).numpy()
            idx = np.argwhere(sel & (p >= threshold))
            pred_pairs = [(int(i), int(j)) for i, j in idx]
        if gt_pairs is None:
            import numpy as np
            a = np.asarray(labels, dtype=float)
            idx = np.argwhere(a > 0.5)
            gt_pairs = [(int(i), int(j)) for i, j in idx if i < j]
    if pred_pairs is not None and gt_pairs is not None:
        metrics.update(pair_metrics(pred_pairs, gt_pairs))
    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        raise ValueError(f"required metrics missing from evaluation: {missing}")
    return metrics


# ---------------------------------------------------------------------------
# provenance gate (G5)
# ---------------------------------------------------------------------------
def reproducibility_report(baselines: Optional[Sequence[Baseline]] = None) -> Dict[str, object]:
    """Report which baselines are missing commit / weight-hash provenance."""
    items = list(baselines) if baselines is not None else registry()
    missing = []
    for b in items:
        gaps = []
        if not b.commit:
            gaps.append("commit")
        if not b.weight_hash:
            gaps.append("weight_hash")
        if not b.eval_script_version:
            gaps.append("eval_script_version")
        if gaps:
            missing.append({"key": b.key, "missing": gaps})
    return {"complete": not missing, "n_entries": len(items), "missing": missing}


def assert_reproducible(baselines: Optional[Sequence[Baseline]] = None) -> None:
    """Gate G5: every baseline must record commit + weight hash (+ script ver)."""
    report = reproducibility_report(baselines)
    if not report["complete"]:
        keys = ", ".join(f"{m['key']}({'+'.join(m['missing'])})" for m in report["missing"])  # type: ignore[union-attr]
        raise ReproducibilityGateError(
            f"G5 provenance gate FAILED: {len(report['missing'])}/{report['n_entries']} "  # type: ignore[arg-type]
            f"baselines missing provenance: {keys}. "
            "Commits / weight hashes must be filled from the official repos -- they are "
            "marked 待核验 and must never be invented.")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def plan_baseline(key: str) -> Dict[str, object]:
    """Dry-run plan for one baseline: what would be run, and with what lock."""
    b = get_baseline(key)
    return {
        "key": b.key,
        "name": b.name,
        "route": b.route,
        "spec_section": b.spec_section,
        "closest_prior": b.closest_prior,
        "needs_dp": b.needs_dp,
        "report_speedup_separately": b.report_speedup_separately,
        "version_lock": b.version_lock,
        "install_hint": b.install_hint,
        "emits": list(b.emits),
        "provenance": {"repo": b.repo, "commit": b.commit or HASH_UNKNOWN,
                       "weight_hash": b.weight_hash or HASH_UNKNOWN,
                       "eval_script_version": b.eval_script_version},
        "dry_run": True,
        "metrics": None,
    }


def run_baseline(key: str, *, dry_run: bool = False, probs=None, labels=None,
                 mask=None, pred_pairs=None, gt_pairs=None,
                 threshold: float = 0.5, n_bins: int = 10) -> Dict[str, object]:
    """Run one baseline.

    * ``dry_run=True``  -> returns the plan; no tool needed, no metrics.
    * no predictions    -> raises :class:`ToolUnavailableError` (tool not installed).
    * predictions given -> computes F1 / INF / ECE / marginal calibration / Brier
      / NLL for real, plus the provenance record.
    """
    b = get_baseline(key)
    if dry_run:
        return plan_baseline(key)
    if probs is None and pred_pairs is None:
        raise ToolUnavailableError(
            f"baseline {b.key!r} ({b.name}) is not installed in this environment. "
            f"Required version lock: {b.version_lock or HASH_UNKNOWN}. "
            f"Install hint: {b.install_hint or HASH_UNKNOWN}. "
            "Provide predictions (probs=... / pred_pairs=...) to evaluate a real run, "
            "or use --dry-run for the plan.",
            key=b.key, tool=b.name, version_lock=b.version_lock,
            install_hint=b.install_hint)
    metrics = evaluate_predictions(probs=probs, labels=labels, mask=mask,
                                   pred_pairs=pred_pairs, gt_pairs=gt_pairs,
                                   threshold=threshold, n_bins=n_bins)
    return {
        "key": b.key, "name": b.name, "route": b.route, "closest_prior": b.closest_prior,
        "emits": list(b.emits), "metrics": metrics, "dry_run": False,
        "provenance": {"repo": b.repo, "commit": b.commit or HASH_UNKNOWN,
                       "weight_hash": b.weight_hash or HASH_UNKNOWN,
                       "eval_script_version": b.eval_script_version},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def render_matrix() -> str:
    lines = ["# 基线复现矩阵（spec §7.2）", "",
             "> 头号对比对象（§7.2.1，closest prior work）以 **★** 标记；"
             "LinearPartition 的速度增益单独报告。", ""]
    lines.append("| route | key | 模型 | closest | needs_dp | commit | weight_hash | eval_script |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in registry():
        lines.append("| {r} | `{k}` | {n}{star} | {c} | {d} | {cm} | {wh} | {es} |".format(
            r=b.route, k=b.key, n=b.name, star=" ★" if b.closest_prior else "",
            c="★" if b.closest_prior else "—", d="yes" if b.needs_dp else "no",
            cm=b.commit or HASH_UNKNOWN, wh=b.weight_hash or HASH_UNKNOWN,
            es=b.eval_script_version))
    lines.append("")
    rep = reproducibility_report()
    lines.append(f"**G5 复现性门**：{'PASS' if rep['complete'] else 'FAIL'}"
                 f"（{len(rep['missing'])}/{rep['n_entries']} 条缺 provenance）")  # type: ignore[arg-type]
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RNA SS baseline reproduction matrix")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (no tool needed)")
    ap.add_argument("--key", default="", help="restrict to one baseline key")
    ap.add_argument("--matrix", action="store_true", help="print the whole baseline matrix")
    args = ap.parse_args(argv)

    if args.matrix or not args.key:
        sys.stdout.write(render_matrix())
    if args.key:
        if args.dry_run:
            import json
            print(json.dumps(plan_baseline(args.key), ensure_ascii=False, indent=2))
        else:
            try:
                run_baseline(args.key)
            except ToolUnavailableError as exc:
                print(f"[baselines] ToolUnavailableError: {exc}", file=sys.stderr)
                return 3
    # report the gate but do not crash the matrix print
    rep = reproducibility_report()
    print(f"[baselines] G5 gate: {'PASS' if rep['complete'] else 'FAIL'} "
          f"({len(rep['missing'])} missing)", file=sys.stderr)  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
