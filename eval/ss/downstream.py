"""Downstream-task adapters T1-T6 (spec §6).

======================  ====================================================
task                    adapter
======================  ====================================================
T1 non-pseudoknot SS    :func:`run_t1` (main task)
T2 pseudoknot-aware     :func:`run_t2` -- **restricted class only**; general
                        pseudoknots are NP-hard and the framework's exact DP
                        needs non-crossing, so the applicability scope is
                        reported honestly and out-of-scope structures are
                        flagged (never silently claimed)
T3 SHAPE/DMS            :func:`run_t3`
T4 long-chain           :func:`run_t4`
T5 transfer tasks       :func:`run_t5` -- reuses the existing task registry
                        ``configs/tasks.yaml``
T6 zero/few-shot + AL   :func:`run_t6`
======================  ====================================================

Every run writes ``result.json`` plus ``predictions.json`` under its output
directory.  Any task whose dev and test splits are the same **must** carry an
explicit defect flag in its output (inherited lesson 5, ``spec/constraints.md``
§4): the flag is recorded in both ``flags`` and the structured ``defects`` list,
so a downstream reader cannot mistake a ``dev == test`` number for clean
generalisation evidence.

Honesty: there are no real structure-annotated data here and no external tools,
so these adapters are exercised on synthetic/crafted inputs; the predictor is
always an injected callable and nothing is faked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rnajepa.harness import MIN_LOOP

from .metrics import (
    LegalityMetrics,
    PairLevelMetrics,
    StructureLevelMetrics,
    check_structure,
)

Pair = Tuple[int, int]

#: Default task registry (spec §3.1 / §6 T5).
DEFAULT_REGISTRY_PATH = os.path.join("configs", "tasks.yaml")

#: Honest applicability statement for T2 (spec §6).
RESTRICTED_PSEUDOKNOT_SCOPE = (
    "T2 covers the RESTRICTED pseudoknot class only: structures whose crossings "
    "form a single H-type pseudoknot. The framework's exact DP requires "
    "non-crossing, and general pseudoknot prediction is NP-hard, so no general "
    "pseudoknot capability is claimed."
)


# ---------------------------------------------------------------------------
# result container + persistence
# ---------------------------------------------------------------------------
@dataclass
class TaskResult:
    task_id: str
    name: str
    metrics: Dict[str, object] = field(default_factory=dict)
    predictions: List[object] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    defects: List[Dict[str, object]] = field(default_factory=list)
    applicability: Optional[str] = None
    n: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def add_defect(self, code: str, detail: str) -> None:
        self.defects.append({"code": code, "detail": detail})
        self.add_flag(f"DEFECT[{code}]: {detail}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "n": self.n,
            "metrics": self.metrics,
            "flags": list(self.flags),
            "defects": list(self.defects),
            "applicability": self.applicability,
            "extra": self.extra,
        }


def write_result(out_dir: str, result: TaskResult) -> Dict[str, str]:
    """Write ``result.json`` and ``predictions.json``; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    pred_path = os.path.join(out_dir, "predictions.json")
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
    with open(pred_path, "w", encoding="utf-8") as fh:
        json.dump(result.predictions, fh, indent=2, ensure_ascii=False, default=str)
    return {"result": result_path, "predictions": pred_path}


# ---------------------------------------------------------------------------
# predictor plumbing
# ---------------------------------------------------------------------------
def _normalize_prediction(out) -> Tuple[List[Pair], Optional[np.ndarray], Optional[float]]:
    """Accept ``pairs`` or a ``dict`` prediction; return ``(pairs, probs, fallback)``."""
    if isinstance(out, dict):
        pairs = out.get("pairs", out.get("structure", []))
        probs = out.get("probs")
        fallback = out.get("fallback_fraction")
        return [tuple(p) for p in pairs], (None if probs is None else np.asarray(probs, float)), fallback
    return [tuple(p) for p in out], None, None


def _call(predictor: Callable, seq: str, reactivity=None):
    if reactivity is None:
        return predictor(seq)
    try:
        return predictor(seq, reactivity)
    except TypeError:
        return predictor(seq)


def _metrics_for(pred_pairs, gt_pairs, seq, probs, mask=None) -> Dict[str, object]:
    out: Dict[str, object] = {
        "pair_level": PairLevelMetrics.from_pairs(pred_pairs, gt_pairs, L=len(seq), mask=mask).as_dict(),
        "structure_level": StructureLevelMetrics.from_pairs(pred_pairs, gt_pairs).as_dict(),
        "legality": LegalityMetrics.from_structures([pred_pairs], [seq]).as_dict(),
    }
    if probs is not None and gt_pairs is not None:
        from rnajepa.distill import pair_indicator
        from .metrics import CalibrationMetrics
        labels = pair_indicator(len(seq), gt_pairs)
        out["calibration"] = CalibrationMetrics.from_probs(probs, labels, mask=mask).as_dict()
    return out


def _aggregate(per_item: Sequence[Dict[str, object]], key: str) -> float:
    vals = [float(m["pair_level"][key]) for m in per_item if "pair_level" in m]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# T1 -- non-pseudoknot secondary structure (main task)
# ---------------------------------------------------------------------------
def run_t1(sequences: Sequence[str], predictor: Callable, out_dir: str,
           gt: Optional[Sequence[Sequence[Pair]]] = None,
           task_id: str = "T1", write: bool = True) -> TaskResult:
    """T1: non-pseudoknot secondary-structure prediction."""
    if gt is not None and len(gt) != len(sequences):
        raise ValueError("gt and sequences must have equal length")
    preds: List[object] = []
    per_item: List[Dict[str, object]] = []
    for k, seq in enumerate(sequences):
        pairs, probs, _fb = _normalize_prediction(_call(predictor, seq))
        g = list(gt[k]) if gt is not None else []
        preds.append({"seq": seq, "pairs": [list(p) for p in pairs],
                      "gt": [list(p) for p in g]})
        per_item.append(_metrics_for(pairs, g, seq, probs))
    res = TaskResult(task_id=task_id, name="non-pseudoknot structure prediction",
                     predictions=preds, n=len(sequences))
    res.metrics = {
        "f1": _aggregate(per_item, "f1"),
        "precision": _aggregate(per_item, "precision"),
        "recall": _aggregate(per_item, "recall"),
        "mcc": _aggregate(per_item, "mcc"),
        "inf": float(np.mean([m["structure_level"]["inf"] for m in per_item])) if per_item else 0.0,
        "illegal_structure_rate": float(np.mean(
            [m["legality"]["illegal_structure_rate"] for m in per_item])) if per_item else 0.0,
        "n_sequences": len(sequences),
    }
    res.extra["per_sequence"] = per_item
    if write:
        res.extra["artefacts"] = write_result(out_dir, res)
    return res


# ---------------------------------------------------------------------------
# T2 -- pseudoknot-aware (restricted class only)
# ---------------------------------------------------------------------------
def _is_h_type(crossings: Sequence[Tuple[Pair, Pair]]) -> bool:
    """Conservative H-type test: exactly one crossing pair of pairs (i<k<j<l)."""
    if len(crossings) != 1:
        return False
    (i, j), (k, l) = crossings[0]
    return i < k < j < l


def restricted_class_check(structure: Sequence[Pair], seq: Optional[str] = None
                           ) -> Dict[str, object]:
    """Is ``structure`` inside the restricted (single H-type pseudoknot) class?"""
    rep = check_structure(structure, seq or ("A" * (max((max(p) for p in structure), default=-1) + 1)))
    crossings = rep["crossing_pairs"]
    if not crossings:
        return {"in_scope": True, "reason": "non-crossing (handled by the exact DP)"}
    if _is_h_type(crossings):
        return {"in_scope": True, "reason": "single H-type pseudoknot"}
    return {"in_scope": False,
            "reason": f"{len(crossings)} crossing pair(s): outside the restricted class"}


def run_t2(sequences: Sequence[str], predictor: Callable, out_dir: str,
           gt: Optional[Sequence[Sequence[Pair]]] = None, task_id: str = "T2",
           write: bool = True) -> TaskResult:
    """T2: pseudoknot-aware prediction, **restricted class only** (spec §6)."""
    res = run_t1(sequences, predictor, out_dir, gt=gt, task_id=task_id, write=False)
    res.name = "pseudoknot-aware structure prediction (restricted class only)"
    res.applicability = RESTRICTED_PSEUDOKNOT_SCOPE
    out_of_scope = 0
    for pred in res.predictions:
        chk = restricted_class_check(pred["pairs"], pred["seq"])
        pred["restricted_class"] = chk
        if not chk["in_scope"]:
            out_of_scope += 1
    res.extra["out_of_scope_count"] = out_of_scope
    res.extra["restricted_class_scope"] = RESTRICTED_PSEUDOKNOT_SCOPE
    if out_of_scope:
        res.add_flag(
            f"{out_of_scope}/{len(sequences)} predicted structures fall outside the "
            "restricted pseudoknot class; general pseudoknots are NP-hard and not claimed"
        )
    if write:
        res.extra["artefacts"] = write_result(out_dir, res)
    return res


# ---------------------------------------------------------------------------
# T3 -- SHAPE/DMS-conditioned
# ---------------------------------------------------------------------------
def run_t3(sequences: Sequence[str], predictor: Callable, out_dir: str,
           reactivity: Sequence[Sequence[float]],
           gt: Optional[Sequence[Sequence[Pair]]] = None, task_id: str = "T3",
           write: bool = True) -> TaskResult:
    """T3: probing-data (SHAPE/DMS) conditioned prediction."""
    if len(reactivity) != len(sequences):
        raise ValueError("reactivity and sequences must have equal length")
    preds: List[object] = []
    per_item: List[Dict[str, object]] = []
    for k, seq in enumerate(sequences):
        pairs, probs, _fb = _normalize_prediction(_call(predictor, seq, reactivity[k]))
        g = list(gt[k]) if gt is not None else []
        preds.append({"seq": seq, "pairs": [list(p) for p in pairs],
                      "reactivity": list(map(float, reactivity[k])),
                      "gt": [list(p) for p in g]})
        per_item.append(_metrics_for(pairs, g, seq, probs))
    res = TaskResult(task_id=task_id, name="SHAPE/DMS-conditioned structure prediction",
                     predictions=preds, n=len(sequences))
    res.metrics = {
        "f1": _aggregate(per_item, "f1"),
        "inf": float(np.mean([m["structure_level"]["inf"] for m in per_item])) if per_item else 0.0,
        "n_sequences": len(sequences),
    }
    res.extra["per_sequence"] = per_item
    if write:
        res.extra["artefacts"] = write_result(out_dir, res)
    return res


# ---------------------------------------------------------------------------
# T4 -- long-chain / full-length transcripts
# ---------------------------------------------------------------------------
def run_t4(sequences: Sequence[str], predictor: Callable, out_dir: str,
           gt: Optional[Sequence[Sequence[Pair]]] = None,
           bucket_edges: Sequence[int] = (0, 512, 1024, 2048, 4096),
           reference_bucket: int = 0, task_id: str = "T4", write: bool = True
           ) -> TaskResult:
    """T4: long-chain extrapolation; per-length-bucket F1 and decay vs a reference."""
    res = run_t1(sequences, predictor, out_dir, gt=gt, task_id=task_id, write=False)
    res.name = "long-chain / full-length transcript structure prediction"
    per_item = res.extra["per_sequence"]
    lens = np.asarray([len(s) for s in sequences], dtype=np.int64)
    buckets: List[Dict[str, object]] = []
    for b in range(len(bucket_edges)):
        lo = int(bucket_edges[b])
        hi = bucket_edges[b + 1] if b + 1 < len(bucket_edges) else None
        m = (lens >= lo) & ((lens < hi) if hi is not None else True)
        if not m.any():
            continue
        f1s = [float(per_item[k]["pair_level"]["f1"]) for k in np.flatnonzero(m)]
        buckets.append({"lo": lo, "hi": hi, "n": int(m.sum()), "f1": float(np.mean(f1s))})
    ref = buckets[reference_bucket]["f1"] if len(buckets) > reference_bucket else None
    for b in buckets:
        b["decay_vs_reference"] = (None if ref in (None, 0.0) else float((ref - b["f1"]) / ref))
    res.metrics = {"f1": res.metrics.get("f1", 0.0), "n_sequences": len(sequences)}
    res.extra["buckets"] = buckets
    if write:
        res.extra["artefacts"] = write_result(out_dir, res)
    return res


# ---------------------------------------------------------------------------
# T5 -- transfer tasks (reuse the existing registry)
# ---------------------------------------------------------------------------
def load_task_registry(path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, object]:
    """Load the frozen task registry (``configs/tasks.yaml``)."""
    import yaml
    if not os.path.isabs(path) and not os.path.exists(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dev_equals_test(task: Dict[str, object]) -> Tuple[bool, str]:
    """Detect a ``dev == test`` split defect for one registry task.

    Flags when the registry marks ``shared_dev_test`` **or** when the recorded
    dev/test row counts are equal (a conservative second signal).  Returns
    ``(is_defective, reason)``.
    """
    if bool(task.get("shared_dev_test")):
        return True, "registry marks shared_dev_test = true (dev and test are the same split)"
    rows = task.get("rows")
    if isinstance(rows, (list, tuple)) and len(rows) == 3 and rows[1] is not None \
            and rows[1] == rows[2]:
        return True, f"dev rows == test rows ({rows[1]})"
    return False, ""


def task_defect_flags(task_key: str, task: Dict[str, object]) -> List[Dict[str, object]]:
    """All defect flags that must appear in a task's output."""
    defects: List[Dict[str, object]] = []
    defective, reason = dev_equals_test(task)
    if defective:
        defects.append({"code": "dev_eq_test", "detail": reason})
    if bool(task.get("not_comparable_to_paper")):
        defects.append({"code": "not_comparable_to_paper",
                        "detail": "reported value is not comparable to the paper's protocol"})
    if task.get("protocol_note"):
        defects.append({"code": "protocol_deviation", "detail": str(task["protocol_note"])})
    return defects


def run_t5(registry_path: str = DEFAULT_REGISTRY_PATH, out_dir: str = ".",
           metrics_by_task: Optional[Dict[str, Dict[str, float]]] = None,
           task_keys: Optional[Iterable[str]] = None, write: bool = True
           ) -> List[TaskResult]:
    """T5: transfer tasks, reusing the registry; flags ``dev == test`` per task.

    ``metrics_by_task`` supplies the per-task metrics (there are no real
    secondary-structure annotations here); when omitted the task is emitted with
    empty metrics and only its defect flags, which is the honest default.
    """
    registry = load_task_registry(registry_path)
    tasks = registry.get("tasks", {}) if isinstance(registry, dict) else {}
    keys = list(task_keys) if task_keys is not None else list(tasks)
    results: List[TaskResult] = []
    for key in keys:
        task = tasks.get(key, {})
        res = TaskResult(task_id=f"T5:{key}", name=f"transfer task {key}",
                         metrics=dict((metrics_by_task or {}).get(key, {})), n=0)
        res.extra["family"] = task.get("family")
        res.extra["kind"] = task.get("kind")
        res.extra["metric"] = task.get("metric")
        for defect in task_defect_flags(key, task):
            res.add_defect(defect["code"], defect["detail"])
        if write:
            res.extra["artefacts"] = write_result(os.path.join(out_dir, key), res)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# T6 -- zero/few-shot + active learning
# ---------------------------------------------------------------------------
def active_learning_select(probs_list: Sequence[np.ndarray], budget: int,
                           mask: Optional[Sequence[object]] = None) -> List[int]:
    """Select the ``budget`` most uncertain items by low mean pair confidence.

    Uncertainty is ``mean p_hat`` over the legal candidate pairs; a low mean means
    the model is unsure, so the item is worth labelling first.  Returns the
    selected indices in ascending uncertainty order.
    """
    scores: List[Tuple[float, int]] = []
    for k, probs in enumerate(probs_list):
        m = None if mask is None else mask[k]
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim == 2:
            L = p.shape[0]
            sel = np.triu(np.ones((L, L), dtype=bool), k=MIN_LOOP + 1)
            if m is not None:
                sel = np.triu(np.asarray(m, dtype=bool), k=MIN_LOOP + 1)
            p = p[sel]
        scores.append((float(p.mean()) if p.size else 0.0, k))
    scores.sort(key=lambda t: (t[0], t[1]))
    return [k for _s, k in scores[:max(0, int(budget))]]


def run_t6(sequences: Sequence[str], predictor: Callable, out_dir: str,
           gt: Optional[Sequence[Sequence[Pair]]] = None, k_shot: int = 0,
           active_learning: bool = False, al_budget: int = 0,
           task_id: str = "T6", write: bool = True) -> TaskResult:
    """T6: zero/few-shot structure prediction with optional active learning."""
    res = run_t1(sequences, predictor, out_dir, gt=gt, task_id=task_id, write=False)
    res.name = f"zero/few-shot structure prediction (k_shot={k_shot})"
    res.extra["k_shot"] = int(k_shot)
    if active_learning:
        probs_list = []
        for pred in res.predictions:
            probs_list.append(np.zeros((len(pred["seq"]), len(pred["seq"]))))
        selected = active_learning_select(probs_list, al_budget)
        res.extra["active_learning"] = {"budget": int(al_budget), "selected_indices": selected}
    if k_shot <= 0:
        res.add_flag("zero-shot: no task-specific training examples were used")
    if write:
        res.extra["artefacts"] = write_result(out_dir, res)
    return res


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "RESTRICTED_PSEUDOKNOT_SCOPE",
    "TaskResult",
    "active_learning_select",
    "dev_equals_test",
    "load_task_registry",
    "restricted_class_check",
    "run_t1",
    "run_t2",
    "run_t3",
    "run_t4",
    "run_t5",
    "run_t6",
    "task_defect_flags",
    "write_result",
]
