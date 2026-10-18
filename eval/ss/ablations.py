"""The 15 ablation switches required by spec §7.5 (gate SC3).

Each ablation is an :class:`Ablation` record with

* ``key`` -- stable identifier used in artefact paths and gate SC3;
* ``description`` -- the §7.5 wording;
* ``expected_direction`` -- the direction the spec expects (an ablation that
  removes a claimed mechanism is expected to hurt the metric that mechanism
  targets; where the spec states no direction the ablation is explicitly
  ``undirected`` and no winner is hard-coded);
* ``changes`` -- the concrete :class:`ExperimentConfig` fields the switch flips.

:meth:`Ablation.apply` returns a *new* config with the switch actually applied,
so "the switch changes the model/training configuration" is a testable property
rather than a comment.  Two ablations also have a real mechanism here:
:func:`independent_threshold_structure` (non-crossing constraint removed / DP
harness removed -> UFold-style independent-sigmoid + threshold decoding, which
can and does produce crossing pairs) and :func:`configure_head` (zeroing
``MLP_T`` / disabling temperature calibration on a live head).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from rnajepa.encoder import SIZE_PRESETS

# ---------------------------------------------------------------------------
# configuration under ablation
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """The switchable model/training configuration the ablations mutate.

    Defaults are the *proposed* (non-ablated) configuration.
    """

    noncrossing: bool = True              # DP-layer non-crossing constraint (C1)
    use_turner_residual: bool = True      # MLP_T Turner residual prior
    use_distillation: bool = True         # thermodynamic decision distillation
    distill_kind: str = "kl"              # "kl" | "l2"
    teacher_mode: str = "ensemble"        # "ensemble" | "single"
    use_rlcd: bool = True                 # RLCD-style calibration objective
    rlcd_reward: str = "brier"            # "brier" | "log"
    lambda_rlcd: float = 1.0
    beta: float = 1.0
    rlcd_sweep: bool = False              # lambda_RLCD / beta sweep enabled
    decision_order: str = "cotranscriptional"   # cotranscriptional | diagonal | random
    use_dp_harness: bool = True           # deterministic DP harness (System 2 decode)
    calibration: bool = True              # temperature-scaling calibration switch
    diff_dp: str = "implicit"             # "implicit" | "soft" (soft only for L <= 256)
    pair_slot_attention: str = "off"      # "off" | "topk" | "full_l4"
    encoder_size: str = "150M"            # 35M | 150M | 650M
    sparsification: str = "full"          # "full" | "banded"

    def validate(self) -> None:
        if self.distill_kind not in ("kl", "l2"):
            raise ValueError(f"distill_kind must be 'kl' or 'l2', got {self.distill_kind!r}")
        if self.teacher_mode not in ("ensemble", "single"):
            raise ValueError(f"teacher_mode must be 'ensemble' or 'single', got {self.teacher_mode!r}")
        if self.rlcd_reward not in ("brier", "log"):
            raise ValueError(f"rlcd_reward must be 'brier' or 'log', got {self.rlcd_reward!r}")
        if self.decision_order not in ("cotranscriptional", "diagonal", "random"):
            raise ValueError(f"unknown decision_order {self.decision_order!r}")
        if self.diff_dp not in ("implicit", "soft"):
            raise ValueError(f"diff_dp must be 'implicit' or 'soft', got {self.diff_dp!r}")
        if self.pair_slot_attention not in ("off", "topk", "full_l4"):
            raise ValueError(f"unknown pair_slot_attention {self.pair_slot_attention!r}")
        if self.encoder_size not in SIZE_PRESETS:
            raise ValueError(f"unknown encoder_size {self.encoder_size!r}")
        if self.sparsification not in ("full", "banded"):
            raise ValueError(f"sparsification must be 'full' or 'banded', got {self.sparsification!r}")


# ---------------------------------------------------------------------------
# the ablation record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ablation:
    key: str
    description: str
    expected_direction: str
    expected_sign: Dict[str, str]      # metric -> "up" | "down" | "undirected"
    changes: Dict[str, object]
    note: str = ""

    def apply(self, cfg: Optional[ExperimentConfig] = None, on: bool = True) -> ExperimentConfig:
        """Return a config with this ablation switched ``on`` (or off/unchanged)."""
        base = cfg or ExperimentConfig()
        if not on:
            return base
        out = replace(base, **self.changes)
        out.validate()
        return out

    def as_dict(self) -> Dict[str, object]:
        return {"key": self.key, "description": self.description,
                "expected_direction": self.expected_direction,
                "expected_sign": dict(self.expected_sign), "changes": dict(self.changes),
                "note": self.note}


# ---------------------------------------------------------------------------
# the 15 ablations (spec §7.5, in spec order)
# ---------------------------------------------------------------------------
ABLATIONS: Tuple[Ablation, ...] = (
    Ablation(
        key="no_noncrossing_constraint",
        description="去掉非交叉约束（改独立 sigmoid + 阈值，即 UFold 式后处理）",
        expected_direction="legalisation cost (H1): illegal-structure rate jumps from 0",
        expected_sign={"illegal_structure_rate": "up", "f1": "down"},
        changes={"noncrossing": False},
        note="independent sigmoid + threshold decoding (see independent_threshold_structure)",
    ),
    Ablation(
        key="turner_residual_zeroed",
        description="Turner 残差 MLP_T 置零（精确回退纯物理模型）",
        expected_direction="net contribution of the physical prior (undirected; cold-start control)",
        expected_sign={"f1": "undirected"},
        changes={"use_turner_residual": False},
        note="zeroing MLP_T yields exactly 'Nussinov + Turner stacking', NOT ViennaRNA (spec §0.7)",
    ),
    Ablation(
        key="no_distillation",
        description="去掉热力学决策蒸馏（仅硬标签监督）",
        expected_direction="distillation gain (H3), especially on bpRNA-new",
        expected_sign={"cross_family_f1": "down"},
        changes={"use_distillation": False},
    ),
    Ablation(
        key="distill_kind_l2",
        description="蒸馏目标形式：KL vs L2",
        expected_direction="objective-function choice (undirected)",
        expected_sign={"f1": "undirected"},
        changes={"distill_kind": "l2"},
    ),
    Ablation(
        key="single_teacher",
        description="教师集成 vs 单一教师",
        expected_direction="whether the ensemble teacher adds value (undirected)",
        expected_sign={"f1": "undirected"},
        changes={"teacher_mode": "single"},
    ),
    Ablation(
        key="no_rlcd",
        description="去掉 RLCD 校准目标（仅似然 + 蒸馏）",
        expected_direction="calibration gain (H6): ECE rises, bpRNA-new F1 drops",
        expected_sign={"ece": "up", "cross_family_f1": "down"},
        changes={"use_rlcd": False},
    ),
    Ablation(
        key="rlcd_reward_log",
        description="RLCD 奖励形式：Brier vs 对数评分",
        expected_direction="proper-scoring-rule choice (undirected)",
        expected_sign={"ece": "undirected"},
        changes={"rlcd_reward": "log"},
    ),
    Ablation(
        key="rlcd_strength_sweep",
        description="RLCD 强度 lambda_RLCD / beta 扫描",
        expected_direction="calibration-accuracy trade-off curve (undirected)",
        expected_sign={"ece": "undirected", "f1": "undirected"},
        changes={"rlcd_sweep": True, "lambda_rlcd": 0.1, "beta": 0.5},
    ),
    Ablation(
        key="decision_order",
        description="决策顺序：共转录 vs 对角 vs 随机",
        expected_direction="limited effect (H4, deliberately undirected; report as negative if null)",
        expected_sign={"f1": "undirected"},
        changes={"decision_order": "diagonal"},
        note="see ss.ordering; the ordering must never be described as a contribution",
    ),
    Ablation(
        key="no_dp_harness",
        description="去掉 DP harness（直接阈值化得分矩阵）",
        expected_direction="legality collapse: illegal-structure rate jumps above 0",
        expected_sign={"illegal_structure_rate": "up", "f1": "down"},
        changes={"use_dp_harness": False},
        note="direct thresholding of the score matrix, no non-crossing enforcement",
    ),
    Ablation(
        key="calibration_off",
        description="校准开关（温度缩放 on/off）",
        expected_direction="calibration gain (H5): ECE rises when off",
        expected_sign={"ece": "up"},
        changes={"calibration": False},
    ),
    Ablation(
        key="soft_nussinov",
        description="可微 DP：隐式微分 vs soft-Nussinov（L <= 256）",
        expected_direction="training stability vs memory cost (undirected; soft only for L <= 256)",
        expected_sign={"f1": "undirected"},
        changes={"diff_dp": "soft"},
        note="soft-Nussinov stores the O(L^3) DP graph; only valid for L <= 256",
    ),
    Ablation(
        key="pair_slot_attention",
        description="配对槽结构化注意力（top-k，O(L^4) 备选）",
        expected_direction="explicit pair coupling vs DP-layer built-in constraint (undirected)",
        expected_sign={"f1": "undirected"},
        changes={"pair_slot_attention": "topk"},
        note="'full_l4' is the O(L^4) alternative, not run at scale",
    ),
    Ablation(
        key="encoder_size",
        description="编码器规模 35M / 150M / 650M",
        expected_direction="speed-accuracy Pareto (undirected)",
        expected_sign={"f1": "undirected", "latency": "down"},
        changes={"encoder_size": "35M"},
    ),
    Ablation(
        key="sparsification_banded",
        description="稀疏化：banded vs full",
        expected_direction="long-chain feasibility; banding sacrifices long-range pairs",
        expected_sign={"long_range_f1": "down", "latency": "down"},
        changes={"sparsification": "banded"},
        note="banding deletes every pair with span > band, unlike the cascade (spec §5.0.2(d))",
    ),
)

ABLATION_KEYS: Tuple[str, ...] = tuple(a.key for a in ABLATIONS)

#: Default grid for the ``lambda_RLCD`` / ``beta`` sweep ablation.
SWEEP_GRID: Dict[str, Tuple[float, ...]] = {
    "lambda_rlcd": (0.0, 0.1, 0.5, 1.0, 2.0),
    "beta": (0.0, 0.25, 0.5, 1.0, 2.0),
}


def validate_registry() -> None:
    """Assert the registry is exactly the 15 spec §7.5 ablations, all distinct."""
    if len(ABLATIONS) != 15:
        raise AssertionError(f"spec §7.5 requires 15 ablations, registry has {len(ABLATIONS)}")
    if len(set(ABLATION_KEYS)) != 15:
        raise AssertionError("ablation keys must be unique")
    valid_fields = {f.name for f in fields(ExperimentConfig)}
    for abl in ABLATIONS:
        unknown = set(abl.changes) - valid_fields
        if unknown:
            raise AssertionError(f"ablation {abl.key!r} changes unknown fields {sorted(unknown)}")


def get_ablation(key: str) -> Ablation:
    for abl in ABLATIONS:
        if abl.key == key:
            return abl
    raise KeyError(f"unknown ablation {key!r}; known: {list(ABLATION_KEYS)}")


def list_ablations() -> List[Dict[str, object]]:
    return [a.as_dict() for a in ABLATIONS]


def ablation_table() -> List[Dict[str, object]]:
    """Table with the *effective* config delta for each ablation (for reporting)."""
    base = ExperimentConfig()
    rows = []
    for abl in ABLATIONS:
        applied = abl.apply(base, on=True)
        deltas = {k: getattr(applied, k) for k in abl.changes}
        rows.append({"key": abl.key, "description": abl.description,
                     "expected_direction": abl.expected_direction, "delta": deltas})
    return rows


# ---------------------------------------------------------------------------
# real mechanisms behind two of the ablations
# ---------------------------------------------------------------------------
def independent_threshold_structure(scores, mask=None, threshold: float = 0.0
                                    ) -> List[Tuple[int, int]]:
    """UFold-style decoding: threshold the per-pair score matrix **independently**.

    This is the mechanism of the "remove the non-crossing constraint" and "remove
    the DP harness" ablations.  Every pair ``(i, j)`` with ``sigmoid(score) > 0.5``
    (equivalently ``score > threshold``) is emitted with no non-crossing
    enforcement, so crossing pairs are possible -- which is exactly why the
    illegal-structure rate is expected to rise above 0.
    """
    s = np.asarray(scores, dtype=np.float64)
    L = s.shape[0]
    if mask is None:
        sel = np.triu(np.ones((L, L), dtype=bool), k=1)
    else:
        sel = np.triu(np.asarray(mask, dtype=bool), k=1)
    idx = np.argwhere(sel & (s > threshold))
    return [(int(i), int(j)) for i, j in idx]


def configure_head(cfg: ExperimentConfig, head) -> object:
    """Apply the head-level switches to a live decision head (in place).

    * ``turner_residual_zeroed`` -> ``head.zero_residual()``;
    * ``calibration_off`` -> temperature set to 1 for every bucket.

    Returns the head.  Only touches attributes the head is documented to expose.
    """
    if not cfg.use_turner_residual and hasattr(head, "zero_residual"):
        head.zero_residual()
    if not cfg.calibration and hasattr(head, "calibration") and \
            hasattr(head.calibration, "log_temperature"):
        import torch
        with torch.no_grad():
            head.calibration.log_temperature.zero_()
    return head


def encoder_size_spec(size: str) -> Dict[str, int]:
    """The ``(n_layer, d_model, n_head, d_ff)`` preset for an encoder size."""
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown encoder size {size!r}; expected {sorted(SIZE_PRESETS)}")
    return dict(SIZE_PRESETS[size])


__all__ = [
    "ABLATIONS",
    "ABLATION_KEYS",
    "SWEEP_GRID",
    "Ablation",
    "ExperimentConfig",
    "ablation_table",
    "configure_head",
    "encoder_size_spec",
    "get_ablation",
    "independent_threshold_structure",
    "list_ablations",
    "validate_registry",
]
