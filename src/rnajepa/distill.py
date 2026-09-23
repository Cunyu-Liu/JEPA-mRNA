"""Thermodynamic decision distillation (spec §5.7) -- System-1 / System-2 split.

Role in the architecture
------------------------
This module implements the *pretraining* objective that lets a single-forward-pass
decision head (System-1) match the calibrated pair probabilities produced by a
slow partition-function teacher (System-2).  The exact marginals ``p_hat`` are
used **as a training-time teacher**, never as an inference-time dependency
(spec §5.0.1); that is exactly what allows the head to skip the ``O(L^3)``
partition function at inference while still emitting calibrated probabilities.

Honesty / provenance (read before citing anything here)
-------------------------------------------------------
* The Gibbs / log-linear CRF framework used to obtain the teacher marginals is
  **not our contribution**.  It is the standard machinery of CONTRAfold
  (Do, Woods & Batzoglou, *Bioinformatics* 2006) and the structured-prediction /
  CRF literature.  See ``rnajepa.harness`` for the same statement.
* Jev's **RLCD algorithm is not publicly disclosed** and Jev has **no
  peer-reviewed paper** (its numbers are vendor self-reported, spec §0.3).  The
  calibration objective lives in ``rnajepa.rlcd`` and is an *RLCD-inspired,
  self-designed* objective; nothing here reproduces or matches Jev's RLCD.
* The real thermodynamic teacher (ViennaRNA / RNAstructure / LinearPartition) is
  **not installed and cannot be downloaded in this environment**.  We therefore
  ship a clean interface plus an explicit :class:`ThermodynamicTeacher` stub that
  raises an actionable error, and a :class:`MockTeacher` used only in tests.

Why the saturation diagnostic exists
------------------------------------
A previous line of this project (the JEPA region-mean summary target) saturated
almost immediately: cosine similarity reached 0.993 within 60 optimizer steps
and the loss collapsed to 0.02-0.05 (see ``records/A3_latent_target_comparison.md``).
A distillation objective is *claimed* not to have that failure mode (spec §5.7),
so the claim has to be measured rather than asserted: :class:`SaturationDiagnostic`
tracks the cosine similarity between the student and teacher probability matrices
and flags early saturation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from rnajepa.harness import MIN_LOOP, inside_outside, valid_pair_mask

__all__ = [
    "distillation_loss",
    "assemble_teacher",
    "exact_marginal_teacher",
    "empirical_pair_frequencies",
    "teacher_scores_from_structures",
    "pair_indicator",
    "cosine_similarity",
    "SaturationDiagnostic",
    "expected_calibration_error",
    "calibration_cost",
    "ThermodynamicTeacher",
    "ThermodynamicUnavailableError",
    "MockTeacher",
    "generate_teacher_labels",
    "verify_teacher_labels",
    "load_teacher_shard",
    "sha256_file",
    "sha256_hex",
]

_EPS = 1e-7


# ---------------------------------------------------------------------------
# small tensor helpers (shared with rnajepa.rlcd)
# ---------------------------------------------------------------------------
def as_tensor(x, dtype=torch.float64) -> torch.Tensor:
    """Return ``x`` as a torch tensor of ``dtype`` (no-op for tensors)."""
    if torch.is_tensor(x):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def as_bool_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.bool()
    return torch.as_tensor(np.asarray(x), dtype=torch.bool)


def pair_selector(mask, L: int) -> torch.Tensor:
    """Boolean ``L x L`` selector for the strict upper-triangle masked pairs.

    ``scores`` / probabilities are only ever read on the strict upper triangle
    (``i < j``), matching the frozen ``rnajepa.harness`` convention.  ``mask`` may
    be ``None`` (all ``i < j`` pairs) or an ``L x L`` boolean array/tensor.
    """
    if mask is None:
        return torch.ones(L, L, dtype=torch.bool).triu(diagonal=1)
    m = as_bool_tensor(mask)
    if m.shape[0] != L:
        raise ValueError(f"mask shape {tuple(m.shape)} does not match L={L}")
    return m.triu(diagonal=1)


def select_pairs(p_hat, labels, mask):
    """Flatten ``p_hat`` and ``labels`` onto the strict upper-triangle masked pairs."""
    p = as_tensor(p_hat)
    a = as_tensor(labels)
    sel = pair_selector(mask, p.shape[0])
    return p[sel], a[sel]


def pair_indicator(L: int, pairs, dtype=np.float64) -> np.ndarray:
    """Binary ``L x L`` indicator of ``pairs`` (upper triangle only)."""
    y = np.zeros((L, L), dtype=dtype)
    for i, j in pairs:
        if i < j:
            y[i, j] = 1.0
        else:
            y[j, i] = 1.0
    return y


def _xlogy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x * log(y)`` with the convention ``0 * log(0) = 0`` (differentiable in x)."""
    y_safe = torch.where(y > 0, y, torch.ones_like(y))
    return torch.where(x > 0, x * torch.log(y_safe), torch.zeros_like(x))


# ---------------------------------------------------------------------------
# distillation loss (spec §5.7)
# ---------------------------------------------------------------------------
def distillation_loss(student_scores_or_probs, teacher_probs, mask,
                      kind: str = "kl", *, student_from_logits: bool = False,
                      reduction: str = "mean", eps: float = _EPS):
    """``L_distill`` between a student probability matrix and a teacher target.

    ``L_distill = mean_{i<j, mask} D(p^teacher_ij || p^student_ij)`` where ``D`` is
    the Bernoulli KL divergence (``kind="kl"``) or the squared error
    (``kind="l2"``); spec §5.7 lists both as an ablation.

    Parameters
    ----------
    student_scores_or_probs:
        ``L x L`` student matrix.  Interpreted as probabilities unless
        ``student_from_logits=True`` (then ``sigmoid`` is applied first).
    teacher_probs:
        ``L x L`` teacher probability matrix (e.g. exact marginals or a
        thermodynamic teacher's pair probabilities).
    mask:
        ``L x L`` boolean candidate-pair mask (``None`` means all ``i < j``).
    kind:
        ``"kl"`` or ``"l2"``.

    Numerical safety
    ----------------
    Teacher probabilities are allowed to be exactly ``0`` or ``1``: the KL terms
    are written so that the corresponding ``0 * log`` contributions vanish and the
    student probabilities are clamped (only where they are actually read) before
    the logarithm.  The loss is therefore finite and has a finite gradient at the
    boundaries, and it is exactly ``0`` when ``student == teacher``.

    Returns a 0-dim ``torch`` tensor (numpy inputs are converted), so gradients
    flow when the student matrix is a differentiable tensor.
    """
    if kind not in ("kl", "l2"):
        raise ValueError(f"kind must be 'kl' or 'l2', got {kind!r}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    p_s = as_tensor(student_scores_or_probs)
    if student_from_logits:
        p_s = torch.sigmoid(p_s)
    p_t = as_tensor(teacher_probs)
    if p_s.shape != p_t.shape:
        raise ValueError(f"student shape {tuple(p_s.shape)} != teacher shape {tuple(p_t.shape)}")

    s, t = select_pairs(p_s, p_t, mask)
    if s.numel() == 0:
        return p_s.sum() * 0.0

    if kind == "l2":
        per_pair = (s - t) ** 2
    else:
        s_c = torch.clamp(s, 0.0, 1.0)
        log_ps = torch.log(torch.clamp(s_c, min=eps))
        log_1mps = torch.log(torch.clamp(1.0 - s_c, min=eps))
        t_c = torch.clamp(t, 0.0, 1.0)
        term_pos = _xlogy(t_c, t_c) - t_c * log_ps
        term_neg = _xlogy(1.0 - t_c, 1.0 - t_c) - (1.0 - t_c) * log_1mps
        per_pair = term_pos + term_neg

    if reduction == "sum":
        return per_pair.sum()
    return per_pair.mean()


# ---------------------------------------------------------------------------
# teacher assembly (spec §5.7: ensemble vs single teacher)
# ---------------------------------------------------------------------------
def assemble_teacher(teachers, weights: Optional[Sequence[float]] = None,
                     mode: str = "mean"):
    """Combine teacher probability matrices into one target.

    * A **single** matrix (``ndarray`` / tensor, or a length-1 sequence) is
      returned unchanged -- this is the *single-teacher switch*.
    * Multiple matrices are combined by the weighted ``mean`` (spec §5.7:
      ``p^teacher = mean(p^ViennaRNA, p^RNAstructure, p^LinearPartition)``).

    Returns a numpy array (the teacher is a constant target, not a differentiable
    quantity).
    """
    if isinstance(teachers, np.ndarray) or torch.is_tensor(teachers):
        return np.asarray(teachers if not torch.is_tensor(teachers) else teachers.detach().cpu().numpy())
    mats = [t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t) for t in teachers]
    if len(mats) == 0:
        raise ValueError("assemble_teacher needs at least one teacher matrix")
    if len(mats) == 1:
        return mats[0]
    if mode != "mean":
        raise ValueError(f"unsupported ensemble mode {mode!r}; only 'mean' is implemented")
    stack = np.stack([m.astype(np.float64) for m in mats], axis=0)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != len(mats):
        raise ValueError("weights length must match the number of teachers")
    w = w / w.sum()
    return np.tensordot(w, stack, axes=(0, 0))


# ---------------------------------------------------------------------------
# exact-marginal teacher (the C1 mechanism)
# ---------------------------------------------------------------------------
def empirical_pair_frequencies(structures: Sequence[Sequence[Tuple[int, int]]],
                               L: int) -> np.ndarray:
    """Fraction of structures containing each pair (symmetric, diagonal 0)."""
    counts = np.zeros((L, L), dtype=np.float64)
    if len(structures) == 0:
        return counts
    for struct in structures:
        for i, j in struct:
            counts[i, j] += 1.0
            counts[j, i] += 1.0
    return counts / float(len(structures))


def teacher_scores_from_structures(structures, L: int, pseudo: float = 0.05) -> np.ndarray:
    """Turn ground-truth structure(s) into a smoothed log-odds score matrix.

    The empirical pair frequency is smoothed towards 1/2 with pseudo-counts and
    mapped to ``logit``, so a single hard structure (0/1 frequencies) yields a
    finite score matrix rather than ``+/-inf``.  Only the strict upper triangle is
    populated (the frozen harness convention).
    """
    freq = empirical_pair_frequencies(structures, L)
    smoothed = (freq + pseudo) / (1.0 + 2.0 * pseudo)
    scores = np.log(smoothed) - np.log1p(-smoothed)
    return np.triu(scores, k=1)


def exact_marginal_teacher(scores=None, structures=None, seq: Optional[str] = None,
                           mask=None, min_loop: int = MIN_LOOP, pseudo: float = 0.05):
    """Exact marginals ``p_hat`` used as the distillation target (C1).

    Two input modes:

    * ``scores`` -- a CRF-trained / thermodynamic score matrix.  ``p_hat`` is the
      exact Gibbs marginal from :func:`rnajepa.harness.inside_outside`.
    * ``structures`` -- a set of ground-truth structures, converted to a smoothed
      log-odds score matrix (:func:`teacher_scores_from_structures`) and then run
      through ``inside_outside``, i.e. projected onto the non-crossing Gibbs
      family.

    ``mask`` is derived from ``seq`` when not given.  Returns ``(logZ, p_hat)``.
    """
    if scores is None and structures is None:
        raise ValueError("provide either `scores` or `structures`")
    if mask is None:
        if seq is None:
            raise ValueError("`mask` or `seq` is required")
        mask = valid_pair_mask(seq, min_loop)
    if scores is None:
        L = np.asarray(mask).shape[0]
        scores = teacher_scores_from_structures(structures, L, pseudo=pseudo)
    logZ, p_hat = inside_outside(scores, mask)
    return logZ, p_hat


# ---------------------------------------------------------------------------
# saturation diagnostic (spec §5.7 / records/A3_latent_target_comparison.md)
# ---------------------------------------------------------------------------
def cosine_similarity(a, b, mask=None) -> float:
    """Cosine similarity between two matrices on the strict upper-triangle pairs."""
    x = as_tensor(a)
    y = as_tensor(b)
    sel = pair_selector(mask, x.shape[0])
    xv = x[sel]
    yv = y[sel]
    nx = float(torch.linalg.vector_norm(xv))
    ny = float(torch.linalg.vector_norm(yv))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(torch.dot(xv, yv) / (nx * ny))


@dataclass
class SaturationDiagnostic:
    """Detect early saturation of the distillation objective.

    A previous project line found the JEPA region-mean summary target saturated
    almost immediately (cos 0.993 within 60 steps, ``records/A3_latent_target_comparison.md``).
    Distillation is claimed *not* to have that failure mode, so it is monitored:
    the diagnostic records the student/teacher cosine similarity per step and
    flags saturation when it reaches ``cos_threshold`` within ``early_steps``.
    """

    cos_threshold: float = 0.99
    early_steps: int = 60
    min_updates: int = 2
    history: List[Tuple[int, float]] = field(default_factory=list)
    fired_at: Optional[int] = None

    def update(self, step: int, student_probs, teacher_probs, mask=None) -> float:
        cos = cosine_similarity(student_probs, teacher_probs, mask)
        self.history.append((int(step), cos))
        if self.fired_at is None and len(self.history) >= self.min_updates \
                and step <= self.early_steps and cos >= self.cos_threshold:
            self.fired_at = int(step)
        return cos

    @property
    def fired(self) -> bool:
        return self.fired_at is not None

    def summary(self) -> Dict[str, object]:
        return {
            "fired": self.fired,
            "fired_at": self.fired_at,
            "cos_threshold": self.cos_threshold,
            "early_steps": self.early_steps,
            "n_updates": len(self.history),
            "final_cos": self.history[-1][1] if self.history else None,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# calibration cost (gate S7)
# ---------------------------------------------------------------------------
def expected_calibration_error(probs, labels, mask=None, n_bins: int = 10) -> float:
    """Standard (hard-binned) ECE on the strict upper-triangle masked pairs.

    ``probs`` are predicted pair probabilities, ``labels`` the binary pair
    indicators.  This is the non-differentiable reference used for reporting;
    the differentiable version lives in :mod:`rnajepa.rlcd`.
    """
    p = as_tensor(probs).detach().cpu().numpy()
    a = as_tensor(labels).detach().cpu().numpy()
    sel = pair_selector(mask, p.shape[0]).numpy()
    p = p[sel]
    a = a[sel]
    if p.size == 0:
        return 0.0
    p = np.clip(p, 0.0, 1.0)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        ece += (m.sum() / p.size) * abs(p[m].mean() - a[m].mean())
    return float(ece)


def calibration_cost(student_probs, exact_marginals, labels, mask=None,
                     n_bins: int = 10, tol: float = 0.02) -> Dict[str, object]:
    """ECE of the student vs ECE of the exact marginals -- gate S7 (spec §8.3).

    S7 asks whether the DP-free student is calibrated to within ``tol`` of the
    exact-marginal ECE.  Returns the two ECEs, their absolute difference, and the
    boolean ``passes_s7``.
    """
    ece_student = expected_calibration_error(student_probs, labels, mask, n_bins)
    ece_exact = expected_calibration_error(exact_marginals, labels, mask, n_bins)
    delta = abs(ece_student - ece_exact)
    return {
        "ece_student": ece_student,
        "ece_exact": ece_exact,
        "delta": delta,
        "tol": tol,
        "passes_s7": delta <= tol,
    }


# ---------------------------------------------------------------------------
# teacher interface + stubs
# ---------------------------------------------------------------------------
class ThermodynamicUnavailableError(RuntimeError):
    """Raised when a real thermodynamic teacher is requested but unavailable."""


class ThermodynamicTeacher:
    """System-2 teacher backed by a thermodynamic partition-function package.

    **Not usable in this environment.**  ViennaRNA / RNAstructure / LinearPartition
    are not installed and cannot be downloaded here, and the teacher version (and
    hence the Turner parameters) must be locked for the soft labels to be
    reproducible (spec §3.5, §5.7, §8.1 G5).  Calling :meth:`predict_probs` raises
    a clear, actionable :class:`ThermodynamicUnavailableError` instead of silently
    degrading to a different model.
    """

    SUPPORTED = ("viennarna", "rnastructure", "linearpartition")

    def __init__(self, tool: str = "viennarna", version_lock: Optional[str] = None):
        if tool not in self.SUPPORTED:
            raise ValueError(f"unknown thermodynamic tool {tool!r}; supported: {self.SUPPORTED}")
        self.tool = tool
        self.version_lock = version_lock
        self.name = f"thermo:{tool}"

    def predict_probs(self, seq: str) -> np.ndarray:
        raise ThermodynamicUnavailableError(
            f"The thermodynamic teacher '{self.tool}' is not installed and cannot be "
            "downloaded in this environment, so exact teacher soft labels cannot be "
            "generated here. Install the locked version of the tool "
            f"({self.tool}, version_lock={self.version_lock!r}) and re-run "
            "generate_teacher_labels(); until then use MockTeacher in tests only. "
            "The teacher software + Turner-parameter version MUST be recorded and "
            "locked (spec §3.5, §8.1 G5)."
        )


class MockTeacher:
    """Deterministic synthetic teacher used **only in tests**.

    Produces exact Gibbs marginals of a seeded random score matrix (so the target
    is a genuine, non-crossing-consistent probability matrix) without any
    external dependency.  It is *not* a physical model and must never be used to
    produce reported soft labels.
    """

    def __init__(self, seed: int = 0, scale: float = 1.5):
        self.seed = seed
        self.scale = scale
        self.name = "mock"
        self.version_lock = "synthetic"

    def predict_probs(self, seq: str) -> np.ndarray:
        L = len(seq)
        if L == 0:
            return np.zeros((0, 0), dtype=np.float64)
        tag = int(sha256_hex(f"{self.seed}:{seq}")[:8], 16)
        rng = np.random.default_rng(tag)
        scores = np.triu(rng.normal(0.0, self.scale, size=(L, L)), k=1)
        _logZ, p_hat = inside_outside(scores, valid_pair_mask(seq))
        return p_hat


# ---------------------------------------------------------------------------
# teacher soft-label generation driver (sharding + resume + hash verification)
# ---------------------------------------------------------------------------
def sha256_hex(text: str) -> str:
    """SHA256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    """SHA256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("shard_size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def generate_teacher_labels(sequences: Sequence[str], teacher, out_dir: Union[str, Path],
                            shard_size: int = 64, resume: bool = True,
                            extra_meta: Optional[Dict[str, object]] = None,
                            write: bool = True) -> Dict[str, object]:
    """Generate teacher probability soft labels, sharded and resumable.

    Writes one ``shard_XXXXX.npz`` per ``shard_size`` sequences (holding the
    sequences and their ``L x L`` probability matrices) plus a ``manifest.json``
    recording, per shard, the file name, the sequence count, the file SHA256 and
    the SHA256 of the concatenated sequences.  With ``resume=True`` a shard whose
    file still matches its recorded hash is skipped, so an interrupted run
    continues instead of recomputing.  The manifest records the teacher name and
    ``version_lock`` so the soft labels are reproducible (spec §5.7, §8.1 G5).
    """
    out_dir = Path(out_dir)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    manifest: Dict[str, object] = {
        "teacher": getattr(teacher, "name", type(teacher).__name__),
        "version_lock": getattr(teacher, "version_lock", None),
        "shard_size": shard_size,
        "n_sequences": 0,
        "meta": dict(extra_meta or {}),
        "shards": [],
    }
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    done = {int(sh["index"]): sh for sh in manifest.get("shards", [])}
    seq_list = list(sequences)
    new_shards: List[Dict[str, object]] = []
    total = 0

    for idx, seqs in enumerate(_chunks(seq_list, shard_size)):
        path = out_dir / f"shard_{idx:05d}.npz"
        prev = done.get(idx)
        if resume and prev is not None and path.exists() and sha256_file(path) == prev["sha256"]:
            new_shards.append(prev)
            total += int(prev["n"])
            continue
        probs = [np.asarray(teacher.predict_probs(s), dtype=np.float64) for s in seqs]
        if write:
            np.savez(path, sequences=np.array(seqs, dtype=object),
                     probs=np.array(probs, dtype=object))
            digest = sha256_file(path)
        else:
            digest = ""
        record = {
            "index": idx,
            "file": path.name,
            "n": len(seqs),
            "sha256": digest,
            "seq_sha256": sha256_hex("\n".join(seqs)),
        }
        new_shards.append(record)
        total += len(seqs)
        if write:
            manifest["shards"] = new_shards
            manifest["n_sequences"] = total
            manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest["shards"] = new_shards
    manifest["n_sequences"] = total
    if write:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_teacher_labels(manifest: Dict[str, object], out_dir: Union[str, Path]) -> bool:
    """Re-hash every shard and confirm it matches the manifest.

    Raises ``AssertionError`` on a missing shard or a hash mismatch (silent
    corruption of the soft-label corpus is not acceptable).
    """
    out_dir = Path(out_dir)
    for shard in manifest.get("shards", []):
        path = out_dir / str(shard["file"])
        if not path.exists():
            raise AssertionError(f"teacher shard missing: {path}")
        actual = sha256_file(path)
        if actual != shard["sha256"]:
            raise AssertionError(
                f"teacher shard hash mismatch for {path}: {actual} != {shard['sha256']}"
            )
    return True


def load_teacher_shard(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load one shard: returns ``(sequences, probs)`` object arrays."""
    with np.load(path, allow_pickle=True) as data:
        return data["sequences"], data["probs"]
