"""Pretraining / fine-tuning driver for the decision model (Task 18).

What this file is
-----------------
The execution harness for the four-term objective of spec §5.2 / §5.7 / §5.8::

    L = L_NLL + lambda_distill * L_distill + lambda_RLCD * L_RLCD + lambda_1 * L_cal

Every weight is independently switchable so the 15 ablations of spec §7.5 (see
``eval/ss/ablations.py``) can be run as pure config changes.  The terms themselves
live in the frozen modules ``rnajepa.rlcd`` (``combined_loss``) and
``rnajepa.distill``; this driver only wires them to a model, a data source and a
teacher, and adds the execution guarantees the project needs:

* **gradient-coverage assertion** -- every learnable block that is on the loss
  path must actually receive a gradient;
* **NaN / Inf monitoring with a hard failure** on divergence (no silent
  continuation with a dead run);
* **learning-rate calibration** -- the earlier phase of this project measured that
  the documented default ``lr=1e-4`` destroys the pretrained encoder on tasks with
  more than ~10k rows (3'UTR-RBP collapsed to ``F1pos = 0.0``; see
  ``records/R3_GATE_AND_LR_CALIBRATION.md`` and
  ``records/INVESTIGATION_rbp_0_degenerate.md``).  The chosen LR is therefore
  *measured* on train/dev and recorded in ``run_meta.json``, never inherited;
* checkpointing, training curves, ``run_meta.json`` and a resumable ledger.

Honesty / provenance (spec §0.3, §0.7 problem 2, §5.8.4)
-------------------------------------------------------
* The Gibbs / partition-function framework used by the ``L_NLL`` term is **not our
  contribution** -- it is the CONTRAfold / CRF lineage.
* ``L_RLCD`` is an **RLCD-inspired, self-designed objective**, not Jev's RLCD
  (which is undisclosed); nothing here claims to reproduce or match it.
* The real thermodynamic teacher is **not installed and cannot be downloaded
  here**, so this driver runs against the synthetic :class:`MockTeacher` or against
  soft labels previously written by the ``rnajepa.distill`` driver.  A run that
  uses the mock teacher records that fact in ``run_meta.json`` and must never be
  reported as a physical distillation result.

Two implementation facts worth knowing (both measured here, not assumed)
-----------------------------------------------------------------------
1. ``-inf`` sentinels and the four-term objective.  ``FlatDecisionHead`` writes
   ``-inf`` on every illegal pair.  ``combined_loss`` contains
   ``total = s.sum() * 0.0`` and ``negative_log_likelihood`` contains
   ``gt_score = scores.sum() * 0.0``; ``(-inf) * 0.0`` is ``NaN``, so passing the
   raw head output makes ``L_NLL`` NaN even though no illegal pair is ever read
   (both the harness and ``select_pairs`` mask them).  The driver therefore
   replaces non-finite scores with a large negative finite sentinel before the
   objective is evaluated.  This is numerically equivalent -- the entries are
   never read -- but keeps the loss and its gradient finite.
2. **Zero-initialised ``MLP_T`` makes the first backward pass dead.**  Because
   ``TurnerResidual``'s last layer is zero-initialised (spec §5.4.2), at step 0
   ``d L / d z = W_last^T ... = 0``, so the encoder, ``PairRepresentation`` and the
   inner ``MLP_T`` layers receive an *exactly zero* gradient; only
   ``turner.net[-1]`` and the calibration temperature do not.  After a single
   optimizer step the last layer is non-zero and full gradient coverage resumes
   (measured: 3/26 non-zero parameters at step 0, 26/26 from step 1).  The
   coverage assertion is therefore evaluated after the first optimizer step, and
   the observation is recorded in ``run_meta.json``.

Usage (CPU, tiny synthetic data -- the path the test suite exercises)::

    python -m rnajepa.train_decision --synthetic 8 --length 24 --tiny \\
        --steps 6 --batch-size 2 --lr 1e-3 --out /tmp/run --dry-run

"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.distill import (
    MockTeacher,
    generate_teacher_labels,
    load_teacher_shard,
    pair_indicator,
    sha256_file,
)
from rnajepa.encoder import RNAEncoder, build_encoder
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.rlcd import ObjectiveWeights, combined_loss

__all__ = [
    "BASE_TO_ID",
    "DEFAULT_EXCLUDED_GRAD_BLOCKS",
    "NEG_BIG",
    "ConfigError",
    "DivergenceError",
    "GradientCoverageError",
    "TrainConfig",
    "DecisionExample",
    "DecisionDataset",
    "TeacherLabelStore",
    "make_synthetic_dataset",
    "write_mock_teacher_dir",
    "build_decision_model",
    "collate",
    "iter_batches",
    "objective_terms",
    "gradient_coverage",
    "assert_gradient_coverage",
    "_check_device_request",
    "_assert_device",
    "_batch_to_device",
    "calibrate_learning_rate",
    "run_training",
    "plan",
    "parse_args",
    "main",
]

#: Base -> encoder index (matches ``rnajepa.encoder.BASE_TO_INDEX``).
BASE_TO_ID: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}

#: Finite stand-in for the head's ``-inf`` illegal-pair sentinel (see docstring).
NEG_BIG = -1.0e4

#: Parameters that are deliberately *not* on the four-term loss path.
#:
#: * ``head.type_head`` -- the ordered pair-type head (spec §5.4.3) is supervised
#:   by its own pair-type objective, not by the pair-existence objective this
#:   driver implements;
#: * ``encoder.channel_proj`` -- the SHAPE / DMS probing channel (spec §5.3.1) is
#:   multiplied by the reactivity input and is a no-op when none is supplied (T3).
DEFAULT_EXCLUDED_GRAD_BLOCKS: Tuple[str, ...] = (
    "head.type_head",
    "encoder.channel_proj",
)

#: Provenance note carried in ``run_meta.json``.
HONESTY = {
    "gibbs_framework": "NOT our contribution -- CONTRAfold (Do et al. 2006) / CRF lineage (spec §0.7 problem 2)",
    "rlcd": "RLCD-inspired, self-designed objective; Jev's RLCD is undisclosed and is NOT reproduced or matched (spec §5.8.4)",
    "turner_prior": "MLP_T=0 yields exactly 'Nussinov + Turner stacking', NOT ViennaRNA (spec §0.7 problem 3)",
}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


class DivergenceError(RuntimeError):
    """Raised on a non-finite loss or gradient (hard failure, never silent)."""


class GradientCoverageError(RuntimeError):
    """Raised when a learnable block on the loss path receives no gradient."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Every hyper-parameter of one run (written verbatim into ``run_meta.json``)."""

    # schedule
    steps: int = 50
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 5
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 5
    save_every: int = 0
    max_hours: float = 0.0

    # four-term objective (all independently switchable -- spec §7.5 ablations)
    lambda_nll: float = 1.0
    lambda_distill: float = 1.0
    lambda_rlcd: float = 1.0
    lambda_cal: float = 1.0
    distill_kind: str = "kl"
    rlcd_reward: str = "brier"
    beta: float = 1.0
    n_bins: int = 10
    soft_ece_tau: float = 0.1
    #: How ``L_NLL`` is scaled before the four terms are added.  ``"sum"`` is the
    #: historical ``log Z - sum s_ij`` (``O(L)``); ``"length"`` divides it by ``L``.
    #: With ``"sum"`` the auxiliary terms measured ~0.4 against an NLL of ~55, so
    #: ``lambda_* = 1`` made them contribute <3% of the objective.  Defaults to
    #: ``"sum"`` so that resuming any pre-existing run is bit-exact.
    nll_normalization: str = "sum"

    # head / objective family
    #: ``"flat"`` = the original single ``L x L`` head with the four-term objective.
    #: ``"cascade"`` = the hierarchical head with the *layered* objective of
    #: ``rnajepa.cascade_objective``.  Before 2026-09-24 the cascade existed but was on
    #: neither the training nor the evaluation path, and could not have been trained
    #: even if it had been (its ``-inf`` mask made ``log Z`` meaningless and its hard
    #: ``top_k`` was not differentiable).  See ``spec/spec.md`` §0.9.5 finding A.
    head: str = "flat"
    cascade_block_size: int = 8
    cascade_top_k: int = 2
    cascade_d_z: int = 128
    cascade_hidden: int = 64
    cascade_soft_gate: bool = True
    cascade_gate_theta: float = 0.0
    cascade_gate_tau: float = 0.5
    cascade_l0: float = 1.0
    cascade_l1: float = 1.0
    cascade_l2: float = 1.0
    cascade_sparse: float = 0.1
    cascade_miss_cost: float = 20.0

    # model
    tiny: bool = False
    encoder_size: str = "150M"
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 3072
    d_z: int = 128
    hidden: int = 64
    #: Pair-scorer architecture: 'mlp' (per-pair, all existing arms) or
    #: 'resnet2d' (Plan B: 2D-context scorer over the pairing matrix).
    scorer: str = "mlp"
    resnet_blocks: int = 4
    #: Column-chunk width for the head's pair tensors (0 = no chunking).  The chunk
    #: transient is (B, L, chunk, 3*d), so this is the knob that decides whether a
    #: run fits on a small MIG slice; see tools/patch_head_checkpointing.py.
    head_chunk_size: int = 64
    max_len: int = 4096

    #: Initial value of the head's learnable Turner-prior multiplier.
    #:
    #: The prior used to be added at a hard-coded weight of 1, which training could
    #: not reach: ``MLP_T`` sees only ``z_ij`` (never the prior) and the temperature
    #: divides the sum, so ``MLP_T / prior`` is invariant.  A zero-training sweep
    #: (``tools/probe_prior_weight.py``, selected on bpRNA VL0) measured the optimum
    #: at w = 0.5 for the frozen-RiNALMo head (+0.048 micro F1) and ~1 for the
    #: from-scratch encoder, i.e. it is arm-dependent.  The parameter is learnable, so
    #: this only sets where the search starts; 1.0 keeps every existing checkpoint's
    #: exact behaviour.
    prior_init: float = 1.0

    # lr calibration
    lr_calibrate: bool = False
    lr_candidates: Tuple[float, ...] = (1e-4, 5e-5, 1e-5)
    lr_probe_steps: int = 5

    # bookkeeping
    out_dir: str = ""
    device: str = "cpu"
    #: Directory of frozen embeddings (``--embedding-dir``).  When set, the head
    #: trains on cached activations and no encoder is built at all.
    embedding_dir: str = ""
    #: Hidden size of the cached embeddings; required with ``embedding_dir`` because
    #: there is no encoder to read it from.
    embedding_d_model: int = 0
    #: Explicit opt-in for the CPU path.  Exists so the test suite can run the
    #: driver without a GPU; a real training run must not set it.
    allow_cpu: bool = False
    resume: str = ""
    arm: str = "decision"

    def __post_init__(self) -> None:
        if self.tiny:
            # CPU-sized model with the same topology (encoder + light head).
            self.encoder_size = "tiny"
            self.d_model = 32
            self.n_layer = 1
            self.n_head = 4
            self.d_ff = 64
            self.d_z = 16
            self.hidden = 16
            self.max_len = 256

    def validate(self) -> None:
        """Reject an inconsistent configuration before anything is built."""
        if self.steps <= 0:
            raise ConfigError(f"steps must be positive, got {self.steps}")
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0.0:
            raise ConfigError(f"lr must be positive, got {self.lr}")
        if self.distill_kind not in ("kl", "l2"):
            raise ConfigError(f"distill_kind must be 'kl' or 'l2', got {self.distill_kind!r}")
        if self.rlcd_reward not in ("brier", "log", "log_score"):
            raise ConfigError(
                f"rlcd_reward must be 'brier' or 'log', got {self.rlcd_reward!r}")
        if self.d_model % self.n_head != 0:
            raise ConfigError(f"d_model {self.d_model} not divisible by n_head {self.n_head}")
        weights = (self.lambda_nll, self.lambda_distill, self.lambda_rlcd, self.lambda_cal)
        if all(w == 0.0 for w in weights):
            raise ConfigError("all four objective weights are zero; there is nothing to optimise")
        if any(w < 0.0 for w in weights):
            raise ConfigError(f"objective weights must be non-negative, got {weights}")
        if self.lr_calibrate and not self.lr_candidates:
            raise ConfigError("lr_calibrate requires a non-empty lr_candidates grid")
        if self.head not in ("flat", "cascade"):
            raise ConfigError(f"head must be 'flat' or 'cascade', got {self.head!r}")
        if self.head == "cascade":
            if not self.cascade_soft_gate:
                raise ConfigError(
                    "head='cascade' requires cascade_soft_gate: without the gate the "
                    "cascade's L0 selection is the non-differentiable hard top_k whose "
                    "recall is structurally capped at 0.1548 against the P6 gate of 0.98, "
                    "and its -inf mask makes the CRF's log Z meaningless. Training it "
                    "would produce an arm that cannot support the C2 claim and whose logs "
                    "would not say so.")
            if self.cascade_block_size < 1:
                raise ConfigError(
                    f"cascade_block_size must be >= 1, got {self.cascade_block_size}")
            if self.cascade_gate_tau <= 0.0:
                raise ConfigError(
                    f"cascade_gate_tau must be > 0, got {self.cascade_gate_tau}")
            if min(self.cascade_l0, self.cascade_l1, self.cascade_l2,
                   self.cascade_sparse) < 0.0:
                raise ConfigError("layered cascade weights must be non-negative")
            if (self.cascade_l0, self.cascade_l1, self.cascade_l2,
                    self.cascade_sparse) == (0.0, 0.0, 0.0, 0.0):
                raise ConfigError(
                    "all layered cascade weights are zero; the cascade would have no "
                    "structure objective at all (its auxiliary terms alone cannot learn "
                    "which block pairs carry helices)")

    def objective_weights(self) -> ObjectiveWeights:
        return ObjectiveWeights(
            lambda_nll=self.lambda_nll,
            lambda_distill=self.lambda_distill,
            lambda_rlcd=self.lambda_rlcd,
            lambda_cal=self.lambda_cal,
        )

    def layered_weights(self):
        """The L0/L1/L2/sparse weights of the cascade objective (imported lazily)."""
        from rnajepa.cascade_objective import LayeredWeights

        return LayeredWeights(l0=self.cascade_l0, l1=self.cascade_l1,
                              l2=self.cascade_l2, sparse=self.cascade_sparse,
                              miss_cost=self.cascade_miss_cost)

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["lr_candidates"] = list(self.lr_candidates)
        return d


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class DecisionExample:
    """One training instance: sequence + hard labels + optional teacher soft labels.

    ``teacher_probs`` is ``None`` when the distillation term is switched off and
    the loader was therefore not asked to produce them.  It is never silently
    zero-filled: the objective raises if a run needs them and they are absent.
    """

    seq: str
    gt_pairs: List[Tuple[int, int]]
    teacher_probs: Optional[np.ndarray]
    source: str = "synthetic"
    #: Frozen per-residue embedding ``(L, d)`` from a pretrained backbone, or
    #: ``None`` when this run should compute them with its own encoder.  Never
    #: mixed within a batch -- see :class:`EmbeddingStore`.
    embedding: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def seq_ids(self) -> torch.Tensor:
        return torch.tensor([BASE_TO_ID.get(c, 4) for c in self.seq], dtype=torch.long)

    @property
    def mask(self) -> np.ndarray:
        return valid_pair_mask(self.seq)

    @property
    def labels(self) -> np.ndarray:
        return pair_indicator(self.length, self.gt_pairs)


class DecisionDataset:
    """A list of :class:`DecisionExample` plus its provenance."""

    def __init__(self, examples: Sequence[DecisionExample], *, source: str,
                 version: str) -> None:
        self.examples: List[DecisionExample] = list(examples)
        self.source = source
        self.version = version

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self.examples[index]

    def describe(self) -> Dict[str, object]:
        lengths = [e.length for e in self.examples]
        return {
            "source": self.source,
            "version": self.version,
            "n_examples": len(self.examples),
            "min_length": int(min(lengths)) if lengths else 0,
            "max_length": int(max(lengths)) if lengths else 0,
            # provenance: distinguishes "no soft labels needed" from "labels lost"
            "has_teacher_probs": bool(self.examples) and
                                 all(e.teacher_probs is not None for e in self.examples),
            "has_embeddings": bool(self.examples) and
                              all(e.embedding is not None for e in self.examples),
        }


def _random_seq(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGU")) for _ in range(length))


def make_synthetic_dataset(n: int, length: int, seed: int = 0,
                           teacher_seed: int = 0) -> DecisionDataset:
    """Tiny self-contained dataset: random sequences, legal structures, mock teacher.

    The hard labels are the Nussinov MAP of an *independent* random score matrix,
    so they are guaranteed legal (non-crossing, minimum hairpin loop respected)
    without needing any annotation file.  The teacher soft labels come from
    :class:`rnajepa.distill.MockTeacher`, whose target is a genuine Gibbs marginal
    of a seeded random score matrix -- a real probability matrix, but **not a
    physical model**.
    """
    if n <= 0:
        raise ConfigError(f"n must be positive, got {n}")
    if length <= 0:
        raise ConfigError(f"length must be positive, got {length}")
    teacher = MockTeacher(seed=teacher_seed)
    examples: List[DecisionExample] = []
    for i in range(n):
        rng = np.random.default_rng((seed, i))
        seq = _random_seq(rng, length)
        mask = valid_pair_mask(seq)
        scores = np.triu(rng.normal(0.0, 1.0, size=(length, length)), k=1)
        gt = nussinov_map(scores, mask)
        examples.append(DecisionExample(seq=seq, gt_pairs=[tuple(p) for p in gt],
                                        teacher_probs=teacher.predict_probs(seq),
                                        source="synthetic"))
    version = f"synthetic:seed={seed}:n={n}:L={length}:teacher_seed={teacher_seed}"
    return DecisionDataset(examples, source="synthetic", version=version)


def write_mock_teacher_dir(sequences: Sequence[str], out_dir: str,
                           seed: int = 0) -> Dict[str, object]:
    """Write a mock-teacher label directory using the ``distill`` driver.

    Exists so the consumption path (shard manifest -> :class:`TeacherLabelStore`)
    is exercised without the real thermodynamic tools.  The returned manifest
    records ``teacher="mock"``, which the driver propagates into
    ``run_meta.json``.
    """
    teacher = MockTeacher(seed=seed)
    return generate_teacher_labels(sequences, teacher, out_dir, shard_size=8)


class TeacherLabelStore:
    """Teacher soft labels keyed by sequence, with provenance for the run record."""

    def __init__(self, mapping: Dict[str, np.ndarray], *, name: str,
                 version_lock: Optional[str], source: str, n_labels: int) -> None:
        self.mapping = mapping
        self.name = name
        self.version_lock = version_lock
        self.source = source
        self.n_labels = n_labels

    @classmethod
    def from_mock(cls, seed: int = 0) -> "TeacherLabelStore":
        return cls({}, name="mock", version_lock="synthetic",
                   source=f"mock:seed={seed}", n_labels=0)

    @classmethod
    def from_dir(cls, directory: str) -> "TeacherLabelStore":
        """Load every shard written by ``distill.generate_teacher_labels``."""
        directory = str(directory)
        manifest_path = os.path.join(directory, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise ConfigError(f"no teacher manifest at {manifest_path}")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        mapping: Dict[str, np.ndarray] = {}
        for shard in manifest.get("shards", []):
            seqs, probs = load_teacher_shard(os.path.join(directory, str(shard["file"])))
            for seq, prob in zip(seqs, probs):
                mapping[str(seq)] = np.asarray(prob, dtype=np.float64)
        return cls(mapping, name=str(manifest.get("teacher", "unknown")),
                   version_lock=manifest.get("version_lock"),
                   source=f"dir:{directory}", n_labels=len(mapping))

    def probs_for(self, seq: str, *, teacher: Optional[MockTeacher] = None,
                  strict: bool = False) -> np.ndarray:
        """Soft labels for ``seq``, from the store or (unless strict) the mock.

        ``strict=True`` is used whenever a real ``--teacher-dir`` was supplied:
        falling back to the mock there would mix a non-physical teacher into
        labels the run reports as thermodynamic, so a missing sequence is an
        error rather than a substitution.
        """
        if seq in self.mapping:
            return self.mapping[seq]
        if strict:
            raise ConfigError(
                f"no teacher label for a {len(seq)} nt sequence (store={self.name!r}, "
                f"source={self.source!r}). Refusing to fall back to the mock teacher: "
                "the soft labels would no longer be the physical teacher this run "
                "reports. Regenerate the labels for this corpus, or drop "
                "--teacher-dir and set --lambda-distill 0.")
        if teacher is None:
            teacher = MockTeacher(seed=0)
        return teacher.predict_probs(seq)

    def meta(self) -> Dict[str, object]:
        return {"name": self.name, "version_lock": self.version_lock,
                "source": self.source, "n_labels": self.n_labels,
                "is_mock": self.name == "mock"}


class EmbeddingStore:
    """Frozen per-residue embeddings from ``tools/extract_rinalmo_embeddings.py``.

    Shards are ``<split>.shard{k}of{n}.npz`` holding a concatenated ``h`` array,
    an ``offsets`` vector and the ``seqs`` they belong to.  Lookup is **by
    sequence**, deliberately: the extractor excludes sequences longer than the
    model's positional limit, so a positional index would silently shift every
    embedding after the first exclusion.

    ``strict`` mirrors the teacher-store convention: when the run declares it is
    using cached embeddings, a missing sequence is an error rather than a quiet
    fallback to a randomly initialised encoder.
    """

    def __init__(self, mapping: Dict[str, np.ndarray], *, name: str,
                 d_model: Optional[int], n_sequences: int, source: str) -> None:
        self.mapping = mapping
        self.name = name
        self.d_model = d_model
        self.n_sequences = n_sequences
        self.source = source

    @classmethod
    def from_dir(cls, directory: str, *, split: str = "") -> "EmbeddingStore":
        directory = str(directory)
        manifest_path = os.path.join(directory, "manifest.json")
        entries: List[Dict[str, object]] = []
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                entries = list(json.load(fh).get("entries", []))
        else:
            # No manifest yet: the extractor writes it only after all its splits
            # finish, so a store must still work while extraction is in progress.
            # The split is taken from the file name, which is <split>.shardKofN.npz.
            entries = [{"path": os.path.join(directory, f),
                        "split": f.split(".")[0]}
                       for f in sorted(os.listdir(directory)) if f.endswith(".npz")]
        if not entries:
            raise ConfigError(f"no embedding shards found under {directory}")

        mapping: Dict[str, np.ndarray] = {}
        d_model = None
        n_excluded = 0
        for entry in entries:
            path = str(entry.get("path", ""))
            if not os.path.isfile(path):
                continue
            entry_split = str(entry.get("split")
                              or os.path.basename(path).split(".")[0])
            if split and not entry_split.startswith(split):
                continue
            n_excluded += int(entry.get("n_excluded_over_max_len", 0) or 0)
            with np.load(path, allow_pickle=True) as data:
                h, offsets, seqs = data["h"], data["offsets"], data["seqs"]
                d_model = int(h.shape[1]) if d_model is None else d_model
                for k, seq in enumerate(seqs):
                    mapping[str(seq)] = np.asarray(
                        h[offsets[k]:offsets[k + 1]], dtype=np.float32)
        if not mapping:
            raise ConfigError(f"embedding shards under {directory} contained no "
                              f"sequences matching split={split!r}")
        return cls(mapping, name=f"frozen:{os.path.basename(directory)}",
                   d_model=d_model, n_sequences=len(mapping),
                   source=f"dir:{directory}:split={split or '*'}:"
                          f"excluded_over_max_len={n_excluded}")

    def get(self, seq: str) -> np.ndarray:
        try:
            return self.mapping[seq]
        except KeyError as exc:
            raise ConfigError(
                f"no cached embedding for a {len(seq)} nt sequence "
                f"(store={self.name!r}, {self.n_sequences} sequences). Refusing to "
                "fall back to the encoder: the run declares it uses frozen "
                "embeddings, and mixing the two would make the result meaningless. "
                "Re-extract this split, or drop --embedding-dir.") from exc

    def meta(self) -> Dict[str, object]:
        return {"name": self.name, "d_model": self.d_model,
                "n_sequences": self.n_sequences, "source": self.source}


def load_dataset_from_jsonl(path: str, teacher: TeacherLabelStore,
                            mock_seed: int = 0, *,
                            need_teacher_probs: bool = True,
                            strict_teacher: bool = False,
                            embedding_store: Optional["EmbeddingStore"] = None
                            ) -> DecisionDataset:
    """Load ``{"seq": ..., "structure": "<dot-bracket>"}`` records from JSONL.

    Hard labels are parsed from the dot-bracket string (reusing the frozen
    ``rnajepa.clean.c3_structure.parse_pairs``).

    ``need_teacher_probs=False`` skips soft-label generation entirely and stores
    ``None``.  That matters: without a ``--teacher-dir`` the store is the mock
    one, whose ``predict_probs`` runs an O(L^3) numpy DP per sequence, so
    computing labels that ``lambda_distill == 0`` will never read costs ~2.5e10
    operations over a 10k-sequence corpus and made a real run appear hung.
    ``objective_terms`` hard-fails if the labels are needed and absent, so this
    is not a silent zero-fill.

    ``strict_teacher=True`` (set whenever a real teacher dir is in use) makes a
    missing sequence an error rather than a mock fallback.
    """
    from rnajepa.clean.c3_structure import parse_pairs

    mock = MockTeacher(seed=mock_seed) if need_teacher_probs else None
    examples: List[DecisionExample] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seq = str(record["seq"]).upper().replace("T", "U")
            pairs = [tuple(p) for p in record["pairs"]] if "pairs" in record \
                else parse_pairs(str(record["structure"]))
            probs = (teacher.probs_for(seq, teacher=mock, strict=strict_teacher)
                     if need_teacher_probs else None)
            embedding = (embedding_store.get(seq) if embedding_store is not None
                         else None)
            examples.append(DecisionExample(
                seq=seq, gt_pairs=sorted(pairs),
                teacher_probs=probs, source="jsonl", embedding=embedding))
    return DecisionDataset(examples, source=f"jsonl:{path}",
                           version=sha256_file(path))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_decision_model(config: TrainConfig):
    """Encoder + flat decision head (single forward pass -> ``L x L`` scores).

    The head is sized from the *encoder that was actually built*, not from
    ``config.d_model``.  ``build_encoder(size=...)`` picks its own hidden size
    (35M -> 512, 150M -> 768, 650M -> 1024) and ``PairRepresentation.proj`` is
    ``nn.Linear(3 * d_model, d_z)``, so reading ``config.d_model`` here builds a
    mismatched head whenever the preset differs from the default 150M.  That is
    what produced "mat1 and mat2 shapes cannot be multiplied (38642x1536 and
    2304x128)" on the first 35M run.
    """
    from rnajepa.decision_head import DecisionModel, FlatDecisionHead, HierarchicalCascade, HeadOnlyModel

    cascade = config.head == "cascade"

    def _build_head(d_model: int):
        if cascade:
            return HierarchicalCascade(
                d_model=d_model, block_size=config.cascade_block_size,
                top_k=config.cascade_top_k, d_z=config.cascade_d_z,
                hidden=config.cascade_hidden, soft_gate=config.cascade_soft_gate,
                gate_theta=config.cascade_gate_theta, gate_tau=config.cascade_gate_tau)
        return FlatDecisionHead(d_model=d_model, d_z=config.d_z, hidden=config.hidden,
            scorer=config.scorer, resnet_blocks=config.resnet_blocks,
                                chunk_size=config.head_chunk_size)

    if config.embedding_dir:
        if not config.embedding_d_model:
            raise ConfigError(
                "--embedding-dir requires --embedding-d-model: with cached embeddings "
                "there is no encoder to read the hidden size from, and guessing it "
                "would build a head whose projection silently mismatches.")
        return HeadOnlyModel(_build_head(int(config.embedding_d_model)))

    if config.tiny:
        encoder = RNAEncoder(d_model=config.d_model, n_layer=config.n_layer,
                             n_head=config.n_head, d_ff=config.d_ff,
                             max_len=config.max_len, window=None, global_stride=None)
    else:
        encoder = build_encoder(size=config.encoder_size)

    encoder_dim = int(getattr(encoder, "d_model", config.d_model))
    return DecisionModel(encoder, _build_head(encoder_dim))


# ---------------------------------------------------------------------------
# batching (pad to the longest sequence in the batch, slice per example)
# ---------------------------------------------------------------------------
def collate(examples: Sequence[DecisionExample]) -> Dict[str, object]:
    """Pad a list of examples into one batch; masks/labels stay per-example."""
    lengths = [e.length for e in examples]
    max_len = max(lengths)
    rows = len(examples)
    seq_ids = torch.zeros((rows, max_len), dtype=torch.long)
    attention = torch.zeros((rows, max_len), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = example.seq_ids
        seq_ids[row, :example.length] = ids
        attention[row, :example.length] = 1
    return {
        "seq_ids": seq_ids,
        "attention_mask": attention,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "masks": [e.mask for e in examples],
        "gt_pairs": [e.gt_pairs for e in examples],
        "teacher_probs": [e.teacher_probs for e in examples],
        "labels": [e.labels for e in examples],
        "embeddings": [e.embedding for e in examples],
    }


def iter_batches(dataset: DecisionDataset, batch_size: int, seed: int,
                 epoch: int = 0) -> Iterator[Dict[str, object]]:
    """Yield length-bucketed batches; bucket order is shuffled per epoch.

    Bucketing by encoded length keeps padding bounded.  The shuffle is seeded by
    ``(seed, epoch)`` so a resumed run reproduces the same batch sequence.
    """
    order = sorted(range(len(dataset)), key=lambda i: dataset[i].length)
    buckets = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng = np.random.default_rng((seed, epoch))
    for index in rng.permutation(len(buckets)):
        yield collate([dataset[i] for i in buckets[int(index)]])


def _batch_stream(dataset: DecisionDataset, config: TrainConfig,
                  start_step: int = 0) -> Iterator[Dict[str, object]]:
    """Endless batch stream; skips ``start_step`` batches when resuming."""
    skipped = 0
    epoch = 0
    while True:
        for batch in iter_batches(dataset, config.batch_size, config.seed, epoch):
            if skipped < start_step:
                skipped += 1
                continue
            yield batch
        epoch += 1


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def _batch_to_device(batch: Dict[str, object], device) -> Dict[str, object]:
    """Move every device-sensitive member of a collated batch onto ``device``.

    ``collate`` builds CPU tensors and numpy arrays, and the frozen loss modules
    infer device from their tensor arguments rather than moving anything
    (``distill.as_tensor`` preserves the device of a tensor but creates a *CPU*
    tensor from a numpy array).  On ``device="cpu"`` that is invisible; on a GPU
    it fails on the first forward pass with "found at least two devices".  The
    batch is therefore normalised here, in one place.

    ``gt_pairs`` is intentionally left alone: it is consumed by the numpy DP and
    by Python-level indexing, never as a tensor.
    """
    if not str(device).startswith("cuda"):
        return batch

    moved: Dict[str, object] = dict(batch)
    for key in ("seq_ids", "attention_mask", "lengths"):
        value = moved.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
    for key in ("masks", "labels", "teacher_probs", "embeddings"):
        values = moved.get(key)
        if values is None:
            continue
        converted = []
        for item in values:  # type: ignore[union-attr]
            if item is None:
                # teacher_probs is None by design when the distillation term is
                # off; np.asarray(None) is an object array and torch refuses it.
                # objective_terms() rejects a None that is actually needed, so
                # passing it through here cannot hide a missing label.
                converted.append(None)
            elif torch.is_tensor(item):
                converted.append(item.to(device, non_blocking=True))
            else:
                converted.append(torch.as_tensor(np.asarray(item), device=device))
        moved[key] = converted
    return moved



def _stack_embeddings(items, max_len: int, device, dtype=torch.float32) -> torch.Tensor:
    """``(B, max_len, d)`` from a list of ``(L_i, d)`` arrays/tensors, zero-padded.

    Accepts numpy as well as tensors: ``_batch_to_device`` deliberately returns the
    batch unchanged on CPU (the frozen loss modules take numpy), so on that path the
    embeddings arrive as arrays.  Converting here keeps a single owner for "cached
    embedding -> padded tensor" instead of adding a second transfer path.

    Padding is safe because ``objective_terms`` slices each row to
    ``[:length, :length]`` before the objective, so padded positions are never read.
    """
    if not items:
        raise ConfigError("empty embedding list")
    converted = [item if torch.is_tensor(item)
                 else torch.as_tensor(np.asarray(item), dtype=dtype)
                 for item in items]
    d = int(converted[0].shape[-1])
    out = torch.zeros((len(converted), max_len, d), dtype=dtype, device=device)
    for row, item in enumerate(converted):
        out[row, :item.shape[0]] = item.to(device=device, dtype=dtype)
    return out


def objective_terms(model, batch: Dict[str, object], weights: ObjectiveWeights, *,
                     distill_kind: str = "kl", reward: str = "brier", beta: float = 1.0,
                     n_bins: int = 10, tau: float = 0.1,
                     nll_normalization: str = "sum",
                     cascade: bool = False,
                     cascade_weights: Optional[LayeredWeights] = None,
                     cascade_block_size: int = 8
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mean four-term objective over a batch, plus the per-term values.

    Non-finite head scores are replaced by :data:`NEG_BIG` before the objective is
    evaluated (see the module docstring: ``(-inf) * 0.0`` is ``NaN`` and the
    frozen objective contains such a product).
    """
    if weights.lambda_distill != 0.0 and any(
            t is None for t in batch["teacher_probs"]):  # type: ignore[union-attr]
        raise ConfigError(
            "lambda_distill != 0 but the dataset carries no teacher probabilities. "
            "Pass --teacher-dir, or set --lambda-distill 0. Refusing to substitute "
            "zeros: the distillation term would be meaningless and the run would "
            "still report it as active.")
    batch = _batch_to_device(batch, next(model.parameters()).device)

    # The cached-embedding path and the own-encoder path are mutually exclusive.
    # A batch half of whose rows came from a frozen 650 M encoder and half from a
    # trainable 35 M encoder would produce a loss that means nothing, and nothing
    # downstream would flag it -- so it is rejected here instead.
    embeddings = batch.get("embeddings")
    has_embeddings = embeddings is not None and any(e is not None for e in embeddings)
    if has_embeddings != bool(getattr(model, "head_only", False)):
        raise ConfigError(
            "the batch and the model disagree about where representations come from: "
            f"batch has_embeddings={has_embeddings}, "
            f"model.head_only={bool(getattr(model, 'head_only', False))}. "
            "Pass --embedding-dir consistently for the whole run.")

    if cascade:
        from rnajepa.cascade_objective import LayeredWeights

        return _cascade_objective_terms(
            model, batch, weights, cascade_weights or LayeredWeights(),
            has_embeddings=has_embeddings, block_size=cascade_block_size,
            distill_kind=distill_kind, reward=reward, beta=beta, n_bins=n_bins, tau=tau)

    if has_embeddings:
        h = _stack_embeddings(embeddings, int(batch["seq_ids"].shape[1]),
                              device=batch["seq_ids"].device)
        scores = model(h, batch["seq_ids"], lengths=batch["lengths"])
    else:
        scores = model(batch["seq_ids"], lengths=batch["lengths"])
    matrix = scores.scores
    total: Optional[torch.Tensor] = None
    aggregated: Dict[str, float] = {}
    n = int(matrix.shape[0])
    for b in range(n):
        length = int(batch["lengths"][b])
        s = matrix[b, :length, :length]
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, NEG_BIG))
        loss, terms = combined_loss(
            s, batch["masks"][b], batch["gt_pairs"][b], weights,
            teacher_probs=batch["teacher_probs"][b],
            student_probs=torch.sigmoid(s),
            labels=torch.as_tensor(batch["labels"][b], dtype=torch.float64),
            distill_kind=distill_kind, reward=reward, beta=beta,
            n_bins=n_bins, tau=tau, return_terms=True,
            nll_normalization=nll_normalization,
        )
        total = loss if total is None else total + loss
        for name, value in terms.items():
            aggregated[name] = aggregated.get(name, 0.0) + float(value.detach())
    if total is None:
        raise ConfigError("empty batch: nothing to optimise")
    for name in aggregated:
        aggregated[name] /= n
    return total / n, aggregated


def _cascade_objective_terms(model, batch: Dict[str, object], weights: ObjectiveWeights,
                             cascade_weights: LayeredWeights, *,
                             has_embeddings: bool, block_size: int,
                             distill_kind: str = "kl", reward: str = "brier",
                             beta: float = 1.0, n_bins: int = 10, tau: float = 0.1,
                             ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Layered objective over a batch for the hierarchical head.

    Uses :meth:`forward_gated`, **not** the hard ``top_k`` path: only the gated forward
    produces a finite score matrix, and only the gated forward has a ``gate`` for the
    L0/L1 terms to act on.  Calling the hard path here would silently train an
    architecture whose L0 is not learnable -- the exact state the second-round audit
    found the cascade in.

    The two objective families **compose** rather than replace each other.  The layered
    terms (L0/L1/L2/sparse) own the structure decision, and ``L2`` is the only CRF
    likelihood; ``weights.lambda_nll`` is therefore forced to 0 in the auxiliary call so
    the likelihood is not counted twice.  The distillation / RLCD / calibration terms
    still act on ``sigmoid`` of the gated score matrix -- i.e. on the head's pair
    probabilities *after* the block gate -- which is what keeps C1's calibration
    machinery testable on the cascade instead of only on the flat head.

    The returned ``terms`` carries both the losses and the **P6 metrics**
    (``l0_helix_recall``, and the stricter ``l0_blockpair_recall`` the L0 loss actually
    optimises) so the gate can be watched during training rather than discovered at
    evaluation time.
    """
    from rnajepa.cascade_objective import (
        block_pair_presence, l0_helix_recall_pooled, l0_metrics, layered_cascade_loss)

    seq_ids = batch["seq_ids"]
    if has_embeddings:
        h = _stack_embeddings(batch["embeddings"], int(seq_ids.shape[1]),
                              device=seq_ids.device)
        out = model.forward_gated(h, seq_ids)
    else:
        out = model.forward_gated(seq_ids)

    gt_pairs = batch["gt_pairs"]
    target, compat = block_pair_presence(seq_ids, gt_pairs, block_size=block_size)

    total, terms = layered_cascade_loss(
        block_logits=out.block_logits, gate=out.gate, target=target, compat=compat,
        helix_logits=out.helix_logits, scores=out.scores, mask=batch["masks"],
        gt_pairs=gt_pairs, weights=cascade_weights, block_size=block_size,
        return_terms=True)

    metrics: Dict[str, float] = {k: float(v.detach()) for k, v in terms.items()}

    # ---- auxiliary (calibration) terms, composed on top of L2 ---------------- #
    aux = replace(weights, lambda_nll=0.0)
    if any(w != 0.0 for w in (aux.lambda_distill, aux.lambda_rlcd, aux.lambda_cal)):
        n = int(seq_ids.shape[0])
        aux_total: Optional[torch.Tensor] = None
        aux_agg: Dict[str, float] = {}
        for b in range(n):
            length = int(batch["lengths"][b])
            s = out.scores[b, :length, :length]
            loss_b, terms_b = combined_loss(
                s, batch["masks"][b], gt_pairs[b], aux,
                teacher_probs=batch["teacher_probs"][b],
                student_probs=torch.sigmoid(s),
                labels=torch.as_tensor(batch["labels"][b], dtype=torch.float64),
                distill_kind=distill_kind, reward=reward, beta=beta,
                n_bins=n_bins, tau=tau, return_terms=True)
            aux_total = loss_b if aux_total is None else aux_total + loss_b
            for name, value in terms_b.items():
                aux_agg[name] = aux_agg.get(name, 0.0) + float(value.detach())
        if aux_total is not None:
            total = total + aux_total / n
            for name, value in aux_agg.items():
                metrics[f"aux_{name}"] = value / n

    metrics.update(l0_helix_recall_pooled(out.block_kept, gt_pairs, block_size))
    metrics.update(l0_metrics(out.gate.detach(), target, compat))
    return total, metrics


# ---------------------------------------------------------------------------
# gradient coverage
# ---------------------------------------------------------------------------
def gradient_coverage(model, *, excluded: Sequence[str] = DEFAULT_EXCLUDED_GRAD_BLOCKS
                      ) -> Dict[str, List[str]]:
    """Classify every learnable parameter by the gradient it currently holds.

    Returns ``{"covered", "zero", "missing", "excluded", "nonfinite"}``.  A
    parameter is ``missing`` when ``.grad is None`` (it is disconnected from the
    loss graph), which is the failure this check exists to catch; ``zero`` is
    reported separately because an exactly zero gradient is legitimate for a
    zero-initialised sub-network or a zero input.
    """
    covered, zero, missing, excluded_names, nonfinite = [], [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in excluded):
            excluded_names.append(name)
            continue
        if param.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(param.grad).all()):
            nonfinite.append(name)
        elif float(param.grad.abs().sum()) == 0.0:
            zero.append(name)
        else:
            covered.append(name)
    return {"covered": covered, "zero": zero, "missing": missing,
            "excluded": excluded_names, "nonfinite": nonfinite}


def assert_gradient_coverage(model, *, excluded: Sequence[str] = DEFAULT_EXCLUDED_GRAD_BLOCKS,
                             require_nonzero: bool = True) -> Dict[str, List[str]]:
    """Hard-fail unless every non-excluded learnable block receives a gradient.

    ``require_nonzero=True`` additionally rejects an all-zero gradient, which is
    what the zero-initialised ``MLP_T`` produces on the very first backward pass
    (see the module docstring); call this after the first optimizer step.
    """
    report = gradient_coverage(model, excluded=excluded)
    if report["nonfinite"]:
        raise GradientCoverageError(
            f"non-finite gradient on {len(report['nonfinite'])} parameter(s): "
            f"{report['nonfinite'][:5]}")
    if report["missing"]:
        raise GradientCoverageError(
            f"{len(report['missing'])} learnable block(s) receive no gradient: "
            f"{report['missing'][:8]}; either wire them into the objective or add "
            f"them to `excluded` with a documented reason")
    if require_nonzero and report["zero"]:
        raise GradientCoverageError(
            f"{len(report['zero'])} learnable block(s) have an exactly zero gradient: "
            f"{report['zero'][:8]}")
    return report


# ---------------------------------------------------------------------------
# device enforcement
# ---------------------------------------------------------------------------
def _check_device_request(device: str, *, allow_cpu: bool = False) -> str:
    """Validate a device request *before* any tensor is moved to it.

    Must run before ``model.to(device)``: on a CPU-only torch build,
    ``.to("cuda")`` raises ``AssertionError("Torch not compiled with CUDA
    enabled")`` from inside torch, so checking afterwards never happens and the
    error the user sees is torch's rather than ours.
    """
    requested = str(device)
    if requested.startswith("cuda"):
        try:
            available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - CPU-only builds raise instead of returning False
            available = False
        if not available:
            raise ConfigError(
                f"device={requested!r} was requested but CUDA is not available in "
                "this interpreter; refusing to fall back to CPU silently. Install a "
                "CUDA build of torch, or pass --allow-cpu for the CPU path (tests only).")
        return f"cuda:{torch.cuda.current_device()}"
    if requested == "cpu" and not allow_cpu:
        raise ConfigError(
            "device='cpu' requires --allow-cpu. Training and GPU validation must "
            "run on a GPU (project rule); the CPU path exists for the test suite "
            "only, and a CPU run must never be reported as a training result.")
    return requested


def _assert_device(model, device: str, *, allow_cpu: bool = False) -> str:
    """Validate the request, then confirm the parameters really landed there.

    Hard-fails rather than silently training on the wrong device: a run that
    quietly fell back to CPU would still produce a loss curve and a checkpoint,
    which is exactly the failure mode worth preventing.  Returns the resolved
    device string for the run record.
    """
    resolved = _check_device_request(device, allow_cpu=allow_cpu)
    if resolved.startswith("cuda") and not any(p.is_cuda for p in model.parameters()):
        raise ConfigError(
            f"device={device!r} was requested but the model parameters are not on "
            "CUDA; refusing to continue")
    return resolved


# ---------------------------------------------------------------------------
# divergence monitoring
# ---------------------------------------------------------------------------
def _finite(value, what: str) -> float:
    """Return ``value`` as a float, raising :class:`DivergenceError` if not finite."""
    number = float(value)
    if not math.isfinite(number):
        raise DivergenceError(f"{what} is not finite ({number}); refusing to continue")
    return number


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# learning-rate calibration
# ---------------------------------------------------------------------------
def calibrate_learning_rate(config: TrainConfig, dataset: DecisionDataset, *,
                            candidates: Optional[Sequence[float]] = None,
                            probe_steps: Optional[int] = None,
                            verbose: bool = False) -> Dict[str, object]:
    """Probe each candidate LR for a few steps; return the chosen value and the table.

    The earlier phase of this project measured that the documented default
    ``lr=1e-4`` destroys the pretrained encoder on tasks with more than ~10k rows
    (``records/R3_GATE_AND_LR_CALIBRATION.md``).  Rather than inherit a default,
    each candidate is run for ``probe_steps`` optimiser steps on a freshly built
    model and scored by its final (finite) objective value; the best non-diverged
    candidate wins.  Diverged candidates (non-finite loss or gradient) are recorded
    and never selected.

    Note (honest scope): the probe selects on the *training* objective.  In a real
    run the selection must use a dev split -- the spec freezes "the test set never
    participates in hyper-parameter selection", and this driver's caller is
    responsible for passing the dev dataset here.
    """
    candidates = tuple(candidates if candidates is not None else config.lr_candidates)
    if not candidates:
        raise ConfigError("calibrate_learning_rate needs at least one candidate")
    probe_steps = int(probe_steps if probe_steps is not None else config.lr_probe_steps)
    if probe_steps <= 0:
        raise ConfigError(f"probe_steps must be positive, got {probe_steps}")

    weights = config.objective_weights()
    rows: List[Dict[str, object]] = []
    for lr in candidates:
        torch.manual_seed(config.seed)
        _check_device_request(config.device, allow_cpu=config.allow_cpu)
        model = build_decision_model(replace(config, lr=float(lr)))
        model.to(config.device)
        _assert_device(model, config.device, allow_cpu=config.allow_cpu)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(lr), weight_decay=config.weight_decay, betas=(0.9, 0.98))
        losses: List[float] = []
        diverged = False
        stream = _batch_stream(dataset, replace(config, batch_size=config.batch_size))
        for _ in range(probe_steps):
            batch = next(stream)
            optimizer.zero_grad(set_to_none=True)
            loss, _terms = objective_terms(
                model, batch, weights, distill_kind=config.distill_kind,
                reward=config.rlcd_reward, beta=config.beta,
                n_bins=config.n_bins, tau=config.soft_ece_tau,
                nll_normalization=config.nll_normalization,
                cascade=config.head == "cascade",
                cascade_weights=config.layered_weights(),
                cascade_block_size=config.cascade_block_size)
            try:
                value = _finite(loss.detach(), f"probe loss at lr={lr}")
                loss.backward()
                norm = _grad_norm(model.parameters())
                if not math.isfinite(norm):
                    raise DivergenceError(f"probe gradient norm at lr={lr} is {norm}")
            except DivergenceError:
                diverged = True
                break
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            losses.append(value)
        row = {"lr": float(lr), "diverged": diverged,
               "final_loss": (losses[-1] if losses else None), "losses": losses}
        rows.append(row)
        if verbose:
            print(f"[lr-calib] lr={lr:g} diverged={diverged} final={row['final_loss']}",
                  flush=True)

    usable = [r for r in rows if not r["diverged"] and r["final_loss"] is not None]
    if usable:
        best = min(usable, key=lambda r: float(r["final_loss"]))
        chosen = float(best["lr"])
        reason = (f"lowest finite probe loss ({best['final_loss']:.6g}) among "
                  f"{len(usable)} non-diverged candidate(s)")
    else:
        chosen = float(candidates[-1])
        reason = "every candidate diverged; falling back to the most conservative candidate"
    return {"chosen": chosen, "candidates": [float(c) for c in candidates],
            "probe_steps": probe_steps, "rows": rows, "reason": reason,
            "selection_split": "train objective (caller must pass a dev split in real runs)"}


# ---------------------------------------------------------------------------
# bookkeeping helpers
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    """Best-effort commit hash; ``待核验`` when unavailable (never invented)."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never crash a run
        pass
    return "待核验"


def _ledger_row(path: str, row: Dict[str, object]) -> None:
    row = dict(row)
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    row.setdefault("host", socket.gethostname())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_resume(path: str, model, optimizer, scheduler, step: int, config: TrainConfig,
                 coverage: Optional[Dict[str, object]] = None) -> None:
    torch.save({
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.as_dict(),
        "gradient_coverage": coverage,
    }, path)


# ---------------------------------------------------------------------------
# plan / dry run
# ---------------------------------------------------------------------------
def plan(config: TrainConfig, dataset: DecisionDataset, teacher: TeacherLabelStore
         ) -> Dict[str, object]:
    """Validate the configuration and describe what a real run would do.

    Used by ``--dry-run``: it touches no model, no optimizer and no checkpoint, so
    it cannot start training by accident.
    """
    config.validate()
    weights = config.objective_weights()
    active = [name for name, value in (("nll", weights.lambda_nll),
                                       ("distill", weights.lambda_distill),
                                       ("rlcd", weights.lambda_rlcd),
                                       ("cal", weights.lambda_cal)) if value != 0.0]
    steps_per_epoch = max(1, math.ceil(len(dataset) / config.batch_size))
    return {
        "dry_run": True,
        "arm": config.arm,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "lr_calibrate": config.lr_calibrate,
        "objective": {name: getattr(weights, f"lambda_{name}") for name in
                      ("nll", "distill", "rlcd", "cal")},
        "active_terms": active,
        "distill_kind": config.distill_kind,
        "rlcd_reward": config.rlcd_reward,
        "beta": config.beta,
        "device": config.device,
        "out_dir": config.out_dir,
        "data": dataset.describe(),
        "teacher": teacher.meta(),
        "steps_per_epoch": steps_per_epoch,
        "epochs": config.steps / steps_per_epoch,
        "model": {"tiny": config.tiny, "encoder_size": config.encoder_size,
                  "d_model": config.d_model, "n_layer": config.n_layer},
        "would_write": [os.path.join(config.out_dir, name) for name in
                        ("run_meta.json", "train_log.jsonl", "ledger.jsonl", "resume.pt")],
        "honesty": dict(HONESTY),
    }


# ---------------------------------------------------------------------------
# the training run
# ---------------------------------------------------------------------------
def _pad_optimizer_groups(saved: Dict[str, object], optimizer,
                          old_state_keys: Sequence[str],
                          current_names: Sequence[str],
                          defaultable: Sequence[str]) -> Dict[str, object]:
    """Make a saved optimizer state loadable after parameters were added.

    ``Optimizer.load_state_dict`` matches parameters **by position within a group**, so
    a checkpoint taken before a parameter existed cannot be loaded.  Padding the tail
    is *not* enough: a module's own ``Parameter``s are enumerated **before** its
    submodules', so ``head.prior_weight`` lands right after the encoder -- at index 78
    of 89, not at the end.  Appending shifted every later slot by one and AdamW then
    applied a saved moment buffer to the wrong parameter and died with
    ``The size of tensor a (128) must match the size of tensor b (1536)``.

    So the slots are matched **by name** instead: the checkpoint's parameter order is
    recovered from its own ``state_dict`` keys, and each current parameter either
    reuses the saved index whose state belongs to it, or gets a fresh index with no
    state.  A fresh index is correct -- AdamW creates its moment buffers lazily, so an
    added parameter starts from zero momentum while every pre-existing one keeps its
    history.

    Guarded rather than permissive: every parameter in the checkpoint must still exist
    (otherwise a parameter was removed and resume cannot repair it), and every current
    parameter missing from the checkpoint must be in ``defaultable``.
    """
    saved = dict(saved)
    groups = [dict(group) for group in saved["param_groups"]]
    if len(groups) != len(optimizer.param_groups):
        raise ConfigError(
            f"optimizer state has {len(groups)} parameter group(s) but the model has "
            f"{len(optimizer.param_groups)}; refusing to guess")

    current = list(current_names)
    current_set = set(current)
    # The checkpoint stores no names, but its state_dict does, in the same order the
    # optimizer saw them.  That correspondence only holds if the state_dict contains
    # exactly the parameters -- a persistent buffer would occupy a key without a slot,
    # so the equality is asserted instead of assumed.
    old = list(old_state_keys)
    n_slots = len(groups[0]["params"])
    if len(old) != n_slots:
        raise ConfigError(
            f"the checkpoint's state_dict has {len(old)} entries but its optimizer has "
            f"{n_slots} slot(s); they must correspond one-to-one for the name-based "
            "remap to be valid, and a persistent buffer would break that")
    removed = [name for name in old if name not in current_set]
    if removed:
        raise ConfigError(
            f"the checkpoint has parameter(s) this model does not: {removed[:5]}; a "
            "parameter was removed, which resume cannot repair")
    old_index = {name: i for i, name in enumerate(old)}
    unexplained = [name for name in current
                   if name not in old_index and name not in set(defaultable)]
    if unexplained:
        raise ConfigError(
            f"the checkpoint is missing parameter(s) that are not known-defaultable: "
            f"{unexplained[:5]} (known-defaultable: {sorted(defaultable)})")

    state_keys = set(saved["state"].keys())
    fresh = (max(state_keys) + 1) if state_keys else 0
    new_ids: List[int] = []
    for name in current:
        old_i = old_index.get(name)
        if old_i is not None and old_i in state_keys:
            new_ids.append(old_i)                 # reuse the slot that holds its state
        else:
            new_ids.append(fresh)
            fresh += 1

    # the remapped list describes the *current* model, so it must match the current
    # optimizer's group, not the checkpoint's
    n_current = len(optimizer.param_groups[0]["params"])
    if len(new_ids) != n_current:
        raise ConfigError(
            f"the model has {len(new_ids)} trainable parameter(s) but its optimizer's "
            f"first group has {n_current}; refusing to guess")
    groups[0]["params"] = new_ids
    saved["param_groups"] = groups
    return saved


def run_training(config: TrainConfig, dataset: DecisionDataset,
                 teacher: TeacherLabelStore, *, dry_run: bool = False) -> Dict[str, object]:
    """Run (or, with ``dry_run``, only plan) one training job."""
    config.validate()
    if dry_run:
        return plan(config, dataset, teacher)
    if len(dataset) == 0:
        raise ConfigError("dataset is empty; nothing to train on")

    os.makedirs(config.out_dir, exist_ok=True)
    ledger = os.path.join(config.out_dir, "ledger.jsonl")
    log_path = os.path.join(config.out_dir, "train_log.jsonl")
    meta_path = os.path.join(config.out_dir, "run_meta.json")
    resume_path = config.resume or os.path.join(config.out_dir, "resume.pt")

    lr_calibration: Optional[Dict[str, object]] = None
    if config.lr_calibrate:
        lr_calibration = calibrate_learning_rate(config, dataset, verbose=True)
        config = replace(config, lr=float(lr_calibration["chosen"]))
        print(f"[train] lr calibrated to {config.lr:g} ({lr_calibration['reason']})",
              flush=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # checked before .to(): a CPU-only torch build raises from inside .to("cuda")
    _check_device_request(config.device, allow_cpu=config.allow_cpu)
    model = build_decision_model(config).to(config.device)
    # Start the learnable prior multiplier where a held-out split says it should be.
    # Only the initialisation: it stays a Parameter and training moves it from here.
    if config.prior_init != 1.0:
        prior = getattr(getattr(model, "head", None), "prior_weight", None)
        if prior is None:
            raise ConfigError(
                f"--prior-init {config.prior_init} was given but this model has no "
                "learnable Turner-prior weight (use_turner_prior is off, or the head "
                "predates it); refusing to ignore the flag silently")
        with torch.no_grad():
            prior.fill_(float(config.prior_init))
        print(f"[train] Turner prior initialised to {config.prior_init:g} "
              f"(learnable from here)", flush=True)
    resolved_device = _assert_device(model, config.device, allow_cpu=config.allow_cpu)
    n_params = sum(p.numel() for p in model.parameters())
    n_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.lr, weight_decay=config.weight_decay,
                                  betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda st: min(1.0, (st + 1) / max(1, config.warmup_steps)))

    weights = config.objective_weights()
    start_step = 0
    coverage: Optional[Dict[str, object]] = None
    resume_note: Optional[Dict[str, object]] = None
    if os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        # Resuming must survive a parameter that was added *after* the checkpoint was
        # written, but must not hide a genuine mismatch.  `strict=True` rejected the
        # added Turner-prior multiplier and killed nine running arms on restart;
        # `strict=False` alone would also swallow a real architecture change.  So the
        # missing keys are checked against an explicit allow-list of parameters whose
        # initialisation is the correct default, and *unexpected* keys always fail.
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        defaultable = {"head.prior_weight"}
        unexplained = [k for k in missing if k not in defaultable]
        if unexpected or unexplained:
            raise ConfigError(
                f"cannot resume from {resume_path}: the checkpoint does not match this "
                f"model. unexpected keys={list(unexpected)}; missing keys that are not "
                f"known-defaultable={unexplained} (known-defaultable: {sorted(defaultable)})")
        if missing:
            resume_note = {
                "missing_keys_defaulted": sorted(missing),
                "why": ("added after this checkpoint was written; their initialisation "
                        "is the documented default, so the resumed trajectory is the "
                        "same as if training had continued uninterrupted"),
            }
            print(f"[train] resume: defaulted {sorted(missing)} "
                  f"(added after this checkpoint)", flush=True)
        saved_opt = _pad_optimizer_groups(
            state["optimizer"], optimizer,
            old_state_keys=list(state["model"].keys()),
            current_names=[n for n, p in model.named_parameters() if p.requires_grad],
            defaultable=sorted(defaultable))
        optimizer.load_state_dict(saved_opt)
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        coverage = state.get("gradient_coverage")
        print(f"[train] resumed from {resume_path} at step {start_step}", flush=True)

    meta = {
        "run_id": f"{config.arm}-{int(time.time())}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "spec_reference": (".trae/specs/build-rna-ss-decision-model/spec.md "
                           "§5.2 / §5.7 / §5.8"),
        "config": config.as_dict(),
        "model": {"n_params": n_params, "n_learnable_params": n_learnable,
                  "tiny": config.tiny, "encoder_size": config.encoder_size},
        "data": dataset.describe(),
        "teacher": teacher.meta(),
        "lr_calibration": lr_calibration,
        "objective_terms_active": [name for name, value in
                                   (("nll", weights.lambda_nll),
                                    ("distill", weights.lambda_distill),
                                    ("rlcd", weights.lambda_rlcd),
                                    ("cal", weights.lambda_cal)) if value != 0.0],
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "host": socket.gethostname(),
                        "device": config.device, "resolved_device": resolved_device,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "gpu_name": (torch.cuda.get_device_name(torch.cuda.current_device())
                                     if torch.cuda.is_available() else "cpu")},
        "honesty": dict(HONESTY),
        "implementation_notes": {
            "neginf_sentinel": ("head scores are -inf on illegal pairs; replaced by "
                                f"{NEG_BIG} before the objective because (-inf)*0.0 is NaN"),
            "zero_init_dead_gradient": ("with the zero-initialised MLP_T last layer the "
                                        "step-0 backward gives an exactly zero gradient to "
                                        "the encoder and inner MLP_T layers; coverage is "
                                        "asserted after the first optimizer step"),
        },
        "gradient_coverage": None,
        "resume_note": resume_note,
        "final_loss": None,
        "steps_completed": start_step,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)

    _ledger_row(ledger, {"status": "start", "arm": config.arm, "steps": config.steps,
                         "lr": config.lr, "batch_size": config.batch_size,
                         "resume_from": start_step,
                         "active_terms": meta["objective_terms_active"],
                         "teacher": teacher.name,
                         "data_version": dataset.version})

    stream = _batch_stream(dataset, config, start_step=start_step)
    running: List[float] = []
    running_terms: Dict[str, float] = {}
    start_wall = time.time()
    step = start_step
    final_loss: Optional[float] = None

    try:
        while step < config.steps:
            batch = next(stream)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective_terms(
                model, batch, weights, distill_kind=config.distill_kind,
                reward=config.rlcd_reward, beta=config.beta,
                n_bins=config.n_bins, tau=config.soft_ece_tau,
                nll_normalization=config.nll_normalization,
                cascade=config.head == "cascade",
                cascade_weights=config.layered_weights(),
                cascade_block_size=config.cascade_block_size)
            value = _finite(loss.detach(), f"loss at step {step + 1}")
            loss.backward()
            norm = _grad_norm(model.parameters())
            if not math.isfinite(norm):
                raise DivergenceError(
                    f"gradient norm at step {step + 1} is {norm}; run diverged")

            # Coverage is asserted on the *second* backward pass: the first one is
            # dead for the encoder because the zero-initialised MLP_T last layer
            # blocks the gradient (see the module docstring).  After one optimizer
            # step the last layer is non-zero and coverage is complete.
            if coverage is None and step >= 1:
                coverage = assert_gradient_coverage(model)
                meta["gradient_coverage"] = coverage

            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running.append(value)
            for name, term in terms.items():
                running_terms[name] = running_terms.get(name, 0.0) + term
            final_loss = value

            if config.log_every and step % config.log_every == 0:
                n = max(1, len(running))
                row = {"ts": time.time(), "step": step, "lr": scheduler.get_last_lr()[0],
                       "loss": sum(running) / n,
                       "terms": {k: v / n for k, v in running_terms.items()},
                       "grad_norm": norm}
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"[{config.arm}] step {step}/{config.steps} loss={row['loss']:.4f} "
                      f"terms={ {k: round(v, 4) for k, v in row['terms'].items()} } "
                      f"gnorm={norm:.3f}", flush=True)
                running, running_terms = [], {}

            if config.save_every and step % config.save_every == 0:
                _save_resume(resume_path, model, optimizer, scheduler, step, config, coverage)
                print(f"[{config.arm}] checkpoint written {resume_path} (step {step})",
                      flush=True)

            if config.max_hours and (time.time() - start_wall) / 3600.0 > config.max_hours:
                print(f"[{config.arm}] max_hours reached at step {step}", flush=True)
                break
    except DivergenceError:
        _save_resume(resume_path, model, optimizer, scheduler, step, config, coverage)
        _ledger_row(ledger, {"status": "diverged", "arm": config.arm, "step": step,
                             "lr": config.lr})
        meta["final_loss"] = final_loss
        meta["steps_completed"] = step
        meta["status"] = "diverged"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        raise

    if coverage is None:
        # Very short runs (a single step) never reach the second backward pass, so
        # the dead-gradient window is still open; assert connectivity only.
        coverage = assert_gradient_coverage(model, require_nonzero=False)
        meta["gradient_coverage"] = coverage
        meta["gradient_coverage_note"] = ("require_nonzero=False: the run was too short "
                                         "to leave the zero-init dead-gradient window")

    _save_resume(resume_path, model, optimizer, scheduler, step, config, coverage)
    meta["final_loss"] = final_loss
    meta["steps_completed"] = step
    meta["status"] = "completed"
    meta["wall_seconds"] = time.time() - start_wall
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    _ledger_row(ledger, {"status": "completed", "arm": config.arm, "step": step,
                         "lr": config.lr, "final_loss": final_loss})

    return {"dry_run": False, "final_loss": final_loss, "steps_completed": step,
            "run_meta": meta, "out_dir": config.out_dir, "ledger": ledger}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="decision")
    p.add_argument("--out", default="", help="run directory (required unless --dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the configuration and print the plan; no training")
    # data
    p.add_argument("--synthetic", type=int, default=0,
                   help="generate N synthetic examples instead of reading --data")
    p.add_argument("--length", type=int, default=24)
    p.add_argument("--data", default="", help="JSONL with {seq, structure|pairs}")
    p.add_argument("--teacher-dir", default="",
                   help="soft labels written by the rnajepa.distill driver")
    p.add_argument("--teacher-seed", type=int, default=0)
    # schedule
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--resume", default="")
    # objective
    p.add_argument("--lambda-nll", type=float, default=1.0)
    p.add_argument("--lambda-distill", type=float, default=1.0)
    p.add_argument("--lambda-rlcd", type=float, default=1.0)
    p.add_argument("--lambda-cal", type=float, default=1.0)
    p.add_argument("--distill-kind", default="kl", choices=["kl", "l2"])
    p.add_argument("--rlcd-reward", default="brier", choices=["brier", "log"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--nll-normalization", default="sum", choices=["sum", "length"],
                   help="'sum' = historical log Z - sum s_ij (O(L)); 'length' = that "
                        "divided by L, so the CRF term is comparable to the per-pair-mean "
                        "distill/RLCD/cal terms (with 'sum' they were <3%% of the loss).")
    # head / layered objective
    p.add_argument("--head", default="flat", choices=["flat", "cascade"],
                   help="'flat' = single L x L head with the four-term objective; "
                        "'cascade' = hierarchical head with the layered objective "
                        "(L0 block pairs / L1 helices / L2 base pairs), which requires "
                        "the differentiable gate.")
    p.add_argument("--cascade-block-size", type=int, default=8)
    p.add_argument("--cascade-top-k", type=int, default=2,
                   help="only used for the FLOPs accounting and the hard-path ablation; "
                        "the gated forward selects by a learned threshold, not by k")
    p.add_argument("--cascade-d-z", type=int, default=128)
    p.add_argument("--cascade-hidden", type=int, default=64)
    p.add_argument("--no-cascade-soft-gate", dest="cascade_soft_gate",
                   action="store_false", default=True,
                   help="disable the differentiable gate -- refused when head=cascade")
    p.add_argument("--cascade-gate-theta", type=float, default=0.0)
    p.add_argument("--cascade-gate-tau", type=float, default=0.5)
    p.add_argument("--cascade-l0", type=float, default=1.0)
    p.add_argument("--cascade-l1", type=float, default=1.0)
    p.add_argument("--cascade-l2", type=float, default=1.0)
    p.add_argument("--cascade-sparse", type=float, default=0.1)
    p.add_argument("--cascade-miss-cost", type=float, default=20.0,
                   help="how much more a missed ground-truth block pair costs than a "
                        "false positive in the L0 term; this is what drives L0 recall "
                        "towards the P6 gate of 0.98")
    # model
    p.add_argument("--tiny", action="store_true", help="CPU-sized model (tests / smoke)")
    p.add_argument("--d-z", type=int, default=128,
                   help="width of the permutation-invariant pair representation z_ij. "
                        "The flat head's capacity is dominated by "
                        "PairRepresentation.proj = Linear(3*d_model, d_z), so this is the "
                        "main scale knob.  Measured: d_z=128/hidden=64 gives 519,503 "
                        "trainable parameters against a 1280-dim frozen RiNALMo input -- "
                        "the default is small enough to be the ceiling on F1.")
    p.add_argument("--hidden", type=int, default=64,
                   help="hidden width of the Turner-residual MLP (see --d-z)")
    p.add_argument("--encoder-size", default="150M", choices=["35M", "150M", "650M"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--embedding-dir", default="",
                   help="directory of frozen embeddings; trains the head only")
    p.add_argument("--head-chunk-size", type=int, default=64,
                   help="column-chunk width for the head's pair tensors (0 = none); "
                        "lower it when 3*d_model makes the chunk transient too big")
    p.add_argument("--scorer", default="mlp", choices=["mlp", "resnet2d"],
                   help="pair-scorer architecture (Plan B: resnet2d)")
    p.add_argument("--resnet-blocks", type=int, default=4)
    p.add_argument("--embedding-d-model", type=int, default=0,
                   help="hidden size of the cached embeddings (required with "
                        "--embedding-dir)")
    p.add_argument("--prior-init", type=float, default=1.0,
                   help="initial value of the head's learnable Turner-prior "
                        "multiplier (1.0 = the historical hard-coded behaviour).  The "
                        "weight is learnable either way; this only sets where the "
                        "search starts.  Select it on a held-out split -- "
                        "tools/probe_prior_weight.py measures the optimum without any "
                        "training, and it differs per arm (0.5 for the frozen-RiNALMo "
                        "head, ~1 for the from-scratch encoder).")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit the CPU path (tests only; a real run must use a GPU)")
    # lr calibration
    p.add_argument("--lr-calibrate", action="store_true")
    p.add_argument("--lr-candidates", default="1e-4,5e-5,1e-5")
    p.add_argument("--lr-probe-steps", type=int, default=5)
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip, seed=args.seed, log_every=args.log_every,
        save_every=args.save_every, max_hours=args.max_hours,
        lambda_nll=args.lambda_nll, lambda_distill=args.lambda_distill,
        lambda_rlcd=args.lambda_rlcd, lambda_cal=args.lambda_cal,
        distill_kind=args.distill_kind, rlcd_reward=args.rlcd_reward, beta=args.beta,
        nll_normalization=args.nll_normalization,
        head=args.head, cascade_block_size=args.cascade_block_size,
        cascade_top_k=args.cascade_top_k, cascade_d_z=args.cascade_d_z,
        cascade_hidden=args.cascade_hidden, cascade_soft_gate=args.cascade_soft_gate,
        cascade_gate_theta=args.cascade_gate_theta,
        cascade_gate_tau=args.cascade_gate_tau,
        cascade_l0=args.cascade_l0, cascade_l1=args.cascade_l1,
        cascade_l2=args.cascade_l2, cascade_sparse=args.cascade_sparse,
        cascade_miss_cost=args.cascade_miss_cost,
        tiny=args.tiny, encoder_size=args.encoder_size, device=args.device,
        d_z=args.d_z, hidden=args.hidden,
        allow_cpu=args.allow_cpu,
        out_dir=args.out, resume=args.resume, arm=args.arm,
        embedding_dir=args.embedding_dir,
        embedding_d_model=args.embedding_d_model,
        head_chunk_size=args.head_chunk_size,
        scorer=args.scorer, resnet_blocks=args.resnet_blocks,
        prior_init=args.prior_init,
        lr_calibrate=args.lr_calibrate,
        lr_candidates=tuple(float(x) for x in args.lr_candidates.split(",") if x.strip()),
        lr_probe_steps=args.lr_probe_steps,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.out and not args.dry_run:
        print("FATAL: --out is required unless --dry-run", file=sys.stderr)
        return 2
    config = _config_from_args(args)
    try:
        if args.synthetic:
            dataset = make_synthetic_dataset(args.synthetic, args.length, seed=args.seed,
                                             teacher_seed=args.teacher_seed)
        elif args.data:
            teacher_store = (TeacherLabelStore.from_dir(args.teacher_dir)
                             if args.teacher_dir else TeacherLabelStore.from_mock(args.teacher_seed))
            # Scope the store to the split being trained.  Loading every shard
            # would (a) hold ~14 GB of unused embeddings and (b) try to read shards
            # that a concurrent extraction run is still writing.
            _split = os.path.basename(args.data).split(".")[0]
            embedding_store = (EmbeddingStore.from_dir(args.embedding_dir, split=_split)
                               if args.embedding_dir else None)
            dataset = load_dataset_from_jsonl(
                args.data, teacher_store, mock_seed=args.teacher_seed,
                need_teacher_probs=float(args.lambda_distill) != 0.0,
                strict_teacher=bool(args.teacher_dir),
                embedding_store=embedding_store)
        else:
            print("FATAL: give --synthetic N or --data FILE", file=sys.stderr)
            return 2
        teacher = (TeacherLabelStore.from_dir(args.teacher_dir) if args.teacher_dir
                   else TeacherLabelStore.from_mock(args.teacher_seed))
        result = run_training(config, dataset, teacher, dry_run=args.dry_run)
    except (ConfigError, DivergenceError, GradientCoverageError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result.get("dry_run"):
        print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        print(f"[{config.arm}] DONE steps={result['steps_completed']} "
              f"final_loss={result['final_loss']} out={result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
