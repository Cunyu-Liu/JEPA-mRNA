"""Teacher throughput benchmark and ensemble verification (Task 11, spec §10.2).

Why this module exists
----------------------
Spec §10.2 identifies teacher soft-label generation as a **new compute bottleneck**:
the pretraining corpus is ~36M unlabelled sequences (spec §5.7) and every one of
them needs a partition-function call from the System-2 teacher.  Before committing
to that, the throughput has to be measured per length bucket and extrapolated --
otherwise "distil the whole corpus" is a guess.

**Honesty requirement (read before quoting any number).**  ViennaRNA /
RNAstructure / LinearPartition are *not installed and cannot be downloaded in this
environment* (``rnajepa.distill.ThermodynamicTeacher`` raises
``ThermodynamicUnavailableError``; ``scripts/deploy_teachers.sh`` reports the same).
Every number this module can produce here therefore comes from
:class:`rnajepa.distill.MockTeacher`, which is a **synthetic** teacher, not a
physical model.  The report says so in three places:

* ``teacher.is_mock`` is ``True``;
* ``extrapolation.basis`` is ``"mock_teacher_throughput"`` (never
  ``"measured"``/``"real"``);
* ``extrapolation.warning`` states in words that the extrapolation must not be
  quoted as real tool throughput.

Asking for a real teacher raises rather than silently falling back to the mock.

Contents
--------
* :func:`benchmark_teacher` -- sequences/second per length bucket;
* :func:`extrapolate_corpus` -- hours/days for the full corpus, labelled by basis;
* :func:`verify_ensemble_assembly` -- ``p^teacher = mean(three teachers)``;
* :func:`probability_self_consistency` -- probability-matrix sanity plus the exact
  ``d log Z / d c = E[|M|] = sum_ij p_hat_ij`` identity from the Gibbs framework;
* :func:`format_report` / :func:`main` -- CLI.

Run:  python eval/ss/teacher_throughput.py --mock --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rnajepa.distill import (  # noqa: E402
    MockTeacher,
    ThermodynamicTeacher,
    ThermodynamicUnavailableError,
    assemble_teacher,
    sha256_hex,
)
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402

__all__ = [
    "DEFAULT_BUCKETS",
    "CORPUS_SIZE",
    "MOCK_BASIS",
    "TeacherUnavailableError",
    "length_buckets",
    "benchmark_teacher",
    "extrapolate_corpus",
    "verify_ensemble_assembly",
    "probability_self_consistency",
    "format_report",
    "main",
]

#: Bucket upper edges (spec §7.1 asks for per-length-bucket speed reporting).
DEFAULT_BUCKETS: Tuple[int, ...] = (32, 64, 128, 256, 512)

#: ~36M sequences: the size of the pretraining corpus (spec §5.7 / §10.1).
CORPUS_SIZE = 36_000_000

#: The only extrapolation basis obtainable in this environment.
MOCK_BASIS = "mock_teacher_throughput"


class TeacherUnavailableError(RuntimeError):
    """Raised when a real teacher is requested but cannot be used here."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_sequences(length: int, n: int, seed: int) -> List[str]:
    rng = np.random.default_rng((seed, length))
    return ["".join(rng.choice(list("ACGU"))) for _ in range(n)]


def length_buckets(edges: Sequence[int] = DEFAULT_BUCKETS) -> List[Tuple[int, int]]:
    """``(lo, hi]`` buckets from the upper edges, starting at 0."""
    bounds = [0] + [int(e) for e in edges]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _resolve_teacher(name: str, seed: int):
    """Return a teacher object; a real tool raises instead of degrading to mock."""
    if name == "mock":
        return MockTeacher(seed=seed)
    teacher = ThermodynamicTeacher(tool=name)
    try:
        teacher.predict_probs("ACGUACGU")
    except ThermodynamicUnavailableError as exc:  # re-raise with context
        raise TeacherUnavailableError(
            f"teacher '{name}' is not available in this environment: {exc}"
        ) from exc
    return teacher


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------
def benchmark_teacher(teacher, *, buckets: Sequence[int] = DEFAULT_BUCKETS,
                      n_per_bucket: int = 3, repeats: int = 1, seed: int = 0,
                      warmup: int = 0) -> Dict[str, object]:
    """Measure sequences/second per length bucket.

    Each bucket is represented by sequences of its **upper edge** length (a
    conservative choice: the longest sequence in the bucket).  ``repeats`` runs
    the bucket more than once so the spread is visible; ``warmup`` sequences are
    executed and discarded first (the same protocol rule the speed gate uses,
    spec §7.4).

    Returns a report with per-bucket ``seq_per_sec`` and an overall rate, plus the
    provenance of the teacher (``is_mock``).
    """
    if n_per_bucket <= 0:
        raise ValueError("n_per_bucket must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    rows: List[Dict[str, object]] = []
    total_sequences = 0
    total_seconds = 0.0
    for lo, hi in length_buckets(buckets):
        length = int(hi)
        sequences = _random_sequences(length, n_per_bucket, seed)
        for sequence in sequences[:warmup]:
            teacher.predict_probs(sequence)
        per_repeat: List[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            for sequence in sequences:
                teacher.predict_probs(sequence)
            per_repeat.append((time.perf_counter() - started) / len(sequences))
        mean_sec = statistics.fmean(per_repeat)
        total_sequences += len(sequences) * repeats
        total_seconds += mean_sec * len(sequences) * repeats
        rows.append({
            "lo": lo, "hi": hi, "length": length, "n_sequences": len(sequences),
            "repeats": repeats,
            "sec_per_seq": mean_sec,
            "seq_per_sec": (1.0 / mean_sec) if mean_sec > 0 else float("inf"),
            "sec_per_seq_runs": per_repeat,
            "sec_per_seq_min": min(per_repeat),
            "sec_per_seq_max": max(per_repeat),
        })

    overall = (total_sequences / total_seconds) if total_seconds > 0 else float("inf")
    name = getattr(teacher, "name", type(teacher).__name__)
    return {
        "teacher": name,
        "teacher_version_lock": getattr(teacher, "version_lock", None),
        "is_mock": name == "mock",
        "device": platform.platform(),
        "buckets": rows,
        "overall_seq_per_sec": overall,
        "n_timed_sequences": total_sequences,
        "total_seconds": total_seconds,
        "protocol": ("per-bucket mean over `repeats`; each bucket timed at its upper-edge "
                     "length; warmup sequences discarded"),
    }


def extrapolate_corpus(benchmark: Dict[str, object], *,
                       corpus_size: int = CORPUS_SIZE,
                       mean_length: Optional[float] = None) -> Dict[str, object]:
    """Extrapolate the wall-clock time for the full corpus.

    The default estimate uses the benchmark's **overall** rate, which implicitly
    assumes the corpus length distribution matches the benchmark mix.  Passing
    ``mean_length`` instead picks the bucket whose length is closest to it, which
    is the better estimate when a real length distribution is known.

    ``basis`` is ``"mock_teacher_throughput"`` whenever the benchmark came from the
    mock teacher; the warning says so explicitly so the number cannot be quoted as
    real tool throughput by accident.
    """
    is_mock = bool(benchmark.get("is_mock"))
    basis = MOCK_BASIS if is_mock else f"measured:{benchmark.get('teacher')}"

    if mean_length is not None:
        rows = benchmark["buckets"]
        chosen = min(rows, key=lambda r: abs(float(r["length"]) - float(mean_length)))
        rate = float(chosen["seq_per_sec"])
        rate_source = f"bucket {chosen['lo']}-{chosen['hi']} (length {chosen['length']})"
    else:
        rate = float(benchmark["overall_seq_per_sec"])
        rate_source = "overall rate across all buckets"

    seconds = (float(corpus_size) / rate) if rate > 0 else float("inf")
    warning = None
    if is_mock:
        warning = (
            "MOCK-BASED EXTRAPOLATION -- DO NOT QUOTE AS REAL TOOL THROUGHPUT. "
            "ViennaRNA / RNAstructure / LinearPartition are not installed in this "
            "environment, so these seconds/sequence come from the synthetic "
            "MockTeacher, which is not a physical model. Re-run "
            "scripts/deploy_teachers.sh --check on the cluster and re-benchmark "
            "before committing to a soft-label budget."
        )
    return {
        "corpus_size": int(corpus_size),
        "basis": basis,
        "rate_source": rate_source,
        "seq_per_sec": rate,
        "seconds": seconds,
        "hours": seconds / 3600.0,
        "days": seconds / 86400.0,
        "cpu_hours": (seconds / 3600.0),
        "warning": warning,
        "is_extrapolation": True,
    }


# ---------------------------------------------------------------------------
# ensemble verification (spec §5.7)
# ---------------------------------------------------------------------------
def verify_ensemble_assembly(teachers: Sequence[object], seq: str,
                             mask=None, atol: float = 1e-12) -> Dict[str, object]:
    """Verify ``p^teacher = mean(p^ViennaRNA, p^RNAstructure, p^LinearPartition)``.

    Checks that :func:`rnajepa.distill.assemble_teacher` reproduces the arithmetic
    mean of the member matrices (which is the definition the spec fixes), and that
    each member is itself a valid probability matrix on the candidate pairs.  With
    the mock teacher this validates the *assembly code path*; it says nothing about
    the physical teachers.
    """
    if len(teachers) == 0:
        raise ValueError("need at least one teacher")
    if mask is None:
        mask = valid_pair_mask(seq)
    matrices = [np.asarray(t.predict_probs(seq), dtype=np.float64) for t in teachers]
    stacked = np.stack(matrices, axis=0)
    expected = stacked.mean(axis=0)
    assembled = np.asarray(assemble_teacher(matrices), dtype=np.float64)
    deviation = float(np.max(np.abs(assembled - expected)))
    members = []
    for teacher, matrix in zip(teachers, matrices):
        check = probability_self_consistency(matrix, mask)
        members.append({
            "teacher": getattr(teacher, "name", type(teacher).__name__),
            "valid_probability_matrix": bool(check["passes_structural"]),
            "max_on_masked": float(matrix[np.triu(mask, k=1)].max()) if mask.any() else 0.0,
            "min_on_masked": float(matrix[np.triu(mask, k=1)].min()) if mask.any() else 0.0,
        })
    return {
        "n_teachers": len(teachers),
        "definition": "p^teacher = mean(member matrices)",
        "matches_mean": bool(deviation <= atol),
        "max_abs_deviation": deviation,
        "atol": atol,
        "members": members,
        "note": ("assembly code path verified; with mock teachers this is NOT evidence "
                 "about the physical teachers' agreement"),
    }


# ---------------------------------------------------------------------------
# probability self-consistency
# ---------------------------------------------------------------------------
def probability_self_consistency(probs, mask, *, scores=None, eps: float = 1e-4,
                                 rtol: float = 1e-3, atol: float = 1e-6
                                 ) -> Dict[str, object]:
    """Structural + exact-identity checks on a pair-probability matrix.

    Structural checks (always run):

    * every entry lies in ``[0, 1]``;
    * the matrix is symmetric;
    * the diagonal is zero and every non-candidate entry is exactly zero.

    Exact identity (run when ``scores`` is supplied): the Gibbs framework gives
    ``d log Z / d c = E[|M|] = sum_{i<j} p_hat_ij`` for a uniform shift ``c`` of all
    legal pair scores, so the finite difference of ``log Z`` must reproduce the sum
    of the marginals.  This is the strongest available check that ``p_hat`` really
    is the marginal of the distribution whose ``Z`` was computed -- a plain
    "values are in [0,1]" check would not catch a wrong recursion.
    """
    p = np.asarray(probs, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    upper = np.triu(mask_bool, k=1)
    diag = np.diag(p)
    outside = p[~mask_bool]

    report: Dict[str, object] = {
        "in_unit_interval": bool(np.all(p >= 0.0) and np.all(p <= 1.0)),
        "symmetric": bool(np.allclose(p, p.T, atol=0.0, rtol=0.0)),
        "zero_diagonal": bool(np.all(diag == 0.0)),
        "zero_outside_mask": bool(np.all(outside == 0.0)),
        "n_candidate_pairs": int(upper.sum()),
    }
    report["passes_structural"] = bool(
        report["in_unit_interval"] and report["symmetric"]
        and report["zero_diagonal"] and report["zero_outside_mask"]
    )

    if scores is not None:
        s = np.asarray(scores, dtype=np.float64)
        log_z, p_hat = inside_outside(s, mask_bool)
        expected_pairs = float(np.triu(p_hat, k=1).sum())
        shifted = s + eps * mask_bool.astype(np.float64)
        log_z_shifted, _ = inside_outside(shifted, mask_bool)
        derivative = (log_z_shifted - log_z) / eps
        abs_diff = abs(derivative - expected_pairs)
        report.update({
            "logZ": log_z,
            "expected_pairs_sum_marginals": expected_pairs,
            "logZ_shift_derivative": derivative,
            "identity_abs_diff": abs_diff,
            "identity_ok": bool(abs_diff <= atol + rtol * max(1.0, abs(expected_pairs))),
            "matches_supplied_probs": bool(np.allclose(p, p_hat, atol=1e-9)),
        })
        report["passes"] = bool(report["passes_structural"] and report["identity_ok"])
    else:
        report["passes"] = report["passes_structural"]
    return report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_report(report: Dict[str, object]) -> str:
    """Human-readable rendering of a benchmark + extrapolation report."""
    lines = [
        "teacher throughput benchmark",
        f"  teacher            : {report['benchmark']['teacher']}"
        f"{'  (MOCK -- not a physical model)' if report['benchmark']['is_mock'] else ''}",
        f"  version_lock       : {report['benchmark']['teacher_version_lock']}",
        f"  platform           : {report['benchmark']['device']}",
        "",
        f"  {'bucket':>12}  {'n':>3}  {'sec/seq':>10}  {'seq/s':>10}  {'seq/hour':>12}",
    ]
    for row in report["benchmark"]["buckets"]:
        lines.append(
            f"  {str(row['lo']) + '-' + str(row['hi']):>12}  {row['n_sequences']:>3}  "
            f"{row['sec_per_seq']:>10.6f}  {row['seq_per_sec']:>10.2f}  "
            f"{row['seq_per_sec'] * 3600:>12.0f}"
        )
    overall = report["benchmark"]["overall_seq_per_sec"]
    lines += [
        "",
        f"  overall            : {overall:.2f} seq/s  ({overall * 3600:.0f} seq/hour)",
        "",
        "  corpus extrapolation",
        f"    corpus size      : {report['extrapolation']['corpus_size']:,} sequences",
        f"    basis            : {report['extrapolation']['basis']}",
        f"    rate source      : {report['extrapolation']['rate_source']}",
        f"    estimated time   : {report['extrapolation']['hours']:.1f} h "
        f"({report['extrapolation']['days']:.1f} days)",
    ]
    if report["extrapolation"].get("warning"):
        lines += ["", f"  !! {report['extrapolation']['warning']}"]
    if "ensemble" in report:
        ens = report["ensemble"]
        lines += [
            "",
            "  teacher ensemble assembly",
            f"    members          : {ens['n_teachers']}",
            f"    matches mean     : {ens['matches_mean']} "
            f"(max |dev| {ens['max_abs_deviation']:.3e})",
        ]
    if "self_consistency" in report:
        sc = report["self_consistency"]
        lines += [
            "",
            "  probability self-consistency",
            f"    structural       : {sc['passes_structural']}",
            f"    identity ok      : {sc.get('identity_ok')} "
            f"(|diff| {sc.get('identity_abs_diff', float('nan')):.3e})",
        ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher throughput benchmark (Task 11)")
    p.add_argument("--teacher", default="mock",
                   choices=["mock", "viennarna", "rnastructure", "linearpartition"],
                   help="non-mock teachers raise here: they are not installed")
    p.add_argument("--mock", action="store_true",
                   help="alias for --teacher mock (the only runnable option here)")
    p.add_argument("--buckets", default=",".join(str(b) for b in DEFAULT_BUCKETS))
    p.add_argument("--n-per-bucket", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    p.add_argument("--mean-length", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="write the JSON report here")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    teacher_name = "mock" if args.mock else args.teacher
    buckets = tuple(int(b) for b in args.buckets.split(",") if b.strip())
    try:
        teacher = _resolve_teacher(teacher_name, args.seed)
    except TeacherUnavailableError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        print("This environment cannot benchmark the real teacher; use --mock, and "
              "re-run scripts/deploy_teachers.sh --check on the cluster.", file=sys.stderr)
        return 3

    bench = benchmark_teacher(teacher, buckets=buckets, n_per_bucket=args.n_per_bucket,
                              repeats=args.repeats, seed=args.seed, warmup=args.warmup)
    report: Dict[str, object] = {
        "benchmark": bench,
        "extrapolation": extrapolate_corpus(bench, corpus_size=args.corpus_size,
                                            mean_length=args.mean_length),
    }

    # Ensemble + self-consistency on a representative sequence.
    seq = _random_sequences(max(buckets), 1, args.seed)[0]
    mask = valid_pair_mask(seq)
    report["ensemble"] = verify_ensemble_assembly(
        [MockTeacher(seed=args.seed + k) for k in range(3)], seq, mask)
    # The identity needs scores, so it is checked on an explicit Gibbs instance.
    rng = np.random.default_rng(args.seed)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    _log_z, probs = inside_outside(scores, mask)
    report["self_consistency"] = probability_self_consistency(probs, mask, scores=scores)

    print(format_report(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
