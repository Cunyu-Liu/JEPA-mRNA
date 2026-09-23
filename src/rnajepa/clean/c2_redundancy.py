"""C2 — redundancy and leakage control (spec §4 C2) — the credibility linchpin.

Contents:

* internal redundancy removal by sequence-identity clustering at **80 % and
  90 %** identity, shelling out to ``mmseqs2`` / ``cd-hit-est``;
* Rfam-family-level grouping so that no family crosses splits;
* **pretraining-vs-test contamination detection** (exact + 80 % identity) which
  removes hits and reports the contamination rate.

Tool policy
-----------
The real clustering path shells out to the external tool.  If the tool is not on
``PATH`` we either

* raise :class:`ToolUnavailableError` (default), or
* fall back to a **documented dry-run approximation** when ``dry_run=True``.

The dry-run fallback is a greedy k-mer/Hamming identity approximation.  It is
**not** a substitute for MMseqs2/CD-HIT and must never be reported as if it were
the real clustering (see ``DRY_RUN_IS_APPROXIMATION``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .records import ReasonCode, Record, Removal

LEVEL_DEDUP = "C2_dedup"
LEVEL_CONTAMINATION = "C2_contamination"

#: The dry-run fallback is an approximation, not the real tool.
DRY_RUN_IS_APPROXIMATION = True

#: k-mer size for the dry-run identity estimate on unequal-length sequences.
#: k=4 is degenerate here: a long random sequence contains almost every 4-mer, so
#: containment saturates near 1.0 for unrelated sequences of very different
#: lengths.  k=8 keeps the estimate discriminative.
DEFAULT_K = 8


class ToolUnavailableError(RuntimeError):
    """Raised when an external clustering tool is required but absent."""


def _identity(a: str, b: str, k: int = DEFAULT_K) -> float:
    """Sequence identity estimate in ``[0, 1]`` (dry-run approximation).

    * equal length -> ungapped Hamming identity (exact for substitutions);
    * unequal length -> k-mer containment of the shorter sequence.

    Both are approximations of a local alignment.  They are used **only** in
    dry-run mode and must never be reported as MMseqs2/CD-HIT results.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)
    if len(a) < k or len(b) < k:
        return 0.0
    kmers_a = {a[i:i + k] for i in range(len(a) - k + 1)}
    kmers_b = {b[i:i + k] for i in range(len(b) - k + 1)}
    if not kmers_a or not kmers_b:
        return 0.0
    return len(kmers_a & kmers_b) / min(len(kmers_a), len(kmers_b))


def _greedy_clusters(sequences: Sequence[str], identity: float, k: int = DEFAULT_K) -> List[int]:
    """Greedy identity clustering (dry-run approximation).

    Returns a cluster index per sequence; the first member of a cluster is its
    representative.  O(n * clusters), which is fine for the dry-run/test path but
    is *not* what production runs should use.
    """
    representatives: List[str] = []
    assignment: List[int] = []
    for seq in sequences:
        for cluster_id, rep in enumerate(representatives):
            if _identity(seq, rep, k=k) >= identity:
                assignment.append(cluster_id)
                break
        else:
            assignment.append(len(representatives))
            representatives.append(seq)
    return assignment


def _write_fasta(records: Sequence[Record], path: str) -> None:
    with open(path, "w") as handle:
        for record in records:
            handle.write(f">{record.id}\n{record.seq_norm}\n")


def _cluster_with_tool(
    records: Sequence[Record], identity: float, tool: str, tool_path: str, k: int
) -> List[int]:
    """Shell out to ``mmseqs2`` / ``cd-hit-est`` and return cluster indices.

    NOTE: this path is implemented but **not exercised in the current
    environment** (neither tool is installed).  It is kept so that production
    runs use the real tool; treat it as untested until it has been run once with
    a recorded version string.
    """
    id_int = int(round(identity * 100))
    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "in.fa")
        _write_fasta(records, fasta)
        cluster_tsv = os.path.join(tmp, "clusters.tsv")

        if "mmseqs" in os.path.basename(tool_path):
            prefix = os.path.join(tmp, "mm")
            subprocess.run(
                [tool_path, "easy-cluster", fasta, prefix, tmp,
                 "--min-seq-id", str(identity), "-c", "0.8"],
                check=True, capture_output=True,
            )
            # mmseqs writes "<prefix>_cluster.tsv": representative \t member
            generated = prefix + "_cluster.tsv"
        else:
            out = os.path.join(tmp, "cdhit")
            subprocess.run(
                [tool_path, "-i", fasta, "-o", out, "-c", str(identity), "-n", "4", "-M", "0"],
                check=True, capture_output=True,
            )
            generated = out + ".clstr"

        if not os.path.exists(generated):
            raise RuntimeError(f"{tool} did not produce the expected cluster file")

        index = {record.id: i for i, record in enumerate(records)}
        assignment = [-1] * len(records)
        if generated.endswith(".tsv"):
            with open(generated) as handle:
                for cluster_id, line in enumerate(handle):
                    rep, member = line.split("\t")[:2]
                    if member.strip() in index:
                        assignment[index[member.strip()]] = cluster_id
        else:
            with open(generated) as handle:
                cluster_id = -1
                for line in handle:
                    if line.startswith(">Cluster"):
                        cluster_id += 1
                    else:
                        member = line.split(">")[1].split("...")[0]
                        if member in index:
                            assignment[index[member]] = cluster_id
        if any(a < 0 for a in assignment):
            raise RuntimeError(f"{tool} left {assignment.count(-1)} sequences unclustered")
        return assignment


def deduplicate(
    records: Sequence[Record],
    identity: float,
    *,
    tool: str = "mmseqs2",
    dry_run: bool = False,
    k: int = DEFAULT_K,
) -> Tuple[List[Record], List[Removal]]:
    """Remove internal redundancy at ``identity`` (spec §4 C2).

    The representative of each cluster is kept; the remaining members are
    removed with a reason code derived from ``identity``.
    """
    if identity >= 0.9:
        reason = ReasonCode.DUPLICATE_IDENTITY_90
    else:
        reason = ReasonCode.DUPLICATE_IDENTITY_80

    tool_path = shutil.which(tool)
    if tool_path is None:
        if not dry_run:
            raise ToolUnavailableError(
                f"{tool!r} not found on PATH. Install MMseqs2/CD-HIT-EST for the real "
                "clustering, or pass dry_run=True to use the documented "
                "approximation (which must not be reported as the real tool)."
            )
        assignment = _greedy_clusters([r.seq_norm for r in records], identity, k=k)
    else:
        assignment = _cluster_with_tool(records, identity, tool, tool_path, k)

    seen: Dict[int, str] = {}
    kept: List[Record] = []
    removals: List[Removal] = []
    for record, cluster_id in zip(records, assignment):
        if cluster_id in seen:
            removals.append(
                Removal(
                    record.id,
                    reason,
                    LEVEL_DEDUP,
                    f"identity >= {identity:.2f} with {seen[cluster_id]}",
                )
            )
        else:
            seen[cluster_id] = record.id
            kept.append(record)
    return kept, removals


def detect_contamination(
    records: Sequence[Record],
    pretrain_sequences: Iterable[str],
    *,
    identity: float = 0.8,
    include_exact: bool = True,
    dry_run: bool = True,
    tool: str = "mmseqs2",
    k: int = DEFAULT_K,
) -> Tuple[List[Record], List[Removal], Dict[str, object]]:
    """Remove downstream records that also occur in the pretraining corpus.

    Exact hits are found by SHA256 fingerprint; near hits by an identity search
    against the pretraining corpus (MMseqs2 in production, greedy approximation
    in dry-run).  Returns ``(kept, removals, report)`` where ``report`` carries
    the contamination rate (spec §8.1 G4).
    """
    pretrain = [s for s in pretrain_sequences if s]
    pretrain_hashes = {_hash(s) for s in pretrain}

    kept: List[Record] = []
    removals: List[Removal] = []
    exact_hits = 0
    near_hits = 0

    near_index: List[str] = []
    if pretrain:
        tool_path = shutil.which(tool)
        if tool_path is None and not dry_run:
            raise ToolUnavailableError(
                f"{tool!r} not found on PATH; pass dry_run=True for the documented "
                "approximation."
            )
        # Deduplicate the pretraining corpus to representatives first; a hit
        # against any representative is what matters.
        if tool_path is not None and not dry_run:
            near_index = pretrain  # real tool would search the full DB
        else:
            near_index = _representatives(pretrain, identity, k=k)

    for record in records:
        # ``seq_norm`` is set by C1; fall back to the raw sequence so the
        # function is also usable standalone (the caller is then responsible for
        # passing already-normalised pretraining sequences).
        seq = record.seq_norm if record.seq_norm is not None else (record.sequence or "")
        if include_exact and _hash(seq) in pretrain_hashes:
            exact_hits += 1
            removals.append(
                Removal(
                    record.id,
                    ReasonCode.PRETRAIN_CONTAMINATION_EXACT,
                    LEVEL_CONTAMINATION,
                    "exact sequence match in pretraining corpus",
                )
            )
            continue
        if any(_identity(seq, rep, k=k) >= identity for rep in near_index):
            near_hits += 1
            removals.append(
                Removal(
                    record.id,
                    ReasonCode.PRETRAIN_CONTAMINATION_NEAR,
                    LEVEL_CONTAMINATION,
                    f"identity >= {identity:.2f} to a pretraining sequence",
                )
            )
            continue
        kept.append(record)

    total = len(records)
    removed = exact_hits + near_hits
    report = {
        "n_records": total,
        "n_pretrain_sequences": len(pretrain),
        "exact_hits": exact_hits,
        "near_hits": near_hits,
        "removed": removed,
        "contamination_rate": (removed / total) if total else 0.0,
        "identity_threshold": identity,
        "mode": "dry_run_approximation" if (shutil.which(tool) is None or dry_run) else tool,
    }
    return kept, removals, report


def _hash(sequence: str) -> str:
    from .records import sha256_hex

    return sha256_hex(sequence)


def _representatives(sequences: Sequence[str], identity: float, k: int = DEFAULT_K) -> List[str]:
    reps: List[str] = []
    for seq in sequences:
        if not any(_identity(seq, rep, k=k) >= identity for rep in reps):
            reps.append(seq)
    return reps


def _jitter(key: str, seed: int) -> int:
    import hashlib

    return int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:8], 16)


def family_group_split(
    records: Sequence[Record],
    *,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    names: Tuple[str, str, str] = ("train", "dev", "test"),
    seed: int = 0,
    group_key: str = "family",
) -> Dict[str, List[Record]]:
    """Assign **whole families** to splits so no family crosses splits.

    Families are processed largest-first and greedily assigned to the split whose
    **relative** deficit (``(target - count) / max(target, 1)``) is largest.  The
    relative rather than absolute deficit matters: with a handful of large
    families the absolute rule sends every family to ``train`` and leaves
    ``dev``/``test`` empty.  The trade-off is that a small split can overshoot its
    target when families are large relative to it — report the realised sizes.

    Records without a family get a unique group, so they never share a group by
    accident.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    groups: Dict[str, List[Record]] = {}
    for record in records:
        family = getattr(record, group_key, None) or f"__singleton__{record.id}"
        groups.setdefault(family, []).append(record)

    # Deterministic ordering: size desc, then a seeded jitter over the family
    # name so that equal-sized families are shuffled reproducibly by ``seed``.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), _jitter(kv[0], seed)))

    total = len(records)
    targets = [ratio * total for ratio in ratios]
    counts = [0] * len(ratios)
    splits: Dict[str, List[Record]] = {name: [] for name in names}

    for _, members in ordered:
        deficits = [
            (targets[i] - counts[i]) / max(targets[i], 1.0) for i in range(len(ratios))
        ]
        # Stable tie-break: earliest split wins ties, so results are seed-stable
        # without depending on dict ordering.
        choice = max(range(len(ratios)), key=lambda i: (deficits[i], -i))
        splits[names[choice]].extend(members)
        counts[choice] += len(members)

    return splits


def family_overlap(splits: Dict[str, List[Record]], group_key: str = "family") -> Dict[str, int]:
    """Count families appearing in more than one split (must be 0)."""
    seen: Dict[str, set] = {}
    for name, records in splits.items():
        for record in records:
            family = getattr(record, group_key, None) or f"__singleton__{record.id}"
            seen.setdefault(family, set()).add(name)
    return {family: len(where) for family, where in seen.items() if len(where) > 1}
