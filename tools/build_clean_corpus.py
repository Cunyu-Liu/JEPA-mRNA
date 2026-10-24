#!/usr/bin/env python3
"""Build a clean JSONL corpus and its aligned, sharded teacher labels.

Each ``--input`` is ``JSONL {mfe,gt} [TEACHER_DIR]``.  MFE inputs require an
existing teacher directory.  GT inputs generate upper-triangular one-hot labels
from JSONL pairs and, when a directory is supplied, require exact agreement with
that existing teacher.  The final JSONL and teacher directory must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

GT_VERSION = "jsonl-pairs-upper-one-hot-v1"
BUILD_MANIFEST = "manifest.building.json"
STAT_NAMES = (
    "filtered_blocklist",
    "dropped_exact_duplicate_sequence",
    "conflicting_labels_kept_first",
    "conflicting_structure_kept_first",
    "conflicting_teacher_kept_first",
    "filtered_hash_sample",
)
OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
CLOSE_TO_OPEN = {value: key for key, value in OPEN_TO_CLOSE.items()}


class CorpusBuildError(RuntimeError):
    """Raised when corpus or teacher integrity cannot be proven."""


@dataclass(frozen=True)
class InputSpec:
    path: str
    teacher_kind: str
    teacher_dir: Optional[str] = None


def _sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequences_hash(sequences: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for index, sequence in enumerate(sequences):
        if index:
            digest.update(b"\n")
        digest.update(str(sequence).encode("utf-8"))
    return digest.hexdigest()


def _normalize_sequence(value: object, context: str) -> str:
    sequence = str(value).upper().replace("T", "U")
    if not sequence:
        raise CorpusBuildError(f"{context}: empty sequence")
    invalid = sorted(set(sequence) - set("ACGU"))
    if invalid:
        raise CorpusBuildError(
            f"{context}: sequence contains non-ACGU characters after T->U: {invalid}"
        )
    return sequence


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusBuildError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise CorpusBuildError(f"{path}:{lineno}: JSONL row must be an object")
            yield lineno, record


def _parse_structure(structure: str, context: str) -> List[Tuple[int, int]]:
    stacks: Dict[str, List[int]] = {char: [] for char in OPEN_TO_CLOSE}
    pairs: List[Tuple[int, int]] = []
    for index, char in enumerate(structure):
        if char in ".:":
            continue
        if char in OPEN_TO_CLOSE:
            stacks[char].append(index)
        elif char in CLOSE_TO_OPEN:
            opener = CLOSE_TO_OPEN[char]
            if not stacks[opener]:
                raise CorpusBuildError(
                    f"{context}: unmatched {char!r} at structure position {index}"
                )
            pairs.append((stacks[opener].pop(), index))
        else:
            raise CorpusBuildError(
                f"{context}: invalid structure character {char!r} at position {index}"
            )
    if any(stacks.values()):
        raise CorpusBuildError(f"{context}: unbalanced structure")
    return sorted(pairs)


def _prepare_record(raw: Dict[str, object], context: str) -> Dict[str, object]:
    if "seq" not in raw:
        raise CorpusBuildError(f"{context}: missing seq")
    record = dict(raw)
    sequence = _normalize_sequence(raw["seq"], context)
    length = len(sequence)
    has_pairs = "pairs" in raw
    structure = raw.get("structure")
    if not has_pairs and structure is None:
        raise CorpusBuildError(f"{context}: require pairs or structure")

    structure_pairs: Optional[List[Tuple[int, int]]] = None
    if structure is not None:
        structure_text = str(structure)
        if len(structure_text) != length:
            raise CorpusBuildError(
                f"{context}: structure length {len(structure_text)} != sequence length {length}"
            )
        structure_pairs = _parse_structure(structure_text, context)

    if has_pairs:
        value = raw["pairs"]
        if not isinstance(value, (list, tuple)):
            raise CorpusBuildError(f"{context}: pairs must be a list")
        pairs: List[Tuple[int, int]] = []
        occupied: Set[int] = set()
        seen_pairs: Set[Tuple[int, int]] = set()
        for pair_index, pair in enumerate(value):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise CorpusBuildError(f"{context}: invalid pair at index {pair_index}")
            i_raw, j_raw = pair
            if (isinstance(i_raw, bool) or isinstance(j_raw, bool)
                    or not isinstance(i_raw, (int, np.integer))
                    or not isinstance(j_raw, (int, np.integer))):
                raise CorpusBuildError(f"{context}: pair coordinates must be integers")
            i, j = sorted((int(i_raw), int(j_raw)))
            if i == j or i < 0 or j >= length:
                raise CorpusBuildError(
                    f"{context}: pair {(i_raw, j_raw)} is outside 0-based sequence bounds"
                )
            if (i, j) in seen_pairs:
                raise CorpusBuildError(f"{context}: duplicate pair {(i, j)}")
            if i in occupied or j in occupied:
                raise CorpusBuildError(f"{context}: a residue occurs in multiple pairs")
            seen_pairs.add((i, j))
            occupied.update((i, j))
            pairs.append((i, j))
        pairs.sort()
        if structure_pairs is not None and pairs != structure_pairs:
            raise CorpusBuildError(f"{context}: pairs and structure disagree")
    else:
        assert structure_pairs is not None
        pairs = structure_pairs

    record["seq"] = sequence
    record["pairs"] = [[i, j] for i, j in pairs]
    return record


def _pairs_digest(record: Dict[str, object]) -> bytes:
    pairs = record["pairs"]
    assert isinstance(pairs, list)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(pairs)))
    for pair in pairs:
        digest.update(struct.pack("<II", int(pair[0]), int(pair[1])))
    return digest.digest()


def _probability_matrix(value: object, length: int, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError(f"{context}: teacher matrix is not numeric") from exc
    if matrix.shape != (length, length):
        raise CorpusBuildError(
            f"{context}: teacher shape {matrix.shape} != {(length, length)}"
        )
    if not bool(np.isfinite(matrix).all()):
        raise CorpusBuildError(f"{context}: teacher matrix contains NaN or Inf")
    if bool((matrix < 0.0).any()) or bool((matrix > 1.0).any()):
        raise CorpusBuildError(f"{context}: teacher probabilities are outside [0, 1]")
    return np.ascontiguousarray(matrix)


def _probability_digest(matrix: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(struct.pack("<II", *matrix.shape))
    digest.update(np.ascontiguousarray(matrix).tobytes(order="C"))
    return digest.digest()


def _ground_truth_matrix(record: Dict[str, object], context: str) -> np.ndarray:
    sequence = str(record["seq"])
    matrix = np.zeros((len(sequence), len(sequence)), dtype=np.float64)
    pairs = record["pairs"]
    assert isinstance(pairs, list)
    for pair in pairs:
        matrix[int(pair[0]), int(pair[1])] = 1.0
    return _probability_matrix(matrix, len(sequence), context)


def _load_blocklists(paths: Sequence[Path]) -> Tuple[Set[str], List[Dict[str, object]]]:
    blocked: Set[str] = set()
    metadata: List[Dict[str, object]] = []
    for path in paths:
        before = _sha256_file(path)
        local: Set[str] = set()
        n_records = 0
        for lineno, record in _iter_jsonl(path):
            context = f"{path}:{lineno}"
            if "seq" not in record:
                raise CorpusBuildError(f"{context}: blocklist row is missing seq")
            local.add(_normalize_sequence(record["seq"], context))
            n_records += 1
        after = _sha256_file(path)
        if after != before:
            raise CorpusBuildError(f"{path}: input hash changed while reading")
        blocked.update(local)
        metadata.append({
            "path": str(path),
            "sha256": after,
            "sha256_before": before,
            "sha256_after": after,
            "n_records": n_records,
            "n_unique_sequences": len(local),
        })
    return blocked, metadata


def _select_hash_sample(specs: Sequence[InputSpec], sample_input: Optional[int],
                        sample_count: Optional[int], sample_seed: str
                        ) -> Optional[Tuple[int, Set[str], Dict[str, object]]]:
    if sample_input is None and sample_count is None:
        return None
    if (sample_input is None or sample_count is None
            or isinstance(sample_input, bool) or isinstance(sample_count, bool)
            or sample_input < 1 or sample_input > len(specs) or sample_count <= 0
            or not sample_seed):
        raise CorpusBuildError(
            "sampling requires --sample-input in input range, --sample-count > 0, "
            "and a non-empty --sample-seed"
        )
    index = sample_input - 1
    path = Path(specs[index].path)
    before = _sha256_file(path)
    candidates: List[Tuple[bytes, str]] = []
    seen: Set[str] = set()
    for lineno, record in _iter_jsonl(path):
        sequence = _normalize_sequence(record.get("seq", ""), f"{path}:{lineno}")
        if sequence in seen:
            continue
        seen.add(sequence)
        score = hashlib.sha256(f"{sample_seed}\0{sequence}".encode("utf-8")).digest()
        candidates.append((score, sequence))
    after = _sha256_file(path)
    if after != before:
        raise CorpusBuildError(f"{path}: input hash changed while selecting hash sample")
    if len(candidates) < sample_count:
        raise CorpusBuildError(
            f"{path}: only {len(candidates)} unique sequences, cannot select {sample_count}"
        )
    chosen = heapq.nsmallest(sample_count, candidates)
    selected = {sequence for _score, sequence in chosen}
    metadata: Dict[str, object] = {
        "sample_input": sample_input,
        "sample_count": sample_count,
        "sample_seed": sample_seed,
        "algorithm": "sha256(sample_seed + NUL + normalized_sequence), smallest scores",
        "n_candidates_unique": len(candidates),
        "n_selected_unique": len(selected),
        "selection_sha256": _sequences_hash(sorted(selected)),
        "input_sha256": after,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "selection_before_blocklist_and_dedup": True,
    }
    return index, selected, metadata


class TeacherShardReader:
    """Read one shard at a time, checking file, count, and sequence hashes."""

    def __init__(self, directory: Path, *, manifest_path: Optional[Path] = None,
                 context: str = "teacher") -> None:
        self.directory = directory
        self.manifest_path = manifest_path or directory / "manifest.json"
        self.context = context
        self.manifest_before = _sha256_file(self.manifest_path)
        try:
            with open(self.manifest_path, encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CorpusBuildError(f"{self.manifest_path}: malformed manifest: {exc}") from exc
        if not isinstance(self.manifest, dict):
            raise CorpusBuildError(f"{self.manifest_path}: manifest must be an object")
        shards = self.manifest.get("shards")
        if not isinstance(shards, list):
            raise CorpusBuildError(f"{self.manifest_path}: shards must be a list")
        seen_files: Set[str] = set()
        for position, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise CorpusBuildError(f"{self.manifest_path}: shard entries must be objects")
            index = shard.get("index", position)
            if isinstance(index, bool) or not isinstance(index, int) or index != position:
                raise CorpusBuildError(
                    f"{self.manifest_path}: shard indices must be contiguous and ordered"
                )
            filename = shard.get("file")
            if (not isinstance(filename, str) or not filename.endswith(".npz")
                    or Path(filename).name != filename):
                raise CorpusBuildError(f"{self.manifest_path}: unsafe shard filename {filename!r}")
            if filename in seen_files:
                raise CorpusBuildError(f"{self.manifest_path}: duplicate shard {filename}")
            seen_files.add(filename)
            if "n" in shard and (isinstance(shard["n"], bool)
                                 or not isinstance(shard["n"], int) or shard["n"] < 0):
                raise CorpusBuildError(f"{self.manifest_path}: invalid shard n for {filename}")
        self.shards: List[Dict[str, object]] = shards
        self._next_shard = 0
        self._sequences: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._offset = 0
        self.consumed = 0
        self.seq_hash_algorithm: Optional[str] = None

    def _check_sequence_hash(self, sequences: np.ndarray, expected: str, path: Path) -> None:
        algorithms = {
            "newline_join": lambda: _sequences_hash(sequences),
            "empty_join": lambda: hashlib.sha256(
                "".join(str(value) for value in sequences).encode("utf-8")).hexdigest(),
        }
        if self.seq_hash_algorithm is None:
            for algorithm, compute in algorithms.items():
                if compute() == expected:
                    self.seq_hash_algorithm = algorithm
                    break
            else:
                raise CorpusBuildError(f"{path}: sequence hash disagrees with manifest")
            return
        if algorithms[self.seq_hash_algorithm]() != expected:
            raise CorpusBuildError(
                f"{path}: sequence hash disagrees with manifest using "
                f"{self.seq_hash_algorithm}; mixed hash conventions are not allowed"
            )

    def _load_next_shard(self) -> bool:
        if self._next_shard >= len(self.shards):
            return False
        shard = self.shards[self._next_shard]
        filename = str(shard["file"])
        path = self.directory / filename
        before = _sha256_file(path)
        expected_hash = shard.get("sha256")
        if expected_hash is not None and before != expected_hash:
            raise CorpusBuildError(
                f"{path}: shard sha256 {before} != manifest sha256 {expected_hash}"
            )
        try:
            with np.load(path, allow_pickle=True) as data:
                if "sequences" not in data or "probs" not in data:
                    raise CorpusBuildError(f"{path}: require sequences and probs arrays")
                sequences = data["sequences"]
                probs = data["probs"]
        except CorpusBuildError:
            raise
        except Exception as exc:
            raise CorpusBuildError(f"{path}: cannot read teacher shard: {exc}") from exc
        after = _sha256_file(path)
        if after != before:
            raise CorpusBuildError(f"{path}: shard hash changed while reading")
        if sequences.ndim == 0 or probs.ndim == 0 or len(sequences) != len(probs):
            raise CorpusBuildError(f"{path}: sequences/probs lengths disagree")
        if "n" in shard and len(sequences) != shard["n"]:
            raise CorpusBuildError(
                f"{path}: shard contains {len(sequences)} rows, manifest says {shard['n']}"
            )
        if "seq_sha256" in shard:
            self._check_sequence_hash(sequences, str(shard["seq_sha256"]), path)
        self._sequences = sequences
        self._probs = probs
        self._offset = 0
        self._next_shard += 1
        return True

    def next(self) -> Tuple[str, np.ndarray]:
        while self._sequences is None or self._offset >= len(self._sequences):
            if not self._load_next_shard():
                raise StopIteration
        assert self._probs is not None
        raw_sequence = self._sequences[self._offset]
        context = f"{self.context} record {self.consumed}"
        sequence = _normalize_sequence(raw_sequence, context)
        matrix = _probability_matrix(self._probs[self._offset], len(sequence), context)
        self._offset += 1
        self.consumed += 1
        return sequence, matrix

    def finish(self) -> Dict[str, object]:
        try:
            sequence, _matrix = self.next()
        except StopIteration:
            pass
        else:
            raise CorpusBuildError(
                f"{self.context}: unexplained extra teacher record {self.consumed - 1} "
                f"for sequence {sequence!r}"
            )
        declared = self.manifest.get("n_sequences")
        if declared is not None and declared != self.consumed:
            raise CorpusBuildError(
                f"{self.manifest_path}: n_sequences={declared}, consumed {self.consumed}"
            )
        after = _sha256_file(self.manifest_path)
        if after != self.manifest_before:
            raise CorpusBuildError(f"{self.manifest_path}: manifest hash changed while reading")
        return {
            "teacher_manifest_sha256": after,
            "teacher_manifest_sha256_before": self.manifest_before,
            "teacher_manifest_sha256_after": after,
            "n_teacher_records": self.consumed,
            "n_teacher_shards": len(self.shards),
            "seq_hash_algorithm": self.seq_hash_algorithm,
        }


def _write_npz_member(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    with archive.open(info, mode="w", force_zip64=True) as member:
        np.save(member, array, allow_pickle=True)


def _write_shard(path: Path, sequences: List[str], probs: List[np.ndarray]) -> None:
    sequence_array = np.empty(len(sequences), dtype=object)
    sequence_array[:] = sequences
    probability_array = np.empty(len(probs), dtype=object)
    for index, matrix in enumerate(probs):
        probability_array[index] = matrix
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        _write_npz_member(archive, "sequences", sequence_array)
        _write_npz_member(archive, "probs", probability_array)


def _write_json(path: Path, value: Dict[str, object]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


class BuildConsumer:
    def __init__(self, jsonl_path: Path, teacher_dir: Path, shard_size: int) -> None:
        self.jsonl_path = jsonl_path
        self.teacher_dir = teacher_dir
        self.shard_size = shard_size
        self._jsonl = open(jsonl_path, "w", encoding="utf-8")
        self._sequences: List[str] = []
        self._probs: List[np.ndarray] = []
        self.shards: List[Dict[str, object]] = []
        self.n_records = 0

    def write(self, record: Dict[str, object], matrix: np.ndarray) -> None:
        self._jsonl.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")) + "\n")
        self._sequences.append(str(record["seq"]))
        self._probs.append(matrix)
        self.n_records += 1
        if len(self._sequences) == self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._sequences:
            return
        index = len(self.shards)
        filename = f"shard_{index:05d}.npz"
        path = self.teacher_dir / filename
        _write_shard(path, self._sequences, self._probs)
        self.shards.append({
            "index": index,
            "file": filename,
            "n": len(self._sequences),
            "sha256": _sha256_file(path),
            "seq_sha256": _sequences_hash(self._sequences),
        })
        self._sequences = []
        self._probs = []

    def finish(self) -> Dict[str, object]:
        self._flush()
        self._jsonl.close()
        output_hash = _sha256_file(self.jsonl_path)
        provisional = {
            "teacher": "pending-independent-verification",
            "version_lock": None,
            "shard_size": self.shard_size,
            "n_sequences": self.n_records,
            "shards": self.shards,
        }
        _write_json(self.teacher_dir / BUILD_MANIFEST, provisional)
        return {"output_jsonl_sha256": output_hash, "n_output_records": self.n_records}

    def abort(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.close()


class VerifyConsumer:
    """Independently stream final-format outputs and compare every kept row."""

    def __init__(self, jsonl_path: Path, teacher_dir: Path, *, role: str,
                 blocked: Set[str], expected_jsonl_hash: str) -> None:
        self.jsonl_path = jsonl_path
        self.role = role
        self.blocked = blocked
        self.expected_jsonl_hash = expected_jsonl_hash
        self.before = _sha256_file(jsonl_path)
        if self.before != expected_jsonl_hash:
            raise CorpusBuildError("output JSONL hash changed before independent verification")
        self._records = _iter_jsonl(jsonl_path)
        self.teacher = TeacherShardReader(
            teacher_dir, manifest_path=teacher_dir / BUILD_MANIFEST,
            context="output teacher",
        )
        self.n_records = 0
        self.blocklist_overlap = 0

    def _output_sequence(self, record: Dict[str, object]) -> str:
        if "seq" not in record:
            raise CorpusBuildError("independent verification: output row is missing seq")
        sequence = _normalize_sequence(record["seq"], "independent output verification")
        if self.role == "train" and sequence in self.blocked:
            self.blocklist_overlap += 1
        return sequence

    def write(self, expected_record: Dict[str, object], expected_matrix: np.ndarray) -> None:
        try:
            _lineno, actual_record = next(self._records)
        except StopIteration as exc:
            raise CorpusBuildError(
                f"independent verification: output JSONL ended at record {self.n_records}"
            ) from exc
        actual_sequence = self._output_sequence(actual_record)
        if actual_record != expected_record:
            raise CorpusBuildError(
                f"independent verification: JSONL content/order mismatch at record {self.n_records}"
            )
        try:
            teacher_sequence, actual_matrix = self.teacher.next()
        except StopIteration as exc:
            raise CorpusBuildError(
                f"independent verification: output teacher ended at record {self.n_records}"
            ) from exc
        if teacher_sequence != actual_sequence:
            raise CorpusBuildError(
                f"independent verification: output JSONL/teacher sequence mismatch at "
                f"record {self.n_records}"
            )
        if not np.array_equal(actual_matrix, expected_matrix):
            raise CorpusBuildError(
                f"independent verification: teacher probabilities differ at record {self.n_records}"
            )
        self.n_records += 1

    def finish(self) -> Dict[str, object]:
        try:
            _lineno, extra = next(self._records)
        except StopIteration:
            pass
        else:
            self._output_sequence(extra)
            raise CorpusBuildError("independent verification: unexplained extra output JSONL row")
        teacher_metadata = self.teacher.finish()
        after = _sha256_file(self.jsonl_path)
        if after != self.before or after != self.expected_jsonl_hash:
            raise CorpusBuildError("output JSONL hash changed during independent verification")
        if self.blocklist_overlap:
            raise CorpusBuildError(
                f"independent verification: train output has {self.blocklist_overlap} "
                "exact blocklist overlaps"
            )
        self._records.close()
        return {
            "output_jsonl_sha256": after,
            "output_jsonl_sha256_before": self.before,
            "output_jsonl_sha256_after": after,
            "n_output_records": self.n_records,
            "output_blocklist_overlap": self.blocklist_overlap,
            "output_teacher": teacher_metadata,
            "verified_teacher_manifest": self.teacher.manifest,
        }

    def abort(self) -> None:
        self._records.close()


def _teacher_for_record(spec: InputSpec, reader: Optional[TeacherShardReader],
                        record: Dict[str, object], context: str) -> np.ndarray:
    generated = _ground_truth_matrix(record, context) if spec.teacher_kind == "gt" else None
    if reader is None:
        assert generated is not None
        return generated
    try:
        teacher_sequence, existing = reader.next()
    except StopIteration as exc:
        raise CorpusBuildError(f"{context}: missing aligned teacher record") from exc
    if teacher_sequence != record["seq"]:
        raise CorpusBuildError(
            f"{context}: JSONL sequence {record['seq']!r} != teacher sequence "
            f"{teacher_sequence!r}; refusing positional realignment"
        )
    if generated is not None:
        if not np.array_equal(generated, existing):
            raise CorpusBuildError(
                f"{context}: existing GT teacher differs from JSONL pairs one-hot matrix"
            )
        return generated
    return existing


def _process_pass(specs: Sequence[InputSpec], *, role: str, blocked: Set[str],
                  consumer: object,
                  sample: Optional[Tuple[int, Set[str], Dict[str, object]]] = None
                  ) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    per_input: List[Dict[str, object]] = []
    totals: Counter[str] = Counter()
    seen: Dict[str, Tuple[bytes, bytes]] = {}
    try:
        for input_index, spec in enumerate(specs):
            path = Path(spec.path)
            before = _sha256_file(path)
            reader = (TeacherShardReader(Path(spec.teacher_dir), context=f"teacher for {path}")
                      if spec.teacher_dir else None)
            counts: Counter[str] = Counter()
            n_in = n_kept = 0
            for lineno, raw in _iter_jsonl(path):
                context = f"{path}:{lineno}"
                record = _prepare_record(raw, context)
                matrix = _teacher_for_record(spec, reader, record, context)
                sequence = str(record["seq"])
                n_in += 1
                if (sample is not None and input_index == sample[0]
                        and sequence not in sample[1]):
                    counts["filtered_hash_sample"] += 1
                    continue
                if role == "validation":
                    if sequence in seen:
                        raise CorpusBuildError(
                            f"{context}: duplicate validation sequence {sequence!r}"
                        )
                else:
                    if sequence in blocked:
                        counts["filtered_blocklist"] += 1
                        continue
                    if sequence in seen:
                        counts["dropped_exact_duplicate_sequence"] += 1
                        structure_conflict = _pairs_digest(record) != seen[sequence][0]
                        teacher_conflict = _probability_digest(matrix) != seen[sequence][1]
                        if structure_conflict:
                            counts["conflicting_structure_kept_first"] += 1
                        if teacher_conflict:
                            counts["conflicting_teacher_kept_first"] += 1
                        if structure_conflict or teacher_conflict:
                            counts["conflicting_labels_kept_first"] += 1
                        continue
                seen[sequence] = (_pairs_digest(record), _probability_digest(matrix))
                consumer.write(record, matrix)  # type: ignore[attr-defined]
                n_kept += 1
            after = _sha256_file(path)
            if after != before:
                raise CorpusBuildError(f"{path}: input hash changed while reading")
            teacher_metadata = reader.finish() if reader is not None else {
                "teacher_manifest_sha256": None,
                "teacher_manifest_sha256_before": None,
                "teacher_manifest_sha256_after": None,
                "n_teacher_records": n_in,
                "n_teacher_shards": 0,
            }
            source_manifest = reader.manifest if reader is not None else {}
            item: Dict[str, object] = {
                "path": str(path),
                "sha256": after,
                "sha256_before": before,
                "sha256_after": after,
                "teacher_kind": spec.teacher_kind,
                "teacher_dir": spec.teacher_dir,
                "teacher_name": (source_manifest.get("teacher", "mfe")
                                 if spec.teacher_kind == "mfe" else "gt"),
                "teacher_version_lock": (source_manifest.get("version_lock")
                                         if spec.teacher_kind == "mfe" else GT_VERSION),
                "n_in": n_in,
                "n_kept": n_kept,
            }
            item.update(teacher_metadata)
            for name in STAT_NAMES:
                item[name] = counts[name]
                totals[name] += counts[name]
            totals["total_input_records"] += n_in
            totals["total_kept_records"] += n_kept
            per_input.append(item)
        consumer_result = consumer.finish()  # type: ignore[attr-defined]
    except Exception:
        consumer.abort()  # type: ignore[attr-defined]
        raise
    stats: Dict[str, object] = {
        "total_input_records": totals["total_input_records"],
        "total_kept_records": totals["total_kept_records"],
        "n_records_written": totals["total_kept_records"],
        "n_unique_sequences": len(seen),
    }
    for name in STAT_NAMES:
        stats[name] = totals[name]
    return per_input, stats, consumer_result


def _teacher_identity(inputs: List[Dict[str, object]]) -> Tuple[str, Optional[str]]:
    if len(inputs) == 1:
        return str(inputs[0]["teacher_name"]), inputs[0]["teacher_version_lock"]  # type: ignore[return-value]
    kinds = [str(item["teacher_kind"]) for item in inputs]
    name = "mixed:" + "+".join(kinds)
    version = " | ".join(
        f"input{index}:{item['teacher_kind']}:{item['teacher_version_lock']}"
        for index, item in enumerate(inputs)
    )
    return name, version


def _normalized_specs(values: Sequence[InputSpec]) -> List[InputSpec]:
    if not values:
        raise CorpusBuildError("at least one input is required")
    specs: List[InputSpec] = []
    for value in values:
        kind = value.teacher_kind.lower()
        if kind not in ("mfe", "gt"):
            raise CorpusBuildError(f"unknown teacher kind {value.teacher_kind!r}")
        path = Path(value.path).expanduser().resolve()
        if not path.is_file():
            raise CorpusBuildError(f"input JSONL does not exist: {path}")
        teacher_dir = value.teacher_dir or None
        if kind == "mfe" and teacher_dir is None:
            raise CorpusBuildError(f"MFE input requires an existing teacher directory: {path}")
        if teacher_dir is not None:
            teacher_path = Path(teacher_dir).expanduser().resolve()
            if not teacher_path.is_dir():
                raise CorpusBuildError(f"teacher directory does not exist: {teacher_path}")
            teacher_dir = str(teacher_path)
        specs.append(InputSpec(str(path), kind, teacher_dir))
    return specs


def build_corpus(inputs: Sequence[InputSpec], *, out_jsonl: os.PathLike[str] | str,
                 out_teacher_dir: os.PathLike[str] | str, role: str = "train",
                 blocklists: Sequence[os.PathLike[str] | str] = (),
                 shard_size: int = 256, sample_input: Optional[int] = None,
                 sample_count: Optional[int] = None,
                 sample_seed: str = "round29-v1") -> Dict[str, object]:
    """Build and independently verify a corpus; never overwrite final outputs."""
    if role not in ("train", "validation"):
        raise CorpusBuildError("role must be train or validation")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
        raise CorpusBuildError("shard_size must be a positive integer")
    specs = _normalized_specs(inputs)
    if role == "validation" and blocklists:
        raise CorpusBuildError("validation role does not accept --blocklist")
    if role == "validation" and (sample_input is not None or sample_count is not None):
        raise CorpusBuildError("validation role does not accept hash sampling")
    blocklist_paths = [Path(path).expanduser().resolve() for path in blocklists]
    for path in blocklist_paths:
        if not path.is_file():
            raise CorpusBuildError(f"blocklist JSONL does not exist: {path}")

    final_jsonl = Path(out_jsonl).expanduser().resolve()
    final_teacher = Path(out_teacher_dir).expanduser().resolve()
    if final_jsonl == final_teacher:
        raise CorpusBuildError("output JSONL and teacher directory must be different paths")
    if os.path.lexists(final_jsonl) or os.path.lexists(final_teacher):
        raise CorpusBuildError(
            f"refusing to overwrite existing final output: {final_jsonl} or {final_teacher}"
        )
    final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    final_teacher.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(final_jsonl) or os.path.lexists(final_teacher):
        raise CorpusBuildError("final output appeared while preparing build")

    building = Path(tempfile.mkdtemp(
        prefix=f".{final_teacher.name}.building-", dir=str(final_teacher.parent)))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_jsonl.name}.", suffix=".tmp", dir=str(final_jsonl.parent))
    os.close(descriptor)
    temporary_jsonl = Path(temporary_name)
    final_jsonl_created = False
    try:
        blocked_first, blocklist_metadata_first = _load_blocklists(blocklist_paths)
        sample_first = _select_hash_sample(specs, sample_input, sample_count, sample_seed)
        writer = BuildConsumer(temporary_jsonl, building, shard_size)
        inputs_first, stats_first, write_result = _process_pass(
            specs, role=role, blocked=blocked_first, consumer=writer, sample=sample_first)
        if not stats_first["n_records_written"]:
            raise CorpusBuildError("cleaning produced zero output records")

        blocked_second, blocklist_metadata_second = _load_blocklists(blocklist_paths)
        if blocklist_metadata_second != blocklist_metadata_first or blocked_second != blocked_first:
            raise CorpusBuildError("blocklist changed between build and independent verification")
        sample_second = _select_hash_sample(specs, sample_input, sample_count, sample_seed)
        if sample_first != sample_second:
            raise CorpusBuildError("hash sample changed between build and independent verification")
        verifier = VerifyConsumer(
            temporary_jsonl, building, role=role, blocked=blocked_second,
            expected_jsonl_hash=str(write_result["output_jsonl_sha256"]),
        )
        inputs_second, stats_second, verify_result = _process_pass(
            specs, role=role, blocked=blocked_second, consumer=verifier, sample=sample_second)
        if inputs_second != inputs_first:
            raise CorpusBuildError("inputs or counts changed between build and independent verification")
        if stats_second != stats_first:
            raise CorpusBuildError("cleaning counts changed during independent verification")
        if verify_result["n_output_records"] != stats_second["n_records_written"]:
            raise CorpusBuildError("independent verification: output count disagrees with cleaning count")

        teacher_name, version_lock = _teacher_identity(inputs_second)
        verified_manifest = verify_result["verified_teacher_manifest"]
        assert isinstance(verified_manifest, dict)
        stats_second["output_blocklist_overlap"] = verify_result["output_blocklist_overlap"]
        manifest: Dict[str, object] = {
            "format_version": 1,
            "generated_by": "tools/build_clean_corpus.py",
            "role": role,
            "teacher": teacher_name,
            "version_lock": version_lock,
            "shard_size": shard_size,
            "n_sequences": verified_manifest["n_sequences"],
            "shards": verified_manifest["shards"],
            "inputs": inputs_second,
            "blocklists": blocklist_metadata_second,
            "blocklist_applied": role == "train",
            "hash_sampling": sample_second[2] if sample_second is not None else None,
            "stats": stats_second,
            "output": {
                "jsonl": str(final_jsonl),
                "jsonl_basename": final_jsonl.name,
                "sha256": verify_result["output_jsonl_sha256"],
                "n_records": verify_result["n_output_records"],
                "teacher_dir": str(final_teacher),
                "blocklist_overlap": verify_result["output_blocklist_overlap"],
                "independent_second_pass": True,
                "verified_order_count_hashes_probabilities": True,
            },
        }
        final_manifest_tmp = building / "manifest.json.tmp"
        _write_json(final_manifest_tmp, manifest)
        os.replace(final_manifest_tmp, building / "manifest.json")
        os.unlink(building / BUILD_MANIFEST)

        if os.path.lexists(final_jsonl) or os.path.lexists(final_teacher):
            raise CorpusBuildError("refusing to overwrite a final output created during the build")
        os.rename(temporary_jsonl, final_jsonl)
        final_jsonl_created = True
        os.rename(building, final_teacher)
        return manifest
    except Exception:
        if final_jsonl_created:
            try:
                os.unlink(final_jsonl)
            except FileNotFoundError:
                pass
        if temporary_jsonl.exists():
            os.unlink(temporary_jsonl)
        if building.exists():
            shutil.rmtree(building)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", dest="inputs", action="append", nargs="+", required=True,
        metavar=("JSONL", "KIND"),
        help="JSONL mfe TEACHER_DIR, or JSONL gt [TEACHER_DIR]; repeat in corpus order",
    )
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    parser.add_argument("--blocklist", action="append", default=[],
                        help="held-out JSONL to exclude from train; repeat for multiple files")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-teacher-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--sample-input", type=int, action="append",
                        help="1-based input to hash-sample; repeat is rejected")
    parser.add_argument("--sample-count", type=int, action="append",
                        help="unique sequences selected by smallest hash; repeat is rejected")
    parser.add_argument("--sample-seed", action="append",
                        help="sampling seed string; default round29-v1; repeat is rejected")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.role == "train" and not args.blocklist:
        print("FATAL: train role requires at least one explicit --blocklist", file=sys.stderr)
        return 2
    for name in ("sample_input", "sample_count", "sample_seed"):
        values = getattr(args, name)
        if values is not None and len(values) != 1:
            print(f"FATAL: --{name.replace('_', '-')} may only be given once", file=sys.stderr)
            return 2
    sample_input = args.sample_input[0] if args.sample_input else None
    sample_count = args.sample_count[0] if args.sample_count else None
    sample_seed = args.sample_seed[0] if args.sample_seed else "round29-v1"
    specs: List[InputSpec] = []
    for value in args.inputs:
        if len(value) not in (2, 3):
            print("FATAL: each --input needs JSONL {mfe,gt} [TEACHER_DIR]", file=sys.stderr)
            return 2
        specs.append(InputSpec(value[0], value[1], value[2] if len(value) == 3 else None))
    try:
        manifest = build_corpus(
            specs, out_jsonl=args.out_jsonl, out_teacher_dir=args.out_teacher_dir,
            role=args.role, blocklists=args.blocklist, shard_size=args.shard_size,
            sample_input=sample_input, sample_count=sample_count,
            sample_seed=sample_seed,
        )
    except (CorpusBuildError, OSError, ValueError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "role": args.role,
        "n_records": manifest["n_sequences"],
        "stats": manifest["stats"],
        "jsonl": manifest["output"]["jsonl"],  # type: ignore[index]
        "teacher_dir": manifest["output"]["teacher_dir"],  # type: ignore[index]
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
