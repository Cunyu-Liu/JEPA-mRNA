#!/usr/bin/env python3
"""Materialize RiNALMo's official ArchiveII nine-family folds and audit JSONL views."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

ARCHIVE_SHA256 = "0ff7209016ba1775288572794e850d5857937c391d67c9ccacfe9ae77c0188e5"
ARCHIVE_ALL_CT_FILES = 54110
FAM_FOLD_CT_FILES = 34785
FAMILIES = ("16s", "23s", "5s", "RNaseP", "grp1", "srp", "tRNA", "telomerase", "tmRNA")
SPLITS = ("train", "valid", "test")
PREFIX = PurePosixPath("ct/fam-fold")
OFFICIAL_REPO = Path("/home/rna_jev/third_party/RiNALMo")
EXPECTED_OFFICIAL_COMMIT = "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"
EXPECTED_OFFICIAL_PARSER_SHA256 = "75c971655e9a12cbbe18d3f75b8377303d0acc1795fab0f0253721ec14b750d0"
if sys.flags.optimize:
    raise RuntimeError("optimized Python disables audit assertions; refusing to generate a PASS data card")
sys.path.insert(0, str(OFFICIAL_REPO))
from rinalmo.utils.sec_struct import parse_ct_file  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive: tarfile.TarFile):
    for member in archive:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isdir() or member.isfile(), f"unsupported archive member: {member.name}"
        ct_root = PREFIX.parent
        assert path == ct_root or ct_root in path.parents, f"unexpected archive member: {member.name}"
        if path == ct_root or path == PREFIX or PREFIX in path.parents:
            yield member, path
        else:
            other_folds = ct_root / "k-fold"
            assert path == other_folds or other_folds in path.parents, \
                f"unexpected non-fam-fold member: {member.name}"


def materialize_ct_files(archive_path: Path, raw_root: Path) -> int:
    count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member, relative in safe_members(archive):
            destination = raw_root.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.extractfile(member).read()
            if destination.exists():
                assert destination.read_bytes() == payload, f"CT identity mismatch: {destination}"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            count += 1
    assert count == FAM_FOLD_CT_FILES, count
    return count


def ct_paths(raw_root: Path, family: str, split: str) -> list[Path]:
    directory = raw_root / PREFIX / family / split
    paths = sorted(directory.glob("*.ct"))
    assert paths, directory
    return paths


def parse_official(path: Path) -> tuple[str, list[list[int]]]:
    sequence, matrix = parse_ct_file(path)
    matrix = np.asarray(matrix)
    assert matrix.shape == (len(sequence), len(sequence))
    assert np.array_equal(matrix, matrix.T), f"asymmetric CT pairs: {path}"
    pairs = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(matrix, k=1)))]
    return sequence, pairs


def write_jsonl_views(raw_root: Path, output_root: Path):
    expected = {}
    for family in FAMILIES:
        for split in SPLITS:
            paths = ct_paths(raw_root, family, split)
            out_path = output_root / family / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = out_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for path in paths:
                    sequence, pairs = parse_official(path)
                    row = {
                        "id": path.stem,
                        "family": family,
                        "held_out_family": family,
                        "split": split,
                        "sequence": sequence,
                        "pairs": pairs,
                        "source_ct": str(path),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            temporary.replace(out_path)
            expected[(family, split)] = (out_path, paths)
            print(f"[archiveII/write] {family}/{split} rows={len(paths)}", flush=True)
    return expected


def validate_views(expected, archive_path: Path, raw_root: Path, output_root: Path, code_path: Path):
    official_commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True).strip()
    assert official_commit == EXPECTED_OFFICIAL_COMMIT
    assert not subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip(), "official RiNALMo tracked files must be clean"
    parser_sha256 = sha256(OFFICIAL_REPO / "rinalmo/utils/sec_struct.py")
    assert parser_sha256 == EXPECTED_OFFICIAL_PARSER_SHA256
    card = {
        "schema_version": 1,
        "status": "PASS",
        "task": "RiNALMo official ArchiveII leave-one-family-out secondary structure",
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "archive_all_ct_files": ARCHIVE_ALL_CT_FILES,
        "fam_fold_ct_files": FAM_FOLD_CT_FILES,
        "excluded_k_fold_ct_files": ARCHIVE_ALL_CT_FILES - FAM_FOLD_CT_FILES,
        "official_repo": str(OFFICIAL_REPO),
        "official_commit": official_commit,
        "official_parser_sha256": parser_sha256,
        "processing": "official parse_ct_file; official fam-fold train/valid/test; no filtering, deduplication, pseudoknot removal, or canonical-pair projection",
        "code_sha256": sha256(code_path),
        "folds": {},
        "jsonl_sha256": {},
    }
    total = 0
    observed_members = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        observed_members = sum(1 for member in archive if member.isfile())
    with tarfile.open(archive_path, "r:gz") as archive:
        observed_fam_members = sum(
            1 for member in archive if member.isfile() and PREFIX in PurePosixPath(member.name).parents)
    assert observed_members == ARCHIVE_ALL_CT_FILES
    assert observed_fam_members == FAM_FOLD_CT_FILES
    for (family, split), (out_path, paths) in expected.items():
        with out_path.open(encoding="utf-8") as handle:
            saved = [json.loads(line) for line in handle if line.strip()]
        assert len(saved) == len(paths)
        source_paths = set()
        for row, path in zip(saved, paths):
            reparsed_sequence, reparsed_pairs = parse_official(path)
            assert row["sequence"] == reparsed_sequence
            assert row["pairs"] == reparsed_pairs
            assert row["id"] == path.stem and row["family"] == family
            assert row["held_out_family"] == family and row["split"] == split
            assert Path(row["source_ct"]) == path
            source_paths.add(str(path))
        assert len(source_paths) == len(paths)
        key = f"{family}/{split}"
        lengths = [len(row["sequence"]) for row in saved]
        card["folds"][key] = {
            "rows": len(saved),
            "pairs": sum(len(row["pairs"]) for row in saved),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "sequence_sha256": hashlib.sha256(
                "\n".join(row["sequence"] for row in saved).encode()).hexdigest(),
            "nucleotides": dict(sorted(Counter("".join(row["sequence"] for row in saved)).items())),
        }
        card["jsonl_sha256"][key] = sha256(out_path)
        total += len(saved)
        print(f"[archiveII/validate] {key} rows={len(saved)}", flush=True)
        del saved
    assert total == FAM_FOLD_CT_FILES
    card["total_rows"] = total
    card_path = output_root / "data_card.json"
    temporary = card_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(card, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(card_path)
    checked = json.loads(card_path.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and checked["total_rows"] == FAM_FOLD_CT_FILES
    assert len(checked["folds"]) == 27 and len(checked["jsonl_sha256"]) == 27
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        default=Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/archiveII"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    assert args.archive.is_file() and sha256(args.archive) == ARCHIVE_SHA256
    raw_root = args.root / "raw"
    output_root = args.root / "jsonl"
    if args.validate_only:
        expected = {}
        for family in FAMILIES:
            for split in SPLITS:
                out_path = output_root / family / f"{split}.jsonl"
                assert out_path.is_file(), out_path
                expected[(family, split)] = (out_path, ct_paths(raw_root, family, split))
    else:
        materialize_ct_files(args.archive, raw_root)
        expected = write_jsonl_views(raw_root, output_root)
    card = validate_views(expected, args.archive, raw_root, output_root, Path(__file__).resolve())
    print("ARCHIVEII_OFFICIAL_DATA_PASS", json.dumps({
        "rows": card["total_rows"], "families": len(FAMILIES), "fold_splits": len(card["folds"]),
        "data_card_sha256": sha256(output_root / "data_card.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
