#!/usr/bin/env python3
"""Independent ArchiveII card/archive/JSONL consistency audit."""
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

if sys.flags.optimize:
    raise RuntimeError("optimized Python disables audit assertions")

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/archiveII")
ARCHIVE = ROOT.parent / "ct-splits.tar.gz"
CARD = ROOT / "jsonl/data_card.json"
PREFIX = PurePosixPath("ct/fam-fold")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    card = json.loads(CARD.read_text())
    assert card["status"] == "PASS" and card["total_rows"] == 34785
    assert card["archive_sha256"] == sha256(ARCHIVE)
    assert card["code_sha256"] == "44839b50da9fed77e8741c94803be8f3c296c5064018d1ca738acdc74a99f841"
    assert card["official_commit"] == "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"
    assert card["official_parser_sha256"] == "75c971655e9a12cbbe18d3f75b8377303d0acc1795fab0f0253721ec14b750d0"
    archive_counts = Counter()
    total_files = 0
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            total_files += 1
            path = PurePosixPath(member.name)
            if PREFIX in path.parents:
                family, split = path.parts[2], path.parts[3]
                archive_counts[f"{family}/{split}"] += 1
    assert total_files == 54110
    assert sum(archive_counts.values()) == 34785
    assert set(archive_counts) == set(card["folds"]) == set(card["jsonl_sha256"])
    total = 0
    for key in sorted(card["folds"]):
        family, split = key.split("/")
        path = ROOT / "jsonl" / family / f"{split}.jsonl"
        assert sha256(path) == card["jsonl_sha256"][key]
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        expected = card["folds"][key]
        assert len(rows) == expected["rows"] == archive_counts[key]
        assert len({row["source_ct"] for row in rows}) == len(rows)
        for row in rows:
            assert row["family"] == row["held_out_family"] == family and row["split"] == split
            assert "/raw/ct/fam-fold/" in row["source_ct"] and "/k-fold/" not in row["source_ct"]
        lengths = [len(row["sequence"]) for row in rows]
        assert min(lengths) == expected["min_length"] and max(lengths) == expected["max_length"]
        assert sum(len(row["pairs"]) for row in rows) == expected["pairs"]
        sequence_sha = hashlib.sha256("\n".join(row["sequence"] for row in rows).encode()).hexdigest()
        assert sequence_sha == expected["sequence_sha256"]
        nucleotides = dict(sorted(Counter("".join(row["sequence"] for row in rows)).items()))
        assert nucleotides == expected["nucleotides"]
        total += len(rows)
        print("ARCHIVE_QA_SPLIT_PASS", key, len(rows), flush=True)
    assert total == 34785
    print("ARCHIVEII_EXTERNAL_QA_PASS", json.dumps({
        "rows": total, "fold_splits": len(card["folds"]), "archive_all_ct": total_files,
        "data_card_sha256": sha256(CARD)}), flush=True)


if __name__ == "__main__":
    main()
