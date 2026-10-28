#!/usr/bin/env python3
"""Validate bpRNA-spot-0 mirror and create full-fidelity JSONL views for RiNALMo."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/bpRNA_spot0")
EXPECTED = {
    "train": (10814, "0fda1d09f4ce685b7e2a5497f5fd06b6a54a5aeaf287f414c089bb1dc9b5b640"),
    "validation": (1300, "00934c684a543e9a7209bd2df9bbf6caf487460e9f536c5f43bdd94fbcd2054d"),
    "test": (1305, "bfdc1b55b3a070932c3ca4d5db146a05ace9dc4e207ba7e51b2efb86d6535341"),
}
OFFICIAL_REPO = Path("/home/rna_jev/third_party/RiNALMo")
sys.path.insert(0, str(OFFICIAL_REPO))
from rinalmo.utils.sec_struct import dot_bracket_to_2d_mat  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_from_structure(structure: str, length: int) -> list[list[int]]:
    matrix = np.asarray(dot_bracket_to_2d_mat(structure))
    assert matrix.shape == (length, length) and np.array_equal(matrix, matrix.T)
    assert np.all(matrix.sum(axis=0) <= 1) and np.all(matrix.sum(axis=1) <= 1)
    return [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(matrix, k=1)))]


def main() -> int:
    card = {
        "schema_version": 1,
        "status": "PASS",
        "task": "RiNALMo bpRNA intra-family secondary structure TR0/VL0/TS0",
        "source": "MultiMolecule bprna-spot-0 mirror of SPOT-RNA's original split",
        "source_url": "https://hf-mirror.com/datasets/multimolecule/bprna-spot-0",
        "original_release": "https://dl.dropboxusercontent.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip",
        "processing": "direct sequence and dot-bracket conversion with official dot_bracket_to_2d_mat; no filtering, deduplication, pseudoknot removal, or canonical-pair projection",
        "code_sha256": sha256(Path(__file__).resolve()),
        "splits": {},
        "jsonl_sha256": {},
    }
    id_sets = {}
    for split, (expected_rows, expected_sha) in EXPECTED.items():
        source = ROOT / "source" / f"{split}.parquet"
        assert source.is_file() and sha256(source) == expected_sha
        frame = pd.read_parquet(source)
        assert len(frame) == expected_rows and frame["id"].is_unique
        assert {"id", "sequence", "secondary_structure"}.issubset(frame.columns)
        id_sets[split] = set(frame["id"])
        canonical_split = "valid" if split == "validation" else split
        out = ROOT / "jsonl" / "TS0" / f"{canonical_split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(".tmp")
        pair_count = 0
        nucleotides = Counter()
        with temporary.open("w", encoding="utf-8") as handle:
            for row in frame.itertuples(index=False):
                sequence = str(row.sequence)
                structure = str(row.secondary_structure)
                assert len(sequence) == len(structure) and sequence and structure
                pairs = pairs_from_structure(structure, len(sequence))
                record = {
                    "id": row.id,
                    "family": "TS0",
                    "held_out_family": "TS0",
                    "split": canonical_split,
                    "sequence": sequence,
                    "secondary_structure": structure,
                    "pairs": pairs,
                    "source_ct": f"bprna-spot-0/{split}.parquet:{row.id}",
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                pair_count += len(pairs)
                nucleotides.update(sequence)
        temporary.replace(out)
        with out.open(encoding="utf-8") as handle:
            saved = [json.loads(line) for line in handle if line.strip()]
        assert len(saved) == expected_rows
        for row, original in zip(saved, frame.itertuples(index=False)):
            assert row["id"] == original.id and row["sequence"] == original.sequence
            assert row["secondary_structure"] == original.secondary_structure
            assert row["pairs"] == pairs_from_structure(row["secondary_structure"], len(row["sequence"]))
        card["splits"][split] = {
            "rows": expected_rows,
            "pairs": pair_count,
            "nucleotides": dict(sorted(nucleotides.items())),
            "sequence_sha256": hashlib.sha256("\n".join(frame["sequence"]).encode()).hexdigest(),
        }
        card["jsonl_sha256"][f"TS0/{canonical_split}"] = sha256(out)
        print("BPRNA_SPLIT_PASS", split, expected_rows, pair_count, flush=True)
    assert sum(len(ids) for ids in id_sets.values()) == 13419
    assert not (id_sets["train"] & id_sets["validation"])
    assert not (id_sets["train"] & id_sets["test"])
    assert not (id_sets["validation"] & id_sets["test"])
    card["total_rows"] = 13419
    card["id_overlap"] = 0
    card["project_subset_note"] = "Project JSONL subsets exclude some N/ambiguous-containing records and are not used as the original full protocol; TS0 project n=1288 is contained in this mirror's 1305 IDs."
    out = ROOT / "jsonl/data_card.json"
    temporary = ROOT / "jsonl/data_card.tmp"
    temporary.write_text(json.dumps(card, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out)
    checked = json.loads(out.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and checked["total_rows"] == 13419
    print("BPRNA_SPOT0_DATA_PASS", json.dumps({
        "rows": card["total_rows"], "data_card_sha256": sha256(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
