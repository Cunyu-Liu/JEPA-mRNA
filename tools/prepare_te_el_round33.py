#!/usr/bin/env python3
"""Validate RiNALMo/UTR-LM author-processed TE/EL CSV files and ten-fold splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/te_el_official")
BASE_URL = "https://files.codeocean.com/files/verified/98b8bfcf-6e29-40a1-b248-1bc17c7a03da_v1.0/data/TE_REL_Endogenous_Cao/"
CELLS = {
    "Muscle": {
        "rows": 1257,
        "sha256": "aebd90663a317c0cacd6ea3f359551946eef36ef041c80fad11fb919d74e5601",
        "raw": "df_counts_and_len.TE_sorted.Muscle.with_annot.txt",
        "raw_sha256": "6f8f11db9f1ee2bfa475a86df3572127b24133ff1fd290bd52e50c775c6468f7",
    },
    "pc3": {
        "rows": 12579,
        "sha256": "a73e42ae803c5ea751535f766abb787a9b1d6f6ea78038314a7c6cda935c970f",
        "raw": "df_counts_and_len.TE_sorted.pc3.with_annot.txt",
        "raw_sha256": "91c41e7b340b348a1f87ef1c1743b5b176222820598860dd87b0e5a1ed97c852",
    },
    "HEK": {
        "rows": 14410,
        "sha256": "bf4adb03edd5ba6170ebaa7deadd27bb92be76093b741e6bc26ed29d9af2739b",
        "raw": "df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt",
        "raw_sha256": "d2323020d32af5fbe5500c97648a37a67c242f31b7c6109e3069fd69c9867bd3",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def indices_sha256(indices) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in indices], separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    card = {
        "schema_version": 1,
        "status": "PASS",
        "task": "RiNALMo/UTR-LM translation efficiency and expression level",
        "processing": "author-processed _sequence.csv downloaded directly; no filtering or reconstruction by this project",
        "source_capsule": "https://codeocean.com/capsule/4214075/tree/v1",
        "source_base_url": BASE_URL,
        "cells": {},
        "code_sha256": sha256(Path(__file__).resolve()),
    }
    for cell, expected in CELLS.items():
        path = ROOT / f"{cell}_sequence.csv"
        raw_path = ROOT / "source" / expected["raw"]
        assert path.is_file() and sha256(path) == expected["sha256"]
        assert raw_path.is_file() and sha256(raw_path) == expected["raw_sha256"]
        frame = pd.read_csv(path)
        raw = pd.read_csv(raw_path, sep=r"\s+", engine="python")
        assert len(frame) == expected["rows"] and frame["enst"].is_unique
        required = ["enst", "te", "rpkm_rnaseq", "rpkm_riboseq", "te_log", "rnaseq_log",
                    "utr_originial_varylength", "length", "utr"]
        assert set(required).issubset(frame.columns)
        assert frame[required].notna().all().all()
        assert np.isfinite(frame[["te", "rpkm_rnaseq", "rpkm_riboseq", "te_log", "rnaseq_log"]].to_numpy()).all()
        assert np.allclose(frame["te_log"], np.log(frame["te"]), rtol=0, atol=1e-12)
        assert np.allclose(frame["rnaseq_log"], np.log(frame["rpkm_rnaseq"]), rtol=0, atol=1e-12)
        assert raw.index.is_unique and frame["enst"].isin(raw.index).all()
        aligned = raw.loc[frame["enst"]]
        assert np.allclose(aligned["te"].to_numpy(), frame["te"].to_numpy(), rtol=0, atol=1e-9)
        assert np.allclose(aligned["rpkm_rnaseq"].to_numpy(), frame["rpkm_rnaseq"].to_numpy(), rtol=0, atol=1e-9)
        assert np.allclose(aligned["rpkm_riboseq"].to_numpy(), frame["rpkm_riboseq"].to_numpy(), rtol=0, atol=1e-9)
        for row in frame.itertuples(index=False):
            original = str(row.utr_originial_varylength)
            assert len(original) == int(row.length)
            expected_utr = original[-100:] + "<pad>" * max(0, 100 - len(original))
            assert str(row.utr) == expected_utr
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        splits = list(kfold.split(frame))
        fold_cards = {}
        for fold in range(10):
            test_idx = splits[fold][1]
            valid_idx = splits[fold - 1][1]
            excluded = set(test_idx) | set(valid_idx)
            train_idx = sorted(set(range(len(frame))) - excluded)
            assert len(train_idx) + len(valid_idx) + len(test_idx) == len(frame)
            assert not set(train_idx) & set(valid_idx) and not set(train_idx) & set(test_idx)
            assert not set(valid_idx) & set(test_idx)
            fold_cards[str(fold)] = {
                "train": len(train_idx), "validation": len(valid_idx), "test": len(test_idx),
                "train_indices_sha256": indices_sha256(train_idx),
                "validation_indices_sha256": indices_sha256(valid_idx),
                "test_indices_sha256": indices_sha256(test_idx),
            }
        card["cells"][cell] = {
            "rows": len(frame),
            "processed_file": path.name,
            "processed_sha256": sha256(path),
            "processed_download_url": BASE_URL + path.name + "?download",
            "raw_annotation_file": f"source/{expected['raw']}",
            "raw_annotation_sha256": sha256(raw_path),
            "utr_protocol": "last 100 nt of utr_originial_varylength; right-pad shorter sequences with <pad> to 100 nt",
            "labels": "te_log=ln(te); rnaseq_log=ln(rpkm_rnaseq)",
            "kfold": "KFold(n_splits=10, shuffle=True, random_state=42); validation=test_indices[fold-1]",
            "folds": fold_cards,
        }
        print("TE_EL_CELL_PASS", cell, len(frame), flush=True)
    out = ROOT / "data_card.json"
    temporary = ROOT / "data_card.tmp"
    temporary.write_text(json.dumps(card, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out)
    checked = json.loads(out.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and len(checked["cells"]) == 3
    print("TE_EL_OFFICIAL_DATA_PASS", json.dumps({
        "rows": {cell: card["cells"][cell]["rows"] for cell in CELLS},
        "data_card_sha256": sha256(out)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
