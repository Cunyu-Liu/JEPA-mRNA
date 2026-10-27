#!/usr/bin/env python3
"""用 RiNALMo 官方流程生成并验收 88 类 ncRNA 数据与 200 nt 边界噪声。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/ncrna_official")
RDA = ROOT / "train_test_val_sets_177_families.RDa"
RDA_SHA256 = "5de7d2d39e0496a751afaf4cb8c65e5119cd731e52d4a3477e8a2564f3913f4f"
OFFICIAL_REPO = Path("/home/rna_jev/third_party/RiNALMo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_write_card(code_path: Path) -> dict:
    frames = {}
    files = [RDA]
    class_maps = {}
    counts = {}
    for split in ["train", "val", "test"]:
        base = None
        for suffix in ["", "_bn200"]:
            key = f"{split}{suffix}"
            path = ROOT / f"{key}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            assert {"name", "width", "sequence", "class_id"}.issubset(frame.columns)
            assert len(frame) > 0 and frame["name"].nunique() == 88
            assert sorted(frame["class_id"].unique().tolist()) == list(range(88))
            assert frame["sequence"].map(lambda seq: set(str(seq)).issubset(set("ACTG"))).all()
            assert (frame["sequence"].astype(str).str.len() == frame["width"]).all()
            assert frame["width"].between(1, 600).all()
            class_map = frame.drop_duplicates("name").set_index("name")["class_id"].to_dict()
            assert len(class_map) == 88
            if key in {"train", "val", "test"}:
                class_maps[key] = class_map
            if suffix:
                assert len(frame) == len(base)
                assert frame["name"].equals(base["name"])
                assert frame["class_id"].equals(base["class_id"])
                original = base["sequence"].astype(str).tolist()
                noisy = frame["sequence"].astype(str).tolist()
                assert all(original_seq in noisy_seq for original_seq, noisy_seq in zip(original, noisy))
                assert (frame["width"] >= base["width"]).all()
                assert (frame["width"] <= base["width"] + 4 * (base["width"] // 2)).all()
            else:
                assert frame["width"].between(1, 200).all()
                base = frame
            frames[key] = frame
            counts[key] = {"rows": len(frame), "classes": frame["name"].nunique()}
            files.append(path)
        fasta = ROOT / f"{split}.fasta"
        assert fasta.is_file() and fasta.stat().st_size > 0
        files.append(fasta)
    assert class_maps["train"] == class_maps["val"] == class_maps["test"]
    assert frames["train"]["name"].value_counts().nunique() == 1, "official training balance was not preserved"

    from rpy2.robjects import r
    card = {
        "schema_version": 1,
        "status": "PASS",
        "task": "RiNALMo official 88-class ncRNA classification",
        "source": "https://raw.githubusercontent.com/bioinformatics-sannio/ncrna-deep/master/datasets/Rfam-novel/train_test_val_sets_177_families.RDa",
        "source_rda_sha256": sha256(RDA),
        "processing": "official prepare_ncrna_classification_data + add_noise_to_ncrna_data; R factors preserved with as.character; random seeds fixed to 42",
        "official_repo": str(OFFICIAL_REPO),
        "official_commit": subprocess.check_output(["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True).strip(),
        "r_version": str(r("R.version.string")[0]),
        "biostrings_version": str(r("as.character(packageVersion(\"Biostrings\"))")[0]),
        "counts": counts,
        "files": {path.name: sha256(path) for path in files},
        "code_sha256": sha256(code_path),
    }
    assert card["source_rda_sha256"] == RDA_SHA256
    out = ROOT / "data_card.json"
    out.write_text(json.dumps(card, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    checked = json.loads(out.read_text(encoding="utf-8"))
    assert checked["status"] == "PASS" and len(checked["files"]) == 10
    return card


def preserve_r_factor_strings() -> None:
    """避免新版 rpy2/pandas 把 R factor 变成整数编码，不改变任何序列内容。"""
    from rpy2.robjects import pandas2ri, r, default_converter
    from rpy2.robjects.conversion import localconverter
    original = pandas2ri.rpy2py
    as_character = r("as.character")

    def convert(obj):
        frame = original(obj)
        if isinstance(frame, pd.DataFrame):
            for column, r_index in [("name", 1), ("sequence", 3)]:
                if column in frame.columns:
                    with localconverter(default_converter):
                        values = [str(value) for value in as_character(obj.rx2(r_index))]
                    assert len(values) == len(frame)
                    frame[column] = values
        return frame

    pandas2ri.rpy2py = convert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    assert RDA.is_file() and sha256(RDA) == RDA_SHA256
    if not args.validate_only:
        from rinalmo.utils.prepare_ncrna_classification_data import (
            add_noise_to_ncrna_data,
            prepare_ncrna_classification_data,
        )
        preserve_r_factor_strings()
        random.seed(42)
        np.random.seed(42)
        prepare_ncrna_classification_data(ROOT)
        random.seed(42)
        np.random.seed(42)
        add_noise_to_ncrna_data(ROOT)
    card = validate_and_write_card(Path(__file__).resolve())
    print("NCRNA88_OFFICIAL_DATA_PASS", json.dumps(card["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
