#!/usr/bin/env python3
"""Build the CPEB3 source table and frozen train/validation/test split."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/nucleicbert/cpeb3")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    mut = pd.read_csv(ROOT / "mut12_data.csv")
    phylo = pd.read_csv(ROOT / "CPEB3_Phylo.csv", header=None, skiprows=1)
    assert list(mut.columns) == ["sequences", "total_cnt_rate", "number_mutations"]
    assert phylo.shape[1] == 11 and len(mut) == 21322 and len(phylo) == 27647
    left = pd.DataFrame({
        "sequences": mut["sequences"].astype(str),
        "fit_mean": pd.to_numeric(mut["total_cnt_rate"], errors="raise"),
        "number_mutations": pd.to_numeric(mut["number_mutations"], errors="raise").astype(int),
        "source": "mut12_data",
    })
    right = pd.DataFrame({
        "sequences": phylo[0].astype(str),
        "fit_mean": phylo.iloc[:, -3:].mean(axis=1).astype(float),
        "number_mutations": phylo[1].astype(int),
        "source": "CPEB3_Phylo",
    })
    data = pd.concat([left, right], ignore_index=True)
    assert len(data) == 48969 and data["sequences"].nunique() == len(data)
    assert set(data["sequences"].str.upper().str.replace("U", "T").str.cat()).issubset(set("ACGT"))
    assert np.isfinite(data["fit_mean"]).all() and data["fit_mean"].between(0, 1).all()
    assert not set(left["sequences"]).intersection(set(right["sequences"]))
    data.to_csv(ROOT / "cpeb3_all.csv", index=False)
    rng = np.random.RandomState(2026)
    order = rng.permutation(len(data))
    n_test = int(round(0.20 * len(data)))
    n_valid = int(round(0.20 * (len(data) - n_test)))
    test = sorted(order[:n_test].tolist())
    valid = sorted(order[n_test:n_test + n_valid].tolist())
    train = sorted(order[n_test + n_valid:].tolist())
    assert len(test) + len(valid) + len(train) == len(data)
    assert not set(test) & set(valid) and not set(test) & set(train) and not set(valid) & set(train)
    split = {"seed": 2026, "train": train, "validation": valid, "test": test,
             "n_train": len(train), "n_validation": len(valid), "n_test": len(test),
             "protocol": "80:20 train/test, with 20% of training data held out for early stopping; identical split for both heads"}
    (ROOT / "split.json").write_text(json.dumps(split, indent=1) + "\n")
    card = {"status": "PASS", "n_sequences": len(data), "n_unique_sequences": data["sequences"].nunique(),
            "source_rows": {"mut12_data": len(left), "CPEB3_Phylo": len(right)},
            "source_overlap": 0, "label": "fit_mean; Phylo uses the mean of three replicate ratios",
            "split": {k: split[k] for k in ["seed", "n_train", "n_validation", "n_test"]},
            "files": {p.name: sha256(p) for p in [ROOT / "mut12_data.csv", ROOT / "CPEB3_Phylo.csv", ROOT / "cpeb3_all.csv", ROOT / "split.json"]}}
    (ROOT / "data_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(card, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
