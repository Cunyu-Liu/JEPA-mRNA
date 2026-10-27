#!/usr/bin/env python3
"""Prepare Spliceator GS_1/db_1 and four-species test sets without network access."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SRC = Path("/root/rinalmo_downstream/splice_unpacked/data/spliceator")
OUT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/spliceator_official")
LIST_SIZE = 400


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge(pos_path, neg_path):
    sequences, groups, labels = [], [], []
    for label, paths in [(1, [pos_path]), (0, [neg_path])]:
        for path in paths:
            with path.open() as handle:
                for line in handle:
                    if line.startswith("ID_uniprot"):
                        continue
                    fields = line.strip().split(";")
                    seq = fields[2] if len(fields[1]) < 100 else fields[1]
                    skip = (len(seq) - LIST_SIZE) // 2
                    sequences.append(seq[skip:skip + LIST_SIZE])
                    groups.append(fields[0].split("_")[-1])
                    labels.append(label)
    return np.array(groups), np.array(sequences), np.array(labels)


def main():
    counts = {}
    for ss_type in ["acceptor", "donor"]:
        groups, sequences, labels = merge(SRC / f"Training_data/Positive/GS/POS_{ss_type}_600.csv",
                                          SRC / f"Training_data/Negative/GS/GS_1/NEG_600_{ss_type}.csv")
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        train_idx, val_idx = next(splitter.split(np.arange(len(labels)), y=labels))
        db = OUT / "GS_1/db_1"
        db.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({0: groups[train_idx], 1: sequences[train_idx], 2: labels[train_idx]}).to_csv(
            db / f"Train_{ss_type}_{LIST_SIZE}.csv", sep=";", header=False, index=False)
        pd.DataFrame({0: groups[val_idx], 1: sequences[val_idx], 2: labels[val_idx]}).to_csv(
            db / f"Val_{ss_type}_{LIST_SIZE}.csv", sep=";", header=False, index=False)
        counts[ss_type] = {"train": len(train_idx), "validation": len(val_idx),
                           "positive": int(labels.sum()), "negative": int((1 - labels).sum())}
        for species in ["Danio", "Fly", "Thaliana", "Worm"]:
            seqs, ys = [], []
            for label, suffix in [(1, "positive"), (0, "negative")]:
                path = SRC / f"Benchmarks/{species}/SA_sequences_{ss_type}_{LIST_SIZE}_Final_3.{suffix}.txt"
                frame = pd.read_csv(path, delimiter=";", header=None)
                seqs.extend(frame[2].astype(str).tolist())
                ys.extend([label] * len(frame))
            target = OUT / "Benchmarks" / species
            target.mkdir(parents=True, exist_ok=True)
            np.savetxt(target / f"SA_sequences_{ss_type}_{LIST_SIZE}_Final_3.fasta", seqs, fmt="%s")
            np.savetxt(target / f"SA_labels_{ss_type}_{LIST_SIZE}_Final_3.fasta", ys, fmt="%d")
            counts[f"{species}_{ss_type}"] = {"test": len(seqs), "positive": int(sum(ys)), "negative": len(ys) - int(sum(ys))}
    files = {str(path.relative_to(OUT)): sha256(path) for path in sorted(OUT.rglob("*")) if path.is_file()}
    card = {"status": "PASS", "protocol": "Spliceator GS_1/db_1, 400 nt centred windows; StratifiedKFold seed=2020; four-species benchmark",
            "counts": counts, "files": files}
    (OUT / "data_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(card["counts"], indent=1))
    print(f"SPLICEATOR_PREP_PASS files={len(files)}")


if __name__ == "__main__":
    main()
