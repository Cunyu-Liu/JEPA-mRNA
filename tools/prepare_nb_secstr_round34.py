#!/usr/bin/env python3
"""Build NucleicBERT's original secondary-structure splits without label projection."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

MAX_CORE = 598
EXPECTED = {"train_rnalign_tr0": 47963, "vl0": 1300, "ts0": 1305, "archiveii600": 3911}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalize(sequence: str) -> str:
    return sequence.upper().replace("U", "T").replace("X", "N").replace("I", "N").replace("T", "U")


def parse_dot_bracket(structure: str, nb_repo: Path) -> list[list[int]]:
    if str(nb_repo) not in sys.path:
        sys.path.insert(0, str(nb_repo))
    from nucleicbert.downstream.secstrdataset import dot_bracket_to_pair_matrix
    matrix = dot_bracket_to_pair_matrix(structure)
    return [[int(i), int(j)] for i, j in np.argwhere(np.triu(matrix, k=1) > 0)]


def crosses(a: tuple[int, int], b: tuple[int, int]) -> bool:
    i, j = a
    k, l = b
    return (i < k < j < l) or (k < i < l < j)


def pairs_to_structure(length: int, pairs: list[list[int]]) -> str:
    tiers: list[list[str]] = []
    placed: list[list[tuple[int, int]]] = []
    for i, j in sorted((tuple(p) for p in pairs), key=lambda p: (p[0], -p[1])):
        for tier, existing in enumerate(placed):
            if not any(crosses((i, j), old) for old in existing):
                break
        else:
            if len(placed) == 4:
                raise ValueError("more than four dot-bracket tiers are required")
            tiers.append(["."] * length)
            placed.append([])
            tier = len(placed) - 1
        tiers[tier][i], tiers[tier][j] = "()[]{}<>"[2 * tier:2 * tier + 2]
        placed[tier].append((i, j))
    chars = ["."] * length
    for tier in tiers:
        for index, char in enumerate(tier):
            if char != ".":
                chars[index] = char
    return "".join(chars)


def parse_bpseq(text: str, name: str) -> tuple[str, list[list[int]]]:
    bases: dict[int, str] = {}
    links: dict[int, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        index = int(fields[0])
        bases[index] = fields[1]
        links[index] = int(fields[2])
    ordered = sorted(bases)
    if ordered != list(range(1, len(ordered) + 1)):
        raise ValueError(f"non-contiguous bpseq indices: {name}")
    pairs = []
    for index in ordered:
        partner = links[index]
        if partner and index < partner:
            if partner not in bases or links.get(partner) != index:
                raise ValueError(f"non-reciprocal pair {index}-{partner}: {name}")
            pairs.append([index - 1, partner - 1])
    return "".join(bases[index] for index in ordered), pairs


def parquet_records(path: Path, source: str, nb_repo: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    records = []
    for row in frame.itertuples(index=False):
        sequence = normalize(str(row.sequence))[:MAX_CORE]
        structure = str(row.secondary_structure)[:MAX_CORE]
        if len(sequence) != len(structure):
            raise ValueError(f"truncated sequence/structure mismatch: {path}:{row.id}")
        records.append({
            "name": f"{source}:{row.id}", "source_id": str(row.id), "source": source,
            "seq": sequence, "structure": structure,
            "pairs": parse_dot_bracket(structure, nb_repo),
        })
    return records


def archive_records(archive: Path, official_list: Path, nb_repo: Path) -> list[dict]:
    selected = {PurePosixPath(line.strip()).name for line in official_list.read_text().splitlines() if line.strip()}
    if len(selected) != EXPECTED["archiveii600"]:
        raise ValueError(f"official ArchiveII600 list has {len(selected)} names")
    found = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            name = PurePosixPath(member.name).name
            if name not in selected:
                continue
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"missing tar stream: {member.name}")
            sequence, pairs = parse_bpseq(stream.read().decode("utf-8"), name)
            sequence = normalize(sequence)[:MAX_CORE]
            pairs = [[i, j] for i, j in pairs if i < MAX_CORE and j < MAX_CORE]
            structure = pairs_to_structure(len(sequence), pairs)
            recovered = parse_dot_bracket(structure, nb_repo)
            if recovered != sorted(pairs):
                raise ValueError(f"dot-bracket round-trip mismatch: {name}")
            found[name] = {
                "name": f"archiveii600:{Path(name).stem}", "source_id": Path(name).stem,
                "source": "BPfold_archiveII_bpseq+RNAErnie_archiveII600.lst",
                "seq": sequence, "structure": structure, "pairs": sorted(pairs),
            }
    if set(found) != selected:
        raise ValueError(f"ArchiveII600 identity mismatch: missing={len(selected-set(found))} extra={len(set(found)-selected)}")
    return [found[name] for name in sorted(found)]


def validate(records: list[dict], split: str) -> dict:
    if len(records) != EXPECTED[split]:
        raise ValueError(f"{split}: rows={len(records)} expected={EXPECTED[split]}")
    names = [r["name"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError(f"{split}: duplicate record names")
    pair_count = 0
    crossing_records = 0
    for record in records:
        length = len(record["seq"])
        if not 0 < length <= MAX_CORE or len(record["structure"]) != length:
            raise ValueError(f"{split}: invalid length: {record['name']}")
        pairs = sorted(record["pairs"])
        if pairs != record["pairs"]:
            raise ValueError(f"{split}: unsorted pairs: {record['name']}")
        if len({tuple(p) for p in pairs}) != len(pairs):
            raise ValueError(f"{split}: duplicate pairs: {record['name']}")
        used = set()
        for i, j in pairs:
            if not 0 <= i < j < length:
                raise ValueError(f"{split}: out-of-range pair: {record['name']}:{i}-{j}")
            if i in used or j in used:
                raise ValueError(f"{split}: conflicting pair: {record['name']}:{i}-{j}")
            used.update((i, j))
        pair_count += len(pairs)
        crossing_records += any(crosses(a, b) for k, a in enumerate(pairs) for b in map(tuple, pairs[k + 1:]))
    return {
        "rows": len(records), "unique_names": len(set(names)),
        "unique_sequences": len({r["seq"] for r in records}), "pairs": pair_count,
        "records_with_crossing_pairs": crossing_records,
        "min_length": min(map(lambda r: len(r["seq"]), records)),
        "max_length": max(map(lambda r: len(r["seq"]), records)),
        "sequence_sha256": hashlib.sha256("".join(r["seq"] + "\n" for r in records).encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/private/tmp/rnajev_audit"))
    parser.add_argument("--nb-repo", type=Path, default=Path("/private/tmp/rnajev_audit/NucleicBERT-official"))
    parser.add_argument("--archive-list", type=Path, default=Path("/private/tmp/rnajev_audit/archiveII600.lst"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir
    sources = {
        "rnastralign": root / "rnastralign_train.parquet",
        "tr0": root / "bprna-spot-0_train.parquet", "vl0": root / "bprna-spot-0_validation.parquet",
        "ts0": root / "bprna-spot-0_test.parquet", "archiveii": root / "BPfold_test_results.tar.gz",
        "archiveii600_list": args.archive_list,
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rnastralign = parquet_records(sources["rnastralign"], "rnastralign", args.nb_repo)
    tr0 = parquet_records(sources["tr0"], "bprna_tr0", args.nb_repo)
    splits = {
        "train_rnalign_tr0": rnastralign + tr0,
        "vl0": parquet_records(sources["vl0"], "bprna_vl0", args.nb_repo),
        "ts0": parquet_records(sources["ts0"], "bprna_ts0", args.nb_repo),
        "archiveii600": archive_records(sources["archiveii"], args.archive_list, args.nb_repo),
    }
    card = {
        "schema_version": 1, "status": "PASS", "protocol": "NucleicBERT secondary structure",
        "preprocessing": "uppercase and X/I->N; RNA U retained in JSONL and mapped U->T only by official tokenization; sequence and structure truncated to 598 nt; no deduplication, length filtering beyond the ArchiveII600 official list, noncanonical-pair removal, or pseudoknot removal",
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        "code_sha256": sha256(Path(__file__).resolve()), "splits": {},
    }
    for split, records in splits.items():
        summary = validate(records, split)
        path = args.out_dir / "jsonl" / f"{split}.jsonl"
        atomic_text(path, "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
        loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if validate(loaded, split) != summary:
            raise ValueError(f"{split}: saved-output validation mismatch")
        card["splits"][split] = {**summary, "path": str(path), "sha256": sha256(path)}
        print(f"PREPARED {split} {json.dumps(summary, sort_keys=True)}", flush=True)
    atomic_text(args.out_dir / "data_card.json", json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    checked = json.loads((args.out_dir / "data_card.json").read_text())
    if checked["status"] != "PASS" or set(checked["splits"]) != set(EXPECTED):
        raise ValueError("data-card verification failed")
    print(f"NB_SECSTR_DATA_PASS {args.out_dir / 'data_card.json'} sha256={sha256(args.out_dir / 'data_card.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
