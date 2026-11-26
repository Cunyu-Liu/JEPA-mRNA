#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATASET = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATASET_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
N_SEQUENCES = 5388
N_REFERENCE_PAIRS = 135184
ARTIFACT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/contextfold_official_recovery_20261126")
OFFICIAL_HOME = "http://www.cs.bgu.ac.il/~negevcb/contextfold/"
OFFICIAL_ARCHIVE_URL = OFFICIAL_HOME + "ContextFold_1_00.zip"
DEFAULT_MODEL = "trained/StHighCoHigh.model"


class ContractError(ValueError):
    pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: str | Path = DATASET, require_full: bool = True) -> list[dict]:
    path = Path(path)
    if require_full and sha256(path) != DATASET_SHA256:
        raise ContractError("bpRNA-new SHA256 mismatch; refusing to claim the frozen 5388-row contract")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = []
    for row in rows:
        name = row.get("name")
        seq = row.get("seq")
        if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
            raise ContractError("Every input needs a nonempty single-line name")
        if not isinstance(seq, str) or not seq or set(seq) - set("ACGU"):
            raise ContractError(f"Invalid RNA alphabet for {name!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ContractError("Duplicate input names are not allowed")
    if require_full:
        if len(rows) != N_SEQUENCES:
            raise ContractError(f"Expected {N_SEQUENCES} rows, found {len(rows)}")
        lengths = [len(row["seq"]) for row in rows]
        if (min(lengths), max(lengths)) != (33, 489):
            raise ContractError("Frozen bpRNA-new length contract mismatch")
        if sum(len(row["pairs"]) for row in rows) != N_REFERENCE_PAIRS:
            raise ContractError("Reference-pair identity mismatch")
    return rows


def native_input(rows: list[dict]) -> str:
    return "".join(f">{row['name']}\n{row['seq']}\n" for row in rows)


def dotbracket_pairs(structure: str, seq: str) -> list[list[int]]:
    if len(structure) != len(seq):
        raise ContractError("Predicted structure and sequence lengths differ")
    if set(structure) - set("()."):
        raise ContractError("ContextFold native output must be dot-bracket over ().")
    stack = []
    pairs = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ContractError("Unbalanced closing bracket")
            pairs.append([stack.pop(), index])
    if stack:
        raise ContractError("Unbalanced opening bracket")
    return sorted(pairs)


def parse_native_output(text: str, rows: list[dict]) -> list[dict]:
    items = []
    comments = []
    seq = None
    structures = []

    def finish():
        nonlocal seq, comments, structures
        if seq is None:
            return
        if len(structures) != 1:
            raise ContractError("Each prediction-only native item must contain exactly one structure line")
        items.append((comments, seq, structures[0]))
        comments, seq, structures = [], None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith((">", "#")):
            if seq is not None:
                finish()
            comments.append(line[1:].strip())
        elif seq is None:
            seq = line
        else:
            structures.append(line)
    finish()
    if len(items) != len(rows):
        raise ContractError(f"Native output has {len(items)} items for {len(rows)} inputs")
    predictions = []
    for row, (item_comments, seq, structure) in zip(rows, items):
        if not item_comments or item_comments[0] != row["name"]:
            raise ContractError("Native output name/order does not match the input")
        if seq != row["seq"]:
            raise ContractError("Native output changed the input sequence")
        predictions.append({
            "name": row["name"],
            "seq": seq,
            "pred_pairs": dotbracket_pairs(structure, seq),
        })
    return validate_prediction_rows(rows, predictions)


def validate_prediction_rows(rows: list[dict], predictions: list[dict]) -> list[dict]:
    if len(predictions) != len(rows):
        raise ContractError("Partial, summary-only, or aggregate output is not a prediction asset")
    validated = []
    for row, prediction in zip(rows, predictions):
        if set(prediction) != {"name", "seq", "pred_pairs"}:
            raise ContractError("Prediction rows must contain exactly name, seq and pred_pairs")
        if prediction["name"] != row["name"] or prediction["seq"] != row["seq"]:
            raise ContractError("Prediction name/sequence identity or order mismatch")
        seen_partners = set()
        canonical = []
        for pair in prediction["pred_pairs"]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ContractError("Each pair must have two indices")
            i, j = pair
            if type(i) is not int or type(j) is not int or not 0 <= i < j < len(row["seq"]):
                raise ContractError("Pairs must be zero-based inclusive i<j within the sequence")
            if i in seen_partners or j in seen_partners:
                raise ContractError("A nucleotide may occur in at most one pair")
            seen_partners.update((i, j))
            canonical.append([i, j])
        if len(canonical) != len({tuple(pair) for pair in canonical}):
            raise ContractError("Duplicate pairs are not allowed")
        canonical.sort()
        for index, (i, j) in enumerate(canonical):
            for k, l in canonical[index + 1:]:
                if i < k < j < l:
                    raise ContractError("ContextFold output must not contain crossing pairs")
        validated.append({"name": row["name"], "seq": row["seq"], "pred_pairs": canonical})
    return validated
