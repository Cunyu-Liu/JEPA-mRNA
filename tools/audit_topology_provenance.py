#!/usr/bin/env python3
"""Audit exact loop-helix topology overlap and remaining provenance gaps.

This is a read-only data audit. Exact topology overlap is not a structural-
similarity search, and a model-card corpus list is not an exposure manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

DATASETS = [
    "archiveii_embok_clean", "bprna_new", "bprna_ts0", "ref_pdb_ts1",
    "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "ref_synthetic_test",
    "stransfer", "pdb669", "rfam_fam", "rfam_temporal",
    "chanrg_train_train", "chanrg_train_validation", "chanrg_test",
    "chanrg_gena", "chanrg_genc", "chanrg_genf",
]
DECLARED_PRETRAINING_SOURCES = [
    "multimolecule/rnacentral", "multimolecule/rfam",
    "multimolecule/ensembl-genome-browser", "multimolecule/nucleotide",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sequence(value: str) -> str:
    return str(value).upper().replace("T", "U")


def record_name(record: dict) -> str:
    return str(record.get("name", record.get("id", "")))


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validated_pairs(record: dict) -> tuple[str, tuple[tuple[int, int], ...]]:
    seq = normalize_sequence(record.get("seq", record.get("sequence", "")))
    pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in record.get("pairs", [])})
    partners = {}
    for i, j in pairs:
        if not 0 <= i < j < len(seq):
            raise ValueError(f"invalid 0-based pair {(i, j)} for length {len(seq)}")
        if i in partners or j in partners:
            raise ValueError(f"partner conflict at pair {(i, j)}")
        partners[i], partners[j] = j, i
    stack = []
    for i, j in sorted(pairs, key=lambda pair: (pair[0], -pair[1])):
        while stack and i > stack[-1]:
            stack.pop()
        if stack and j > stack[-1]:
            raise ValueError(f"crossing pair {(i, j)}; exact nested topology is undefined")
        stack.append(j)
    return seq, tuple(pairs)


def topology(record: dict) -> dict:
    """Return an ordered, exact loop-helix topology representation.

    Consecutive stacked pairs are compressed into helices. Each helix stores
    its length, ordered loop gaps and direct child helices; the virtual root
    stores exterior gaps and roots. Sequence identity and pair coordinates are
    intentionally absent from the signature itself.
    """
    seq, pairs = validated_pairs(record)
    pair_set = set(pairs)
    helices = []
    for i, j in pairs:
        if (i - 1, j + 1) in pair_set:
            continue
        length = 0
        while (i + length, j - length) in pair_set:
            length += 1
        helices.append({
            "outer": (i, j), "inner": (i + length - 1, j - length + 1),
            "length": length, "children": [],
        })
    roots = []
    stack = []
    for helix in sorted(helices, key=lambda item: (item["outer"][0], -item["outer"][1])):
        j = helix["outer"][1]
        while stack and j > stack[-1]["outer"][1]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(helix)
        else:
            roots.append(helix)
        stack.append(helix)

    loop_counts = Counter()
    max_multiloop_order = 0

    def freeze_helix(helix: dict) -> dict:
        nonlocal max_multiloop_order
        a, b = helix["inner"]
        children = sorted(helix["children"], key=lambda item: item["outer"])
        if not children:
            gaps = [b - a - 1]
            loop_type, order = "HAIRPIN", 0
        else:
            gaps = [children[0]["outer"][0] - a - 1]
            gaps.extend(children[k]["outer"][0] - children[k - 1]["outer"][1] - 1
                        for k in range(1, len(children)))
            gaps.append(b - children[-1]["outer"][1] - 1)
            if len(children) == 1:
                left, right = gaps
                loop_type = "STACK" if left == right == 0 else "BULGE" if left == 0 or right == 0 else "INTERNAL"
                order = 1
            else:
                loop_type, order = "MULTILOOP", len(children) + 1
        if min(gaps) < 0:
            raise ValueError("negative loop gap; helix hierarchy is inconsistent")
        loop_counts[loop_type] += 1
        max_multiloop_order = max(max_multiloop_order, order)
        return {
            "helix_length": helix["length"], "loop_type": loop_type,
            "loop_gaps": gaps, "children": [freeze_helix(child) for child in children],
        }

    frozen_roots = [freeze_helix(root) for root in roots]
    exterior_gaps = []
    if roots:
        exterior_gaps.append(roots[0]["outer"][0])
        exterior_gaps.extend(roots[k]["outer"][0] - roots[k - 1]["outer"][1] - 1
                             for k in range(1, len(roots)))
        exterior_gaps.append(len(seq) - 1 - roots[-1]["outer"][1])
    else:
        exterior_gaps.append(len(seq))
    signature = {"exterior_gaps": exterior_gaps, "roots": frozen_roots}
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return {
        "signature_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "signature": signature,
        "n_pairs": len(pairs), "n_helices": len(helices),
        "n_exterior_roots": len(roots), "loop_type_counts": dict(sorted(loop_counts.items())),
        "max_multiloop_order": max_multiloop_order,
    }


def audit(data_root: Path, out: Path, model_card: Path) -> dict:
    if out.exists():
        raise ValueError(f"refusing to overwrite artifact directory: {out}")
    train_path = data_root / "bprna_tr1c.jsonl"
    if not train_path.is_file():
        raise ValueError(f"missing training data: {train_path}")
    if not model_card.is_file():
        raise ValueError(f"missing local RiNALMo model card: {model_card}")
    out.mkdir(parents=True)
    train_counts = Counter()
    for record in read_jsonl(train_path):
        train_counts[topology(record)["signature_sha256"]] += 1
    train_signatures_path = out / "train_topology_signatures.jsonl"
    with train_signatures_path.open("w") as handle:
        for signature, count in sorted(train_counts.items()):
            handle.write(json.dumps({"signature_sha256": signature, "n_records": count}, sort_keys=True) + "\n")
    overlaps_path = out / "topology_overlap_records.jsonl"
    summaries = {}
    source_files = {"bprna_tr1c": {"path": str(train_path), "sha256": sha256_file(train_path)}}
    with overlaps_path.open("w") as overlap_handle:
        for name in DATASETS:
            path = data_root / f"{name}.jsonl"
            if not path.is_file() or path.stat().st_size == 0:
                summaries[name] = {"status": "BLOCKED_DATA", "path": str(path)}
                continue
            source_files[name] = {"path": str(path), "sha256": sha256_file(path)}
            n_records = 0
            n_overlap = 0
            unique_signatures = set()
            overlap_signatures = set()
            loop_counts = Counter()
            for record in read_jsonl(path):
                result = topology(record)
                n_records += 1
                signature = result["signature_sha256"]
                unique_signatures.add(signature)
                loop_counts.update(result["loop_type_counts"])
                if signature not in train_counts:
                    continue
                n_overlap += 1
                overlap_signatures.add(signature)
                seq = normalize_sequence(record.get("seq", record.get("sequence", "")))
                row = {
                    "dataset_id": name, "name": record_name(record),
                    "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                    "topology_sha256": signature, "n_pairs": result["n_pairs"],
                    "n_helices": result["n_helices"],
                    "train_records_with_signature": train_counts[signature],
                }
                overlap_handle.write(json.dumps(row, sort_keys=True) + "\n")
            summaries[name] = {
                "status": "EXACT" if n_overlap else "NOT_DETECTED_UNKNOWN",
                "layer": "exact_loop_helix_topology_signature_vs_TR1c",
                "n_records": n_records, "n_records_with_exact_topology_overlap": n_overlap,
                "fraction_records_with_exact_topology_overlap": n_overlap / n_records,
                "n_unique_topology_signatures": len(unique_signatures),
                "n_unique_signatures_overlapping_TR1c": len(overlap_signatures),
                "loop_type_counts": dict(sorted(loop_counts.items())),
                "structure_similarity_search": "UNCHECKED",
                "interpretation": "Exact signature overlap only; non-overlap is not evidence of topological independence.",
            }
    document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_EXACT_TOPOLOGY_AUDIT_WITH_UNCHECKED_LAYERS",
        "scope": "Read-only data audit; no external performance, model, decoder, checkpoint, or route selection",
        "signature_contract": {
            "coordinates": "0-based noncrossing pairs; stacked pairs compressed into helices",
            "features": ["helix length", "ordered loop gaps", "loop type", "direct child helices", "exterior gaps and roots"],
            "excluded": ["sequence identity", "absolute pair coordinates", "family labels"],
            "nonmatch_status": "NOT_DETECTED_UNKNOWN",
        },
        "source_files": source_files,
        "train_signature_file": str(train_signatures_path),
        "train_signature_file_sha256": sha256_file(train_signatures_path),
        "train_n_records": sum(train_counts.values()),
        "train_n_unique_signatures": len(train_counts),
        "overlap_file": str(overlaps_path),
        "overlap_file_sha256": sha256_file(overlaps_path),
        "datasets": summaries,
        "family_clan_provenance": {
            "TR1c_Rfam_PDB": "BLOCKED_PROVENANCE: aggregated JSONL has no structured family/clan labels",
            "CHANRG": "Checked separately in scientific_coverage_audit_v3 using declared family/clan fields",
        },
        "pretraining_exposure": {
            "status": "UNCHECKED",
            "model_card": str(model_card), "model_card_sha256": sha256_file(model_card),
            "declared_sources": DECLARED_PRETRAINING_SOURCES,
            "declared_n_unique_ncrna_sequences": 36000000,
            "declared_n_mmseqs_clusters": 17000000,
            "reason": "The local model card declares source databases but supplies no per-sequence training manifest; no exposure label is inferred.",
        },
        "remaining_unchecked": ["structural-similarity search", "per-sequence pretraining exposure", "temporal and label provenance for non-CHANRG sets"],
        "old_external_panels_used_for_selection": False,
        "production_repair": False,
        "sota_claim": False,
        "code": str(Path(__file__).resolve()), "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (out / "topology_provenance_audit.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    parser.add_argument("--model-card", type=Path, default=Path("/root/rna_jepa_data/weights/rinalmo-giga/README.md"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = audit(args.data_root, args.out, args.model_card)
    print(json.dumps({
        "status": document["status"], "out": str(args.out),
        "train_n_records": document["train_n_records"],
        "train_n_unique_signatures": document["train_n_unique_signatures"],
        "datasets": {name: {key: value for key, value in summary.items() if key in
                            ("status", "n_records", "n_records_with_exact_topology_overlap")}
                     for name, summary in document["datasets"].items()},
        "pretraining_exposure": document["pretraining_exposure"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
