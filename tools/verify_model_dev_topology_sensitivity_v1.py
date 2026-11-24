#!/usr/bin/env python3
"""Non-importing raw-TR1c replay of the frozen topology sensitivity artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": sha256(path)}


def rows(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ids(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def signature(length: int, raw_pairs) -> tuple[str, int]:
    pair_set = {tuple(sorted((int(i), int(j)))) for i, j in raw_pairs}
    partner = {}
    for i, j in pair_set:
        if not 0 <= i < j < length or i in partner or j in partner:
            raise ValueError("invalid or conflicting pair")
        partner[i], partner[j] = j, i
    ends = []
    for i, j in sorted(pair_set, key=lambda pair: (pair[0], -pair[1])):
        while ends and i > ends[-1]:
            ends.pop()
        if ends and j > ends[-1]:
            raise ValueError("crossing pairs have undefined nested topology")
        ends.append(j)
    starts = []
    for i, j in pair_set:
        if (i - 1, j + 1) in pair_set:
            continue
        size = 0
        while (i + size, j - size) in pair_set:
            size += 1
        starts.append({"outer": (i, j), "inner": (i + size - 1, j - size + 1),
                       "size": size, "children": []})
    roots = []
    active = []
    for helix in sorted(starts, key=lambda value: (value["outer"][0], -value["outer"][1])):
        while active and helix["outer"][1] > active[-1]["outer"][1]:
            active.pop()
        if active:
            active[-1]["children"].append(helix)
        else:
            roots.append(helix)
        active.append(helix)

    def freeze(helix):
        a, b = helix["inner"]
        children = sorted(helix["children"], key=lambda value: value["outer"])
        if not children:
            gaps, kind = [b - a - 1], "HAIRPIN"
        else:
            gaps = [children[0]["outer"][0] - a - 1]
            gaps += [right["outer"][0] - left["outer"][1] - 1
                     for left, right in zip(children, children[1:])]
            gaps += [b - children[-1]["outer"][1] - 1]
            if len(children) == 1:
                kind = "STACK" if gaps == [0, 0] else "BULGE" if 0 in gaps else "INTERNAL"
            else:
                kind = "MULTILOOP"
        if min(gaps) < 0:
            raise ValueError("inconsistent helix nesting")
        return {"helix_length": helix["size"], "loop_type": kind, "loop_gaps": gaps,
                "children": [freeze(child) for child in children]}

    frozen = [freeze(root) for root in roots]
    if roots:
        exterior = [roots[0]["outer"][0]]
        exterior += [right["outer"][0] - left["outer"][1] - 1 for left, right in zip(roots, roots[1:])]
        exterior += [length - 1 - roots[-1]["outer"][1]]
    else:
        exterior = [length]
    canonical = json.dumps({"exterior_gaps": exterior, "roots": frozen}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), len(pair_set)


def concentration(records: list[dict]) -> dict:
    output = {}
    for label, key_function in (("family", lambda row: row["family_id"] or row["family_status"]),
                                ("cluster80", lambda row: str(row["cluster_id"]))):
        counts = {}
        for row in records:
            key = key_function(row)
            cell = counts.setdefault(key, {"n_sequences": 0, "n_pairs": 0})
            cell["n_sequences"] += 1
            cell["n_pairs"] += row["n_pairs"]
        n_sequences = sum(value["n_sequences"] for value in counts.values())
        n_pairs = sum(value["n_pairs"] for value in counts.values())
        top_sequence = min(counts, key=lambda key: (-counts[key]["n_sequences"], key)) if counts else None
        top_pair = min(counts, key=lambda key: (-counts[key]["n_pairs"], key)) if counts else None
        output[label] = {
            "n_groups": len(counts), "n_sequences": n_sequences, "n_pairs": n_pairs,
            "top_group_by_sequences": top_sequence,
            "top_group_sequence_fraction": counts[top_sequence]["n_sequences"] / n_sequences if n_sequences else None,
            "sequence_hhi": sum((value["n_sequences"] / n_sequences) ** 2 for value in counts.values()) if n_sequences else None,
            "top_group_by_pairs": top_pair,
            "top_group_pair_fraction": counts[top_pair]["n_pairs"] / n_pairs if n_pairs else None,
            "pair_hhi": sum((value["n_pairs"] / n_pairs) ** 2 for value in counts.values()) if n_pairs else None,
            "counts": {key: counts[key] for key in sorted(counts)},
        }
    return output


def verify(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["input_identities"]
    source_checks = {name: identity(Path(asset["path"])) == asset for name, asset in assets.items()}
    registry = json.loads(Path(assets["registry"]["path"]).read_text(encoding="utf-8"))
    raw_asset = next(row for row in registry["assets"] if row.get("dataset_id") == "bprna_tr1c"
                     and row.get("role") == "legal_train_dev_source")
    raw_path = Path(raw_asset["source"])
    panel_path = Path(registry["environment_contracts"]["panel_records"]["path"])
    panel = {row["sequence_id"]: row for row in rows(panel_path)}
    train_ids = ids(Path(assets["train_ids"]["path"]))
    dev_ids = ids(Path(assets["dev_ids"]["path"]))
    train_set, dev_set = set(train_ids), set(dev_ids)
    signatures = {}
    raw_hashes = {"train": set(), "dev": set()}
    for index, raw in enumerate(rows(raw_path)):
        sequence_id = f"bprna_tr1c:row{index:06d}:{raw['name']}"
        sequence = str(raw.get("seq", raw.get("sequence"))).upper().replace("T", "U")
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        digest, n_pairs = signature(len(sequence), raw["pairs"])
        signatures[sequence_id] = digest
        metadata = panel[sequence_id]
        if (metadata["source_row_index"], metadata["name"], metadata["length"],
                metadata["sequence_sha256"], metadata["n_pairs"]) != (
                index, raw["name"], len(sequence), sequence_hash, n_pairs):
            raise ValueError(f"raw/panel mismatch: {sequence_id}")
        raw_hashes[metadata["cluster_assignment"]].add(sequence_hash)
    train_counts = Counter(signatures[value] for value in train_ids)
    dev_signatures = {signatures[value] for value in dev_ids}
    shared = set(train_counts) & dev_signatures
    expected_dev = []
    for sequence_id in sorted(dev_ids):
        metadata = panel[sequence_id]
        digest = signatures[sequence_id]
        matched = digest in train_counts
        expected_dev.append({
            "sequence_id": sequence_id, "source_row_index": metadata["source_row_index"],
            "name": metadata["name"], "sequence_sha256": metadata["sequence_sha256"],
            "length": metadata["length"], "n_pairs": metadata["n_pairs"],
            "family_id": metadata.get("family_id"), "family_status": metadata.get("family_status", "UNCHECKED"),
            "cluster_id": int(metadata["cluster_id"]), "topology_signature_sha256": digest,
            "exact_topology_match_with_planned_train": matched,
            "planned_train_records_with_signature": train_counts[digest],
            "own_clean_state": "EXACT" if matched else "NOT_DETECTED_UNKNOWN",
            "matched_clean_state": "EXACT" if matched else "NOT_DETECTED_UNKNOWN",
            "retained_own_clean": not matched, "retained_matched_clean": not matched,
            "approximate_structural_similarity": "UNCHECKED",
        })
    flagged = [row for row in expected_dev if row["exact_topology_match_with_planned_train"]]
    clean = [row for row in expected_dev if not row["exact_topology_match_with_planned_train"]]
    flagged_train = sorted(value for value in train_ids if signatures[value] in shared)
    expected_shared = []
    for digest in sorted(shared):
        train_group = [value for value in train_ids if signatures[value] == digest]
        dev_group = [row for row in flagged if row["topology_signature_sha256"] == digest]
        expected_shared.append({
            "topology_signature_sha256": digest, "n_train_records": len(train_group),
            "n_train_pairs": sum(panel[value]["n_pairs"] for value in train_group),
            "n_dev_records": len(dev_group), "n_dev_pairs": sum(row["n_pairs"] for row in dev_group),
            "train_sequence_ids_sha256": hashlib.sha256("\n".join(sorted(train_group)).encode()).hexdigest(),
            "dev_sequence_ids_sha256": hashlib.sha256("\n".join(sorted(row["sequence_id"] for row in dev_group)).encode()).hexdigest(),
        })
    expected_train_signatures = [{"topology_signature_sha256": digest, "n_train_records": train_counts[digest]}
                                 for digest in sorted(train_counts)]
    artifact_rows = list(rows(root / "dev_topology_records.jsonl"))
    actual_shared = list(rows(root / "shared_topology_signatures.jsonl"))
    actual_train_signatures = list(rows(root / "train_topology_signature_counts.jsonl"))
    concentrations = json.loads((root / "concentration.json").read_text(encoding="utf-8"))
    expected_scopes = {
        "raw": {"rows": expected_dev, "flagged": flagged},
        "own_clean": {"rows": clean, "flagged": []},
        "matched_clean": {"rows": clean, "flagged": []},
    }
    scope_checks = {}
    for name, expected in expected_scopes.items():
        selected = expected["rows"]
        excluded = expected["flagged"]
        actual = manifest["scopes"][name]
        expected_summary = {
            "scope": name, "n_sequences": len(selected), "n_pairs": sum(row["n_pairs"] for row in selected),
            "n_unique_topology_signatures": len({row["topology_signature_sha256"] for row in selected}),
            "n_exact_topology_overlap_sequences": len(excluded),
            "n_exact_topology_overlap_pairs": sum(row["n_pairs"] for row in excluded),
        }
        scope_checks[name] = (all(actual[key] == value for key, value in expected_summary.items())
                              and concentrations[name] == concentration(selected)
                              and actual["leakage_state"]["exact_loop_helix_topology_signature"] == (
                                  "EXACT" if excluded else "NOT_DETECTED_UNKNOWN")
                              and actual["leakage_state"]["approximate_structural_similarity"] == "UNCHECKED")
    artifact_manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_checks = {name: identity(Path(asset["path"])) == asset
                       for name, asset in artifact_manifest["artifacts"].items()}
    phase2_check = True
    if manifest.get("phase2_crosscheck") is not None:
        phase2_path = Path(manifest["phase2_crosscheck"]["identity"]["path"])
        evidence = json.loads(phase2_path.read_text(encoding="utf-8"))["comparisons"]["tr1c_registered_train_vs_dev"]["evidence"]["exact_structure_signature"]
        phase2_check = (identity(phase2_path) == manifest["phase2_crosscheck"]["identity"]
                        and evidence["n_shared"] == len(shared)
                        and evidence["n_left_records_with_shared"] == len(flagged_train)
                        and evidence["n_right_records_with_shared"] == len(flagged))
    checks = {
        **{f"source:{name}": value for name, value in source_checks.items()},
        "raw_exactly_covers_panel": set(signatures) == set(panel),
        "train_dev_id_disjoint": not train_set & dev_set,
        "exact_sequence_disjoint": not raw_hashes["train"] & raw_hashes["dev"],
        "cluster80_disjoint": not ({int(panel[value]["cluster_id"]) for value in train_ids}
                                    & {int(panel[value]["cluster_id"]) for value in dev_ids}),
        "dev_rows_replayed": artifact_rows == expected_dev,
        "shared_signatures_replayed": actual_shared == expected_shared,
        "train_signature_counts_replayed": actual_train_signatures == expected_train_signatures,
        "raw_ids": ids(root / "raw_cluster80_dev_ids.txt") == sorted(dev_ids),
        "own_clean_ids": ids(root / "topology_matched_own_clean_dev_ids.txt") == [row["sequence_id"] for row in clean],
        "matched_clean_ids": ids(root / "topology_matched_clean_dev_ids.txt") == [row["sequence_id"] for row in clean],
        "excluded_dev_ids": ids(root / "exact_topology_overlap_dev_ids.txt") == [row["sequence_id"] for row in flagged],
        "matched_train_ids": ids(root / "exact_topology_overlap_train_ids.txt") == flagged_train,
        **{f"scope:{name}": value for name, value in scope_checks.items()},
        **{f"artifact:{name}": value for name, value in artifact_checks.items()},
        "phase2_crosscheck": phase2_check,
        "shared_signature_count": manifest["n_shared_topology_signatures"] == len(shared),
        "no_primary_gate_change": manifest["primary_gates_changed"] is False,
        "no_approximate_similarity_claim": manifest["not_approximate_structural_dissimilarity"] is True,
    }
    return {
        "schema_version": "foldnar.model_dev_topology_sensitivity.independent_verification.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": "non-importing raw-TR1c replay with independent signature implementation",
        "manifest": identity(manifest_path), "verifier": identity(Path(__file__).resolve()),
        "n_train_sequences": len(train_ids), "n_dev_sequences": len(dev_ids),
        "n_shared_topology_signatures": len(shared), "n_excluded_dev_sequences": len(flagged),
        "n_excluded_dev_pairs": sum(row["n_pairs"] for row in flagged),
        "n_retained_dev_sequences": len(clean), "n_retained_dev_pairs": sum(row["n_pairs"] for row in clean),
        "n_planned_train_records_with_shared_signature": len(flagged_train),
        "n_checks": len(checks), "checks": checks,
        "scope": "Exact signature replay only; approximate structural similarity, broader homology, pretraining and temporal lineage remain unchecked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.out}")
    result = verify(args.manifest)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
