#!/usr/bin/env python3
"""Close Rfam family/clan/release lineage using official releases, not pairs."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path

SCHEMA = "foldnar.rfam_release_lineage.v1"
PANELS = ("rfam_fam", "rfam_temporal")
REGION_RE = re.compile(r"([A-Za-z0-9._]+)_([0-9]+)-([0-9]+)\.bpseq$")
RELEASES = ("14.9", "14.10", "15.0")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def region_key(name):
    match = REGION_RE.fullmatch(name)
    return None if match is None else f"{match.group(1)}/{match.group(2)}-{match.group(3)}"


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fasta_hits(path, target_sequences, target_keys):
    hits = collections.defaultdict(set)
    key, chunks = None, []

    def consume(key, chunks):
        if key is None:
            return
        sequence = "".join(chunks).upper().replace("T", "U")
        if key in target_keys or sequence in target_sequences:
            hits[sequence].add(key)

    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                consume(key, chunks)
                key, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    consume(key, chunks)
    return {sequence: sorted(keys) for sequence, keys in hits.items()}


def family_table(path):
    families = {}
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            families[fields[0]] = {"rfam_id": fields[1], "description": fields[3], "type": fields[17],
                                   "created": fields[32], "updated": fields[33]}
    return families


def clan_tables(clan_path, membership_path):
    clans, membership = {}, collections.defaultdict(set)
    with gzip.open(clan_path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            clans[fields[0]] = {"clan_id": fields[1], "description": fields[3] if len(fields) > 3 else None}
    with gzip.open(membership_path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            membership[fields[1]].add(fields[0])
    return clans, membership


def seed_alignments(path):
    families, exact = {}, collections.defaultdict(set)
    current = None
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("//"):
                current = None
            elif line.startswith("#=GF AC "):
                accession = line.split()[-1]
                current = {"rfam_acc": accession, "clan_acc": None, "seed_regions": set(), "ss_cons": []}
                families[accession] = current
            elif current is not None and line.startswith("#=GF CL "):
                current["clan_acc"] = line.split()[-1]
            elif current is not None and line.startswith("#=GC SS_cons"):
                current["ss_cons"].append(line.split()[-1])
            elif current is not None and line.strip() and not line.startswith("#") and not line.startswith("STOCKHOLM"):
                fields = line.split()
                if len(fields) == 2:
                    key, sequence = fields
                    current["seed_regions"].add(key)
                    exact[sequence.replace(".", "").replace("-", "").upper().replace("T", "U")].add((current["rfam_acc"], key))
    return {"families": families, "exact_sequence_regions": exact}


def full_region_families(path, wanted_keys):
    mapping = collections.defaultdict(set)
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            key = f"{fields[1]}/{fields[2]}-{fields[3]}"
            if key in wanted_keys:
                mapping[key].add(fields[0])
    return mapping


def family_first_release(presence):
    if presence.get("14.9"):
        return "14.9_OR_EARLIER"
    if presence.get("14.10"):
        return "14.10"
    if presence.get("15.0"):
        return "15.0"
    return "UNCHECKED"


def release_root(root, release):
    return root / release


def command_map(args):
    rows = {name: read_jsonl(args.jsonl_root / f"{name}.jsonl") for name in PANELS + ("bprna_tr1c",)}
    target_sequences = {row["seq"].upper().replace("T", "U") for values in rows.values() for row in values}
    target_keys = {key for name in PANELS for row in rows[name] if (key := region_key(row["name"]))}
    sources, hits, seed, region_maps, families, clans = {}, {}, {}, {}, {}, {}
    for release in RELEASES:
        root = release_root(args.releases, release)
        if release == "14.9":
            fasta = args.rfam14_root / "Rfam.fa.gz"
            full_region = args.rfam14_root / "rfam14_9_metadata" / "full_region.txt.gz"
            family = args.rfam14_root / "rfam14_9_metadata" / "family.txt.gz"
        else:
            fasta = args.rfam15_root / "Rfam.fa.gz" if release == "15.0" else None
            full_region = None if release == "14.10" else args.rfam15_root / "full_region.txt.gz"
            family = root / "family.txt.gz" if release == "14.10" else args.rfam15_root / "family.txt.gz"
        sources[release] = {"seed": sha256_file(root / "Rfam.seed.gz"),
                            "family": sha256_file(family),
                            "full_region": sha256_file(full_region) if full_region else "NOT_USED_14_10_RELEASE_FAMILY_AND_ALIGNMENT_ONLY"}
        if fasta:
            sources[release]["fasta"] = sha256_file(fasta)
            hits[release] = fasta_hits(fasta, target_sequences, target_keys)
        else:
            hits[release] = {}
        seed[release] = seed_alignments(root / "Rfam.seed.gz")
        wanted = {key for values in hits[release].values() for key in values} | target_keys
        region_maps[release] = full_region_families(full_region, wanted) if full_region else {}
        families[release] = family_table(family)
        clans[release] = clan_tables(root / "clan.txt.gz", root / "clan_membership.txt.gz")

    output_rows = []
    for panel in PANELS:
        for row_index, row in enumerate(rows[panel]):
            sequence = row["seq"].upper().replace("T", "U")
            key = region_key(row["name"])
            per_release, family_votes = {}, collections.defaultdict(set)
            for release in RELEASES:
                exact_keys = hits[release].get(sequence, [])
                seed_hits = sorted(seed[release]["exact_sequence_regions"].get(sequence, set()))
                votes = set()
                for exact_key in exact_keys:
                    votes |= region_maps[release].get(exact_key, set())
                for family, seed_key in seed_hits:
                    votes.add(family)
                direct_families = region_maps[release].get(key, set()) if key else set()
                direct_sequence_match = bool(key and (key in exact_keys or key in {seed_key for _, seed_key in seed_hits}))
                if direct_sequence_match:
                    votes |= direct_families
                family_votes[release] = votes
                per_release[release] = {
                    "exact_fasta": bool(exact_keys), "exact_keys": exact_keys,
                    "direct_region": key, "direct_families": sorted(direct_families),
                    "direct_fasta_match": bool(key and key in exact_keys),
                    "direct_sequence_match": direct_sequence_match,
                    "seed_exact_regions": [{"family": family, "region": seed_key} for family, seed_key in seed_hits],
                    "family_votes": sorted(votes),
                }
            unique15 = len(family_votes["15.0"]) == 1
            family = next(iter(family_votes["15.0"])) if unique15 else None
            if unique15:
                status = "UNIQUE_FAMILY_EXACT_SEQUENCE_REGION_SEED"
            elif family_votes["15.0"]:
                status = "AMBIGUOUS_FAMILY_UNCHECKED"
            else:
                status = "FAMILY_UNCHECKED"
            first_family = family_first_release({release: family in families[release] for release in RELEASES}) if family else "UNCHECKED"
            clan_acc = sorted(clans["15.0"][1].get(family, set())) if family else []
            output_rows.append({
                "dataset": panel, "row_index": row_index, "source_record_id": row["name"],
                "sequence_sha256": sha256_text(sequence), "direct_region_key": key,
                "family": family, "family_status": status, "family_votes_by_release": {r: sorted(v) for r, v in family_votes.items()},
                "family_first_release": first_family,
                "family_release_presence": {release: {"family_table": family in families[release],
                                                       "seed_alignment": family in seed[release]["families"]} for release in RELEASES},
                "family_release_evidence_scope": "accession presence in official family table determines first observed release; seed is independent crosscheck; created/updated timestamps are not first-release proof",
                "clan_accessions": clan_acc,
                "clans": {acc: clans["15.0"][0].get(acc, {}) for acc in clan_acc},
                "sequence_first_release": "14.9_OR_EARLIER" if per_release["14.9"]["exact_fasta"] else "14.10_TO15.0_PENDING_PER_FAMILY_ALIGNMENT" if per_release["15.0"]["exact_fasta"] else "UNCHECKED",
                "per_release": per_release,
                "label_lineage": "COMPARATIVE_INFERRED_NOT_EXPERIMENTAL",
                "identity_pairs_used": False,
            })
    with (args.output / "release_lineage.jsonl").open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"schema_version": SCHEMA, "n_records": len(output_rows),
               "family_status_counts": dict(collections.Counter(row["family_status"] for row in output_rows)),
               "family_first_release_counts": dict(collections.Counter(row["family_first_release"] for row in output_rows)),
               "source_sha256": sources, "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "release_summary.json", summary)
    needed = sorted({row["family"] for row in output_rows
                     if row["dataset"] == "rfam_temporal" and row["family"] and not row["per_release"]["14.9"]["exact_fasta"]})
    write_json(args.output / "required_14_10_alignments.json", {"families": needed,
               "scope": "Only families needed for temporal sequence-first-release closure; no pair-based selection"})
    print(json.dumps({"status": "PASS_RELEASE_LINEAGE_MAP_WITH_UNCHECKED", **summary}, indent=2))


def download_14_10_family(family, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{family}.fa.gz"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://ftp.ebi.ac.uk/pub/databases/Rfam/14.10/fasta_files/{family}.fa.gz"
    temporary = path.with_suffix(".fa.gz.part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 2:
                return None
    return None


def command_novelty(args):
    rows = [json.loads(line) for line in (args.lineage / "release_lineage.jsonl").open() if line.strip()]
    temporal = [row for row in rows if row["dataset"] == "rfam_temporal"]
    needed = sorted({row["family"] for row in temporal if row["family"] and not row["per_release"]["14.9"]["exact_fasta"]})
    hits14_10 = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda family: download_14_10_family(family, args.output / "alignments_14_10"), needed))
    for family, path in zip(needed, paths):
        if path is None:
            hits14_10[family] = None
            continue
        target_sequences = {row["source_record_id"]: row["sequence_sha256"] for row in temporal if row["family"] == family}
        sequences = {row["seq"].upper().replace("T", "U") for row in read_jsonl(args.jsonl_root / "rfam_temporal.jsonl")
                     if sha256_text(row["seq"].upper().replace("T", "U")) in target_sequences.values()}
        keys = {row["direct_region_key"] for row in temporal if row["family"] == family}
        hits14_10[family] = fasta_hits(path, sequences, keys)
    train_rows = read_jsonl(args.jsonl_root / "bprna_tr1c.jsonl")
    train_sequences = {row["seq"].upper().replace("T", "U") for row in train_rows}
    train_families = set()
    with gzip.open(args.tr1c_family_mapping, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["dataset"] == "bprna_tr1c" and row["status"] == "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY":
                train_families.update(row["rfam_families"])
    previous = json.loads(args.previous_overlap.read_text())
    cross = collections.defaultdict(list)
    for group in previous["cross_panel_exact_sequence_groups"]:
        for member in group["records"]:
            if member["dataset"] in PANELS:
                cross[(member["dataset"], member["source_record_id"])] = [item for item in group["records"] if item["dataset"] not in PANELS]
    novel_rows = []
    sources = {name: read_jsonl(args.jsonl_root / f"{name}.jsonl") for name in PANELS}
    for row in rows:
        source = sources[row["dataset"]][row["row_index"]]
        sequence = source["seq"].upper().replace("T", "U")
        tr1c_exact = sequence in train_sequences
        cross_exact = cross.get((row["dataset"], row["source_record_id"]), [])
        family_overlap = bool(row["family"] and row["family"] in train_families)
        sequence_first = "UNCHECKED"
        if row["per_release"]["14.9"]["exact_fasta"]:
            sequence_first = "14.9_OR_EARLIER"
        elif row["dataset"] == "rfam_temporal" and row["family"]:
            family_hits = hits14_10.get(row["family"], {})
            if family_hits is None:
                sequence_first = "UNCHECKED"
            elif sequence in family_hits:
                sequence_first = "14.10"
            elif row["per_release"]["15.0"]["exact_fasta"]:
                sequence_first = "15.0"
        if row["dataset"] != "rfam_temporal":
            sequence_first = "14.9_OR_EARLIER" if row["per_release"]["14.9"]["exact_fasta"] else "UNCHECKED"
        if row["per_release"]["14.9"]["exact_fasta"]:
            novelty = "NOT_NOVEL_PRE14_9_SEQUENCE"
        elif tr1c_exact:
            novelty = "NOT_INDEPENDENT_TR1C_EXACT"
        elif cross_exact:
            novelty = "NOT_INDEPENDENT_CROSS_PANEL_EXACT"
        elif row["dataset"] == "rfam_temporal" and sequence_first in {"14.10", "15.0"} and row["family"]:
            novelty = "NOVEL_SEQUENCE_RELEASE_SUPPORTED"
        else:
            novelty = "UNCHECKED"
        novel_rows.append({**row, "sequence_first_release": sequence_first,
                           "tr1c_exact_overlap": tr1c_exact, "tr1c_family_overlap": family_overlap,
                           "cross_panel_exact_overlap": cross_exact,
                           "temporal_novelty_layer": novelty,
                           "family_novelty_layer": row["family_first_release"] if row["family_first_release"] in {"14.10", "15.0"} else "NOT_NEW_FAMILY" if row["family_first_release"] == "14.9_OR_EARLIER" else "UNCHECKED"})
    with (args.output / "temporal_lineage.jsonl").open("w") as handle:
        for row in novel_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    strata = collections.defaultdict(list)
    for row in novel_rows:
        if row["dataset"] == "rfam_temporal":
            if row["per_release"]["14.9"]["exact_fasta"]:
                strata["pre14_9_exact"].append(row["source_record_id"])
            if row["tr1c_exact_overlap"]:
                strata["tr1c_exact"].append(row["source_record_id"])
            if row["cross_panel_exact_overlap"]:
                strata["cross_panel_exact"].append(row["source_record_id"])
            if row["temporal_novelty_layer"] == "NOVEL_SEQUENCE_RELEASE_SUPPORTED":
                strata["novel_sequence_supported"].append(row["source_record_id"])
            if row["temporal_novelty_layer"] == "UNCHECKED":
                strata["novelty_unchecked"].append(row["source_record_id"])
    write_json(args.output / "novelty_strata.json", strata)
    recommendation = {"schema_version": SCHEMA, "registration_status": "PROPOSED_ONLY_NOT_REGISTERED",
                      "shared_registry_modified": False,
                      "panels": {panel: {
                          "n_records": sum(row["dataset"] == panel for row in novel_rows),
                          "family_status_counts": dict(collections.Counter(row["family_status"] for row in novel_rows if row["dataset"] == panel)),
                          "tr1c_exact_overlap": sum(row["tr1c_exact_overlap"] for row in novel_rows if row["dataset"] == panel),
                          "tr1c_family_overlap": sum(row["tr1c_family_overlap"] for row in novel_rows if row["dataset"] == panel),
                          "cross_panel_exact_overlap": sum(bool(row["cross_panel_exact_overlap"]) for row in novel_rows if row["dataset"] == panel),
                      } for panel in PANELS},
                      "temporal_novelty_counts": dict(collections.Counter(row["temporal_novelty_layer"] for row in novel_rows if row["dataset"] == "rfam_temporal")),
                      "temporal_sequence_first_release_counts": dict(collections.Counter(row["sequence_first_release"] for row in novel_rows if row["dataset"] == "rfam_temporal")),
                      "temporal_strata_file": "novelty_strata.json",
                      "alignment_14_10_failures": [family for family in needed if hits14_10.get(family) is None],
                      "external_test_policy": "AUDIT_ONLY_NO_MODEL_SELECTION"}
    write_json(args.output / "registry_recommendation.json", recommendation)
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(args.output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(args.output / "artifact_manifest.json", {"schema_version": SCHEMA, "files": files,
                                                         "shared_registry_modified": False})
    print(json.dumps({"status": "PASS_TEMPORAL_NOVELTY_WITH_UNCHECKED", **recommendation}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    map_parser = sub.add_parser("map")
    map_parser.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    map_parser.add_argument("--rfam14-root", type=Path, required=True)
    map_parser.add_argument("--rfam15-root", type=Path, required=True)
    map_parser.add_argument("--releases", type=Path, required=True)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.set_defaults(function=command_map)
    novelty = sub.add_parser("novelty")
    novelty.add_argument("--lineage", type=Path, required=True)
    novelty.add_argument("--jsonl-root", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl"))
    novelty.add_argument("--tr1c-family-mapping", type=Path, required=True)
    novelty.add_argument("--previous-overlap", type=Path, required=True)
    novelty.add_argument("--output", type=Path, required=True)
    novelty.set_defaults(function=command_novelty)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
