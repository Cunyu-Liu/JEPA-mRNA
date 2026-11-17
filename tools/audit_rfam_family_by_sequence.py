import collections
import gzip
import hashlib
import io
import json
from pathlib import Path

from audit_pretrain_exact_overlap import normalize, sha256
from audit_rfam_family_mapping import EXPECTED, FASTA, FASTA_SHA256, METADATA, ROOT, SS, fasta_records


def resolve_sequence_family(sequence, sequence_regions, regions):
    fasta_keys = sequence_regions.get(sequence, set())
    keys = sorted(fasta_keys.intersection(regions))
    families = sorted({values[0] for key in keys for values in regions[key]})
    if not keys:
        status = "EXACT_IN_FASTA_NO_FAMILY_METADATA" if fasta_keys else "NOT_EXACT_IN_RFAM14.9"
        return status, keys, families
    if len(families) != 1:
        return "AMBIGUOUS_EXACT_SEQUENCE_FAMILIES", keys, families
    return "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY", keys, families


def main():
    hashes = {name: sha256(METADATA / name) for name in EXPECTED}
    if hashes != EXPECTED or sha256(FASTA) != FASTA_SHA256:
        raise AssertionError("Rfam source identity changed")
    families = {}
    with gzip.open(METADATA / "family.txt.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            families[fields[0]] = {"rfam_id": fields[1], "description": fields[3], "type": fields[17]}
    regions = {}
    with gzip.open(METADATA / "full_region.txt.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 11:
                raise AssertionError("unexpected full_region field count")
            key = f"{fields[1]}/{fields[2]}-{fields[3]}"
            regions.setdefault(key, set()).add((fields[0], fields[9], fields[10]))
    if any(accession not in families for values in regions.values() for accession, _, _ in values):
        raise AssertionError("unknown family accession")
    sequence_regions = {}
    n_fasta = 0
    for key, sequence in fasta_records(FASTA):
        sequence_regions.setdefault(sequence, set()).add(key)
        n_fasta += 1
    if n_fasta != 3132784:
        raise AssertionError(f"unexpected Rfam FASTA count: {n_fasta}")
    summary = {"status": "PASS_RFAM14.9_EXACT_SEQUENCE_FAMILY_MAPPING_WITH_COVERAGE_LIMITS",
               "scientific_scope": "official Rfam labels propagated only through exact full-sequence matches; ambiguous and unmatched records are not assigned families",
               "metadata_sha256": hashes, "fasta_sha256": FASTA_SHA256,
               "n_families": len(families), "n_fasta_records": n_fasta,
               "n_unique_fasta_sequences": len(sequence_regions), "datasets": {}}
    output_path = ROOT / "rfam14_9_exact_sequence_family_mapping.jsonl.gz"
    family_sets = {}
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                for dataset in ("rfam_fam", "bprna_new", "bprna_tr1c"):
                    counts = collections.Counter()
                    unique_mapped = set()
                    mapped_families = set()
                    unique_sequences = set()
                    input_path = SS / f"{dataset}.jsonl"
                    for row_index, line in enumerate(input_path.open()):
                        row = json.loads(line)
                        sequence = normalize(row.get("seq", row.get("sequence")))
                        status, keys, accessions = resolve_sequence_family(sequence, sequence_regions, regions)
                        unique_sequences.add(sequence)
                        counts[status] += 1
                        if status == "PASS_UNIQUE_EXACT_SEQUENCE_FAMILY":
                            unique_mapped.add(sequence)
                            mapped_families.add(accessions[0])
                        output.write(json.dumps({"dataset": dataset, "row_index": row_index, "name": row["name"],
                                                 "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                                                 "status": status, "matching_region_keys": keys,
                                                 "rfam_families": accessions,
                                                 "family_labels": {key: families[key] for key in accessions}}, sort_keys=True) + "\n")
                    summary["datasets"][dataset] = {"path": str(input_path), "sha256": sha256(input_path),
                                                   "input_rows": row_index + 1, "unique_sequences": len(unique_sequences),
                                                   "counts": dict(counts), "mapped_unique_sequences": len(unique_mapped),
                                                   "n_unique_mapped_families": len(mapped_families)}
                    family_sets[dataset] = mapped_families
    summary["mapped_family_overlap_with_tr1c"] = {name: sorted(family_sets["bprna_tr1c"] & family_sets[name])
                                                 for name in ("rfam_fam", "bprna_new")}
    summary["mapping_records"] = str(output_path)
    summary["mapping_records_sha256"] = sha256(output_path)
    summary["limitations"] = ["only exact full-sequence matches are used; no approximate alignment for family assignment",
                              "multiple families for one sequence remain ambiguous and are not arbitrarily assigned",
                              "unmatched rows are not independent-family proof", "Rfam14.9 does not cover later releases",
                              "TR1c has mixed non-Rfam sources; mapped-family overlap is a partial analysis",
                              "sequence homology and family identity are separate concepts"]
    destination = ROOT / "rfam14_9_exact_sequence_family_mapping_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "datasets": summary["datasets"],
                      "mapped_family_overlap_with_tr1c": {name: len(values) for name, values in summary["mapped_family_overlap_with_tr1c"].items()},
                      "sha256": sha256(destination)}, indent=2))


if __name__ == "__main__":
    main()
