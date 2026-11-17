import collections
import gzip
import io
import json
import re
from pathlib import Path

from audit_pretrain_exact_overlap import normalize, sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
METADATA = ROOT / "raw/rfam14_9_metadata"
SS = Path("/root/rna_jepa_data/ss_data/jsonl")
EXPECTED = {
    "README": "f3eb26bbfa0ea6aaa50c7867fbcfd27552d8b228e938fe0406111178654b841a",
    "full_region.sql": "6bf3364a88f6d4fbfbde266ea0cb4f0f83d27b92b78afa5c8e8080ed52106324",
    "full_region.txt.gz": "edd83a97cb6b4212d27c538bff22eedc5c293b9100d22eba8dd95c17a4b1907b",
    "family.sql": "e6d1c34847c9560e63b96ad0073c24c631a6bfbfde50078f58f13b32a65cedb4",
    "family.txt.gz": "c35ba4ab02e83c362730bc379c2c7651b7f8c2c1feb5f6ef4121fe94b76c668a",
}
FASTA = ROOT / "raw/Rfam.fa.gz"
FASTA_SHA256 = "5909e198df80a0febf6a62cbe128654a2b6ffffafda9b77f258fd4557009cc5c"


def region_key(name):
    match = re.fullmatch(r"([A-Za-z0-9.]+)_([0-9]+)-([0-9]+)\.bpseq", name)
    return None if match is None else f"{match.group(1)}/{match.group(2)}-{match.group(3)}"


def fasta_records(path):
    key = None
    sequence = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if key is not None:
                    yield key, normalize("".join(sequence))
                key = line[1:].split()[0]
                sequence = []
            elif line:
                sequence.append(line)
    if key is not None:
        yield key, normalize("".join(sequence))


def main():
    files = {name: sha256(METADATA / name) for name in EXPECTED}
    if files != EXPECTED or sha256(FASTA) != FASTA_SHA256:
        raise AssertionError("Rfam metadata or FASTA identity changed")
    schema = (METADATA / "full_region.sql").read_text()
    for field in ("rfam_acc", "rfamseq_acc", "seq_start", "seq_end", "type", "is_significant"):
        if f"`{field}`" not in schema:
            raise AssertionError(f"missing metadata field: {field}")
    families = {}
    with gzip.open(METADATA / "family.txt.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[0] in families:
                raise AssertionError(f"duplicate family accession: {fields[0]}")
            families[fields[0]] = {"rfam_id": fields[1], "description": fields[3], "type": fields[17]}
    regions = {}
    with gzip.open(METADATA / "full_region.txt.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 11:
                raise AssertionError("unexpected full_region field count")
            accession, start, end = fields[1:4]
            key = f"{accession}/{start}-{end}"
            regions.setdefault(key, set()).add((fields[0], fields[9], fields[10]))
    if any(accession not in families for values in regions.values() for accession, _, _ in values):
        raise AssertionError("region refers to an unknown family")
    sequences = {}
    for key, sequence in fasta_records(FASTA):
        if key in sequences and sequences[key] != sequence:
            raise AssertionError(f"conflicting FASTA records: {key}")
        sequences[key] = sequence
    summary = {"status": "PASS_RFAM14.9_FAMILY_MAPPING_WITH_VERSION_AND_COVERAGE_LIMITS",
               "scientific_scope": "official accession/coordinate family labels and sequence identity checks; mapped subsets only, not complete independent-family proof",
               "metadata_sha256": files, "fasta_sha256": FASTA_SHA256,
               "n_families": len(families), "n_metadata_regions": len(regions), "n_fasta_records": len(sequences),
               "datasets": {}}
    output_records = ROOT / "rfam14_9_family_mapping.jsonl.gz"
    family_sets = {}
    with output_records.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                for dataset in ("rfam_fam", "bprna_new", "bprna_tr1c"):
                    counts = collections.Counter()
                    mapped_families = set()
                    input_path = SS / f"{dataset}.jsonl"
                    for row_index, line in enumerate(input_path.open()):
                        row = json.loads(line)
                        key = region_key(row["name"])
                        hits = regions.get(key, set()) if key else set()
                        accessions = sorted({hit[0] for hit in hits})
                        sequence = normalize(row.get("seq", row.get("sequence")))
                        fasta_sequence = sequences.get(key)
                        if key is None:
                            status = "NOT_ACCESSION_COORDINATE_BPSEQ_NAME"
                        elif not hits:
                            status = "NOT_IN_RFAM14.9_FULL_REGION"
                        elif len(accessions) != 1:
                            status = "AMBIGUOUS_MULTIPLE_FAMILIES"
                        elif fasta_sequence is None:
                            status = "METADATA_ONLY_NO_FASTA_REGION"
                        elif fasta_sequence != sequence:
                            status = "SEQUENCE_MISMATCH"
                        else:
                            status = "PASS_METADATA_AND_SEQUENCE"
                            mapped_families.add(accessions[0])
                        counts[status] += 1
                        output.write(json.dumps({"dataset": dataset, "row_index": row_index, "name": row["name"],
                                                 "region_key": key, "status": status, "rfam_families": accessions,
                                                 "family_labels": {accession: families[accession] for accession in accessions},
                                                 "metadata_type_and_significance": sorted(hits),
                                                 "sequence_match": fasta_sequence == sequence if fasta_sequence is not None else None}, sort_keys=True) + "\n")
                    summary["datasets"][dataset] = {"path": str(input_path), "sha256": sha256(input_path),
                                                   "input_rows": row_index + 1, "counts": dict(counts),
                                                   "n_unique_mapped_families": len(mapped_families)}
                    family_sets[dataset] = mapped_families
    summary["mapped_family_overlap_with_tr1c"] = {name: sorted(family_sets["bprna_tr1c"] & family_sets[name])
                                                 for name in ("rfam_fam", "bprna_new")}
    summary["mapping_records"] = str(output_records)
    summary["mapping_records_sha256"] = sha256(output_records)
    summary["limitations"] = ["unmapped rows are not assigned a family or treated as independent",
                              "Rfam14.9 mapping does not cover later Rfam versions",
                              "TR1c includes non-bpseq sources without direct Rfam labels; mapped-family overlap is partial",
                              "family identity and sequence homology are different analyses"]
    destination = ROOT / "rfam14_9_family_mapping_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "datasets": summary["datasets"],
                      "mapped_family_overlap_with_tr1c": {name: len(values) for name, values in summary["mapped_family_overlap_with_tr1c"].items()},
                      "sha256": sha256(destination)}, indent=2))


if __name__ == "__main__":
    main()
