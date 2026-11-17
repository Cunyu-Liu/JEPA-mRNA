import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
SS = Path("/root/rna_jepa_data/ss_data/jsonl")
NCRNA = ART / "manifests/v2/data_audit/ncrna_clean_v2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(sequence):
    return sequence.strip().upper().replace("T", "U")


def jsonl_queries(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [(row.get("name", f"row_{i}"), normalize(row.get("seq", row.get("sequence")))) for i, row in enumerate(rows)]


def csv_queries(path):
    with path.open(newline="") as handle:
        return [(row["name"], normalize(row["sequence"])) for row in csv.DictReader(handle)]


def query_sets():
    names = ["bprna_tr1c", "bprna_vl0", "bprna_new", "bprna_ts0", "archiveii_embok_clean",
             "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam"]
    queries = {name: jsonl_queries(SS / f"{name}.jsonl") for name in names}
    for filename in ("train.csv", "val.csv", "test.csv"):
        queries[f"ncrna_{filename}"] = csv_queries(NCRNA / filename)
    interface = ART / "artifacts/v2/interface"
    for split in ("train", "holdout"):
        queries[f"interface_{split}"] = jsonl_queries(interface / f"{split}_contacts.jsonl")
    return queries


def fasta_sequences(path):
    sequence = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    yield normalize("".join(sequence))
                sequence = []
            elif line:
                sequence.append(line)
    if sequence:
        yield normalize("".join(sequence))


def main():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    if manifest["status"] != "PASS_INITIAL_PUBLIC_ASSETS_WITH_NT_AND_ENSEMBL_PENDING":
        raise RuntimeError("public source assets are incomplete")
    clean_manifest = json.loads((ART / "manifests/v2/data_audit/ncrna_clean_v2_manifest.json").read_text())
    for filename in ("train.csv", "val.csv", "test.csv"):
        if sha256(NCRNA / filename) != clean_manifest["splits"][filename]["sha256"]:
            raise AssertionError(f"ncRNA clean identity mismatch: {filename}")
    queries = query_sets()
    unique_queries = {name: {sequence for _, sequence in rows} for name, rows in queries.items()}
    output = {"status": "PASS_EXACT_OVERLAP_WITH_NT_ENSEMBL_PENDING",
              "scientific_scope": "exact sequence overlap against versioned public sources only; not 80% homology, independent-family or complete pretraining decontamination proof",
              "source_status": manifest["status"], "sources": {}}
    for source in manifest["sources"]:
        if source["status"] != "COMPLETE_HASH_VERIFIED":
            raise RuntimeError(f"source is not hash verified: {source['name']}")
        path = Path(source["local_path"])
        if sha256(path) != source["sha256"]:
            raise AssertionError(f"source hash changed: {source['name']}")
        matched = {name: set() for name in queries}
        raw_count = 0
        filtered_count = 0
        for sequence in fasta_sequences(path):
            raw_count += 1
            if not 16 <= len(sequence) <= 8192:
                continue
            filtered_count += 1
            for name, rows in queries.items():
                if sequence in unique_queries[name]:
                    matched[name].add(sequence)
        source_report = {"path": str(path), "sha256": source["sha256"],
                         "raw_fasta_records": raw_count, "records_after_original_length_filter": filtered_count,
                         "queries": {}}
        for name, rows in queries.items():
            unique = unique_queries[name]
            hits = matched[name]
            source_report["queries"][name] = {
                "input_rows": len(rows), "unique_sequences": len(unique),
                "exact_overlap_unique": len(hits),
                "exact_overlap_rows": sum(sequence in hits for _, sequence in rows),
                "overlap_names": sorted(name for name, sequence in rows if sequence in hits),
            }
        source_report["limitations"] = ["streaming exact match only; no cluster or full historical pretrain identity claim"]
        output["sources"][source["name"]] = source_report
        print(source["name"], filtered_count, {name: value["exact_overlap_unique"] for name, value in source_report["queries"].items()}, flush=True)
    destination = ROOT / "exact_overlap.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "sha256": sha256(destination)}, indent=2))


if __name__ == "__main__":
    main()
