import gzip
import hashlib
import io
import json
from pathlib import Path

from audit_pretrain_exact_overlap import ART, NCRNA, SS, query_sets, sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
OUTPUT = ROOT / "homology_round56"


def query_paths():
    names = ["bprna_tr1c", "bprna_vl0", "bprna_new", "bprna_ts0", "archiveii_embok_clean",
             "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam"]
    paths = {name: SS / f"{name}.jsonl" for name in names}
    for filename in ("train.csv", "val.csv", "test.csv"):
        paths[f"ncrna_{filename}"] = NCRNA / filename
    for split in ("train", "holdout"):
        paths[f"interface_{split}"] = ART / f"artifacts/v2/interface/{split}_contacts.jsonl"
    return paths


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queries = query_sets()
    paths = query_paths()
    if set(paths) != set(queries):
        raise AssertionError("query paths and loaded sets differ")
    membership = {}
    for name, rows in queries.items():
        for _, sequence in rows:
            membership.setdefault(sequence, set()).add(name)
    included = sorted(sequence for sequence in membership if 16 <= len(sequence) <= 8192)
    if len(included) != len(set(included)):
        raise AssertionError("duplicate query IDs")
    fasta_path = OUTPUT / "queries.fasta"
    index_path = OUTPUT / "query_index.jsonl.gz"
    with fasta_path.open("w") as fasta, index_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as index:
                for number, sequence in enumerate(included):
                    query_id = f"Q{number:09d}"
                    fasta.write(f">{query_id}\n")
                    for offset in range(0, len(sequence), 80):
                        fasta.write(sequence[offset:offset + 80] + "\n")
                    index.write(json.dumps({"query_id": query_id, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                                            "length": len(sequence), "source_sets": sorted(membership[sequence])}, sort_keys=True) + "\n")
    per_set = {}
    for name, rows in queries.items():
        unique = set(membership).intersection(sequence for _, sequence in rows)
        kept = [sequence for sequence in unique if 16 <= len(sequence) <= 8192]
        per_set[name] = {"input_rows": len(rows), "unique_sequences": len(unique), "homology_queries": len(kept),
                         "excluded_by_original_length_filter": len(unique) - len(kept)}
    manifest = {"status": "PASS_HOMOLOGY_QUERY_PREPARED_EXECUTION_PENDING",
                "scientific_scope": "frozen query identities for all-query search; no homology result yet",
                "normalization": "uppercase; T to U; preserve ambiguous nucleotide symbols",
                "source_length_filter": "same as RiNALMo paper: 16..8192 nt",
                "planned_search": "MMseqs2 easy-search; min-seq-id 0.8; cov-mode 0; c 0.8; all query/source matches, not representatives",
                "query_fasta": str(fasta_path), "query_fasta_sha256": sha256(fasta_path),
                "query_index": str(index_path), "query_index_sha256": sha256(index_path),
                "n_unique_queries": len(included), "queries": per_set,
                "input_file_sha256": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
                "limitations": ["excluded queries are reported, not counted as non-homologous",
                                "80% identity and bidirectional coverage do not establish independent RNA families",
                                "long pretraining inputs are randomly cropped above 1022 nt"]}
    output = OUTPUT / "query_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "n_unique_queries": len(included),
                      "query_manifest_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
