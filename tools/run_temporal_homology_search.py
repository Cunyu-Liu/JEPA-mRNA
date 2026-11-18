import argparse
import hashlib
import json
from pathlib import Path

import run_pretrain_homology_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--eligible-fasta", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source", choices=["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    metadata_hash = hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest()
    assert metadata_hash == "b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c"
    output = Path(args.output_root)
    fasta = Path(args.eligible_fasta)
    rows = [line.strip() for line in fasta.open() if line.strip()]
    identifiers = [line[1:] for line in rows if line.startswith(">")]
    sequences = [line for line in rows if not line.startswith(">")]
    assert len(identifiers) == len(sequences) == 1131
    assert len(set(identifiers)) == len(set(sequences)) == 1131
    assert all(16 <= len(sequence) <= 8192 for sequence in sequences)
    manifest_path = output / "query_manifest.json"
    if args.prepare_only:
        if manifest_path.exists() or (output / "queries.fasta").exists():
            raise FileExistsError("refuse to overwrite temporal homology query assets")
        output.mkdir(parents=True, exist_ok=True)
        (output / "queries.fasta").write_bytes(fasta.read_bytes())
        manifest = {"status": "PASS_HOMOLOGY_QUERY_PREPARED_EXECUTION_PENDING",
                    "metadata_sha256": metadata_hash, "query_fasta_sha256": hashlib.sha256(fasta.read_bytes()).hexdigest(),
                    "n_queries": 1131, "source_fasta": str(fasta),
                    "scope": "unique full polymer sequences; no label or homology quality claim"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_queries"] == 1131
    assert manifest["metadata_sha256"] == metadata_hash
    assert manifest["query_fasta_sha256"] == hashlib.sha256(fasta.read_bytes()).hexdigest()
    run_pretrain_homology_search.OUTPUT = output
    run_pretrain_homology_search.main()


if __name__ == "__main__":
    main()
