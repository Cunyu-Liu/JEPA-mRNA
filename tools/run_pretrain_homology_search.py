import argparse
import json
from pathlib import Path
import subprocess

from audit_pretrain_exact_overlap import sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
OUTPUT = ROOT / "homology_round56"
MMSEQS = Path("/home/rna_pretrain_audit_env/bin/mmseqs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["rnacentral_22_active", "rfam_14_9", "ensembl_109_and_bacteria_56"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not MMSEQS.is_file():
        raise RuntimeError("pinned MMseqs2 environment is not installed")
    version = subprocess.run([str(MMSEQS), "version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != "17.b804f":
        raise RuntimeError(f"unexpected MMseqs2 version: {version}")
    query_manifest_path = OUTPUT / "query_manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    if query_manifest["status"] != "PASS_HOMOLOGY_QUERY_PREPARED_EXECUTION_PENDING":
        raise RuntimeError("query identities have not been frozen")
    if sha256(OUTPUT / "queries.fasta") != query_manifest["query_fasta_sha256"]:
        raise AssertionError("query FASTA changed")
    if args.source == "ensembl_109_and_bacteria_56":
        target_manifest_path = OUTPUT / "ensembl_target_manifest.json"
        target_manifest = json.loads(target_manifest_path.read_text())
        if target_manifest["status"] != "PASS_ENSEMBL_HOMOLOGY_TARGET_PREPARED":
            raise RuntimeError("Ensembl target is not ready")
        target = Path(target_manifest["target_fasta"])
        target_sha256 = target_manifest["target_fasta_sha256"]
    else:
        source_manifest_path = ROOT / "source_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        source = next(item for item in source_manifest["sources"] if item["name"] == args.source)
        target = Path(source["local_path"])
        target_sha256 = source["sha256"]
        target_manifest_path = source_manifest_path
    if sha256(target) != target_sha256:
        raise AssertionError("target FASTA changed")
    destination = OUTPUT / args.source
    destination.mkdir(parents=True, exist_ok=False)
    result = destination / "matches.tsv"
    temporary = destination / "tmp"
    command = [str(MMSEQS), "easy-search", str(OUTPUT / "queries.fasta"), str(target), str(result), str(temporary),
               "--search-type", "3", "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "0",
               "--max-seqs", "300", "--threads", str(args.threads), "--format-output",
               "query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits"]
    invocation = {"source": args.source, "command": command, "query_manifest": str(query_manifest_path),
                  "query_manifest_sha256": sha256(query_manifest_path), "target_manifest": str(target_manifest_path),
                  "target_manifest_sha256": sha256(target_manifest_path), "target_sha256": target_sha256,
                  "mmseqs_version": version, "scope": "all queries searched; up to 300 reported targets per query; no-hit is not proof of no homology",
                  "status": "RUNNING"}
    invocation_path = destination / "search_manifest.json"
    invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
    with (destination / "search.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not result.is_file():
        invocation["status"] = "FAILED"
        invocation["returncode"] = completed.returncode
        invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
        raise RuntimeError(f"MMseqs2 failed: {completed.returncode}")
    invocation["status"] = "PASS_SEARCH_COMPLETED_AGGREGATION_PENDING"
    invocation["result_sha256"] = sha256(result)
    invocation["result_bytes"] = result.stat().st_size
    invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
    print(json.dumps(invocation, indent=2))


if __name__ == "__main__":
    main()
