import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
SOURCES = [
    {"name": "rnacentral_22_active", "url": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/22.0/sequences/rnacentral_active.fasta.gz",
     "file": "rnacentral_active.fasta.gz", "release": "RNAcentral 22.0", "bytes": 7281938321,
     "last_modified": "2023-02-08T15:36:55Z", "license": "CC0 icon displayed on https://rnacentral.org/"},
    {"name": "rfam_14_9", "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/14.9/fasta_files/Rfam.fa.gz",
     "file": "Rfam.fa.gz", "release": "Rfam 14.9", "bytes": 180119223,
     "last_modified": "2023-04-06T09:14:29Z", "license": "CC0 stated on https://rfam.org/"},
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "DOWNLOADING", "scientific_scope": "public pretraining-source audit only; not model training or full historical corpus reconstruction",
                "paper": {"title": "RiNALMo: RNA language model", "sha256": "445be4505a0430514c307b8961f6387ed1b2af27575d2a48b4d334846f6370e4",
                          "reported_preprocessing": "36M unique sequences, 17M clusters at 70% identity and 0.8 coverage"},
                "sources": SOURCES,
                "nt": {"url": "https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz", "current_bytes": 405949398878,
                       "status": "NOT_DOWNLOADED_HISTORICAL_SNAPSHOT_UNPINNED"},
                "ensembl": {"releases": ["109", "bacteria-56"], "status": "PENDING_EXACT_FILE_LIST_AND_LICENSE_REVIEW"}}
    manifest_path = ROOT / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for source in SOURCES:
        destination = raw / source["file"]
        log = raw / f"{source['name']}.download.log"
        command = ["curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
                   "--output", str(destination), source["url"]]
        with log.open("a") as handle:
            jobs.append((source, subprocess.Popen(command, stdout=handle, stderr=handle)))
    for source, process in jobs:
        if process.wait() != 0:
            raise RuntimeError(f"download failed: {source['name']}")
    for source in SOURCES:
        path = raw / source["file"]
        if path.stat().st_size != source["bytes"]:
            raise AssertionError(f"size mismatch: {source['name']}")
        source["sha256"] = sha256(path)
        source["local_path"] = str(path)
        source["status"] = "COMPLETE_HASH_VERIFIED"
    manifest["status"] = "PASS_INITIAL_PUBLIC_ASSETS_WITH_NT_AND_ENSEMBL_PENDING"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
