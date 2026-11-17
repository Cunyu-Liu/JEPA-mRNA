import hashlib
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
MANIFEST = ROOT / "ensembl_download_manifest.json"
OUTPUT = ROOT / "ensembl_asset_verification.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_stats(path):
    n_records = 0
    n_empty = 0
    n_filtered = 0
    maximum = 0
    sequence = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if n_records:
                    length = len("".join(sequence))
                    n_empty += length == 0
                    n_filtered += 16 <= length <= 8192
                    maximum = max(maximum, length)
                n_records += 1
                sequence = []
            elif line:
                sequence.append(line)
    if n_records:
        length = len("".join(sequence))
        n_empty += length == 0
        n_filtered += 16 <= length <= 8192
        maximum = max(maximum, length)
    if n_records == 0 or n_empty:
        raise AssertionError(f"empty or invalid FASTA: {path}")
    return {"n_records": n_records, "n_after_original_length_filter": n_filtered,
            "n_empty_records": n_empty, "max_sequence_length": maximum}


def verify_file(item):
    path = Path(item["path"])
    if not path.is_file():
        raise AssertionError(f"missing file: {path}")
    size = path.stat().st_size
    if size != item["bytes"] or size == 0:
        raise AssertionError(f"byte count mismatch: {path}")
    if sha256(path) != item["sha256"]:
        raise AssertionError(f"SHA256 mismatch: {path}")
    if not item["url"].endswith(".ncrna.fa.gz"):
        raise AssertionError(f"unexpected source type: {item['url']}")
    return {**item, **fasta_stats(path)}


def main():
    if not MANIFEST.is_file():
        raise RuntimeError(f"download has not completed: {MANIFEST} is absent")
    manifest = json.loads(MANIFEST.read_text())
    source_list = json.loads((ROOT / "ensembl_sources.json").read_text())
    if manifest["status"] != "PASS_ENSEMBL_PUBLIC_ASSETS_HASH_VERIFIED":
        raise RuntimeError("manifest has not passed its download gate")
    if source_list["status"] != "PASS_ENSEMBL_EXACT_NCRNA_SOURCE_LIST_WITH_DOWNLOAD_PENDING":
        raise RuntimeError("source list is incomplete")
    if any(source["errors"] for source in source_list["sources"].values()):
        raise AssertionError("source-list errors remain")
    expected = {(name, url) for name, source in source_list["sources"].items() for url in source["files"]}
    actual = {(name, item["url"]) for name, values in manifest["sources"].items() for item in values}
    if expected != actual:
        raise AssertionError(f"source/file identity mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
    if len(actual) != 31046 or len(actual) != manifest["n_files"]:
        raise AssertionError("unexpected Ensembl file count")
    jobs = [(name, item) for name in source_list["sources"] for item in manifest["sources"][name]]
    verified = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, ((name, item), result) in enumerate(zip(jobs, executor.map(verify_file, [item for _, item in jobs])), 1):
            verified.setdefault(name, []).append(result)
            if index % 1000 == 0 or index == len(jobs):
                print(f"independently verified {index}/{len(jobs)}", flush=True)
    totals = {name: {"n_files": len(values), "bytes": sum(item["bytes"] for item in values),
                     "raw_fasta_records": sum(item["n_records"] for item in values),
                     "records_after_original_length_filter": sum(item["n_after_original_length_filter"] for item in values)}
              for name, values in verified.items()}
    report = {"status": "PASS_ENSEMBL_ASSETS_INDEPENDENTLY_VERIFIED",
              "scientific_scope": "versioned Ensembl ncRNA source integrity only; not complete pretraining exposure, 80% homology or family independence proof",
              "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
              "n_files": sum(item["n_files"] for item in totals.values()),
              "total_bytes": sum(item["bytes"] for item in totals.values()),
              "sources": totals,
              "preprocessing": "RiNALMo paper removes sequences shorter than 16 and longer than 8192; pretraining randomly crops inputs longer than 1022 nt",
              "exposure_boundary": "source membership is not a guarantee that every base of a long sequence appeared in every training crop",
              "verification": ["manifest/source URL bijection", "nonempty files and exact bytes", "independent SHA256",
                               "gzip CRC/decompression", "nonempty FASTA records", "length distribution"],
              "verified_files": verified}
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({**report, "verified_files": "omitted from stdout",
                      "verification_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
