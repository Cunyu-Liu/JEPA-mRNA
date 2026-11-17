import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
RAW = ROOT / "raw" / "ensembl"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url):
    relative = url.split("/pub/", 1)[1]
    destination = RAW / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["curl", "--fail", "--location", "--silent", "--head", url],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lengths = [int(line.split(":", 1)[1].strip()) for line in head.stdout.splitlines()
               if line.lower().startswith("content-length:")]
    expected_size = lengths[-1] if head.returncode == 0 and lengths else None
    if expected_size is not None and destination.exists() and destination.stat().st_size == expected_size:
        return {"url": url, "path": str(destination), "bytes": expected_size, "sha256": sha256(destination)}
    command = ["curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
               "--output", str(destination), url]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"download failed: {url}: {result.stderr[-2000:]}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise AssertionError(f"missing or empty source: {url}")
    if expected_size is not None and destination.stat().st_size != expected_size:
        raise AssertionError(f"size mismatch: {url}")
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def main():
    sources = json.loads((ROOT / "ensembl_sources.json").read_text())
    if sources["status"] != "PASS_ENSEMBL_EXACT_NCRNA_SOURCE_LIST_WITH_DOWNLOAD_PENDING":
        raise RuntimeError("Ensembl source list has not passed")
    jobs = [(name, url) for name, source in sources["sources"].items() for url in source["files"]]
    manifest = {"status": "DOWNLOADING", "scientific_scope": "hash-bound Ensembl source files for exposure audit only; not model training",
                "license_evidence": sources["license_evidence"], "third_party_constraint_note": sources["third_party_constraint_note"],
                "n_files": len(jobs), "sources": {}}
    for name in sources["sources"]:
        manifest["sources"][name] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download, url): (name, url) for name, url in jobs}
        for index, future in enumerate(futures, 1):
            future.result()
            if index % 100 == 0 or index == len(jobs):
                print(f"downloaded {index}/{len(jobs)}", flush=True)
    for name, url in jobs:
        relative = url.split("/pub/", 1)[1]
        path = RAW / relative
        manifest["sources"][name].append({"url": url, "path": str(path),
                                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest["status"] = "PASS_ENSEMBL_PUBLIC_ASSETS_HASH_VERIFIED"
    output = ROOT / "ensembl_download_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "n_files": len(jobs),
                      "total_bytes": sum(item["bytes"] for values in manifest["sources"].values() for item in values)}, indent=2))


if __name__ == "__main__":
    main()
