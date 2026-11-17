import hashlib
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
RAW = ROOT / "raw"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_part(source, index, start, end):
    path = RAW / f"{source['name']}.part{index:02d}"
    expected = end - start + 1
    current = path.stat().st_size if path.exists() else 0
    if current > expected:
        raise RuntimeError(f"oversized range part: {path}")
    if current < expected:
        url = source["url"]
        command = ["curl", "--fail", "--location", "--retry", "3",
                   "--range", f"{start + current}-{end}", "--output", str(path), url]
        with (RAW / f"{source['name']}.part{index:02d}.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
    if path.stat().st_size != expected:
        raise RuntimeError(f"range size mismatch: {path}")
    return path


def main():
    manifest_path = ROOT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    jobs = []
    for source in manifest["sources"]:
        if source.get("status") == "COMPLETE_HASH_VERIFIED":
            continue
        original = RAW / source["file"]
        if original.exists():
            partial = RAW / f"{source['name']}.serial_partial.gz"
            if not partial.exists():
                original.rename(partial)
        count = 16 if source["name"].startswith("rnacentral") else 4
        size = source["bytes"]
        chunk = (size + count - 1) // count
        for index in range(count):
            start = index * chunk
            end = min(size - 1, (index + 1) * chunk - 1)
            jobs.append((source, index, start, end))
        source["status"] = "DOWNLOADING_RANGED"
        source["range_parts"] = count
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_part, *job) for job in jobs]
        for future in futures:
            future.result()
    for source in manifest["sources"]:
        if source.get("status") == "COMPLETE_HASH_VERIFIED":
            continue
        destination = RAW / source["file"]
        temporary = RAW / f"{source['name']}.assembled"
        with temporary.open("wb") as output:
            for index in range(source["range_parts"]):
                part = RAW / f"{source['name']}.part{index:02d}"
                with part.open("rb") as handle:
                    for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                        output.write(block)
        if temporary.stat().st_size != source["bytes"]:
            raise AssertionError(f"assembled size mismatch: {source['name']}")
        os.replace(temporary, destination)
        source["sha256"] = sha256(destination)
        source["local_path"] = str(destination)
        source["status"] = "COMPLETE_HASH_VERIFIED"
    manifest["status"] = "PASS_INITIAL_PUBLIC_ASSETS_WITH_NT_AND_ENSEMBL_PENDING"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
