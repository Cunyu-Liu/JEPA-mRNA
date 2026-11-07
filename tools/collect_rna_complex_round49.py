#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

DSSR_SHA256 = "c7261c0ab099e1f70c508c51363c7aa6915e216a78bb6d65ccda533092d6c39a"


def download(url, path):
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                if response.status_code == 404:
                    raise FileNotFoundError(url)
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(4 << 20):
                        handle.write(chunk)
            temporary.replace(path)
            return
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def collect(pdb_id, root, python, dssr):
    directory = root / pdb_id
    directory.mkdir(exist_ok=True)
    output = directory / "labels.json"
    if output.is_file():
        if json.loads(output.read_text()).get("dssr_seq_id_mode") == "observed_nts_v1":
            return pdb_id, "cached"
        backup = directory / "labels.auth_seq.json"
        if not backup.exists():
            backup.write_bytes(output.read_bytes())
    try:
        cif = directory / f"{pdb_id}.cif"
        if not cif.is_file():
            download(f"https://files.rcsb.org/download/{pdb_id}.cif", cif)
        subprocess.run([python, "/home/rna_jev/tools/fetch_rna_parent_map_round49.py",
                        "--cif", str(cif), "--output", str(directory / "parents.json")],
                       check=True, capture_output=True, text=True)
        dssr_json = directory / "dssr.json"
        cached_dssr = json.loads(dssr_json.read_text()) if dssr_json.is_file() else None
        if cached_dssr is not None and cached_dssr.get("warning") == "no nucleotides found":
            cached_dssr.update({"nts": [], "pairs": []})
        if cached_dssr is not None and not isinstance(cached_dssr.get("nts"), list):
            backup = directory / "dssr.pair_only.json"
            if not backup.exists():
                backup.write_bytes(dssr_json.read_bytes())
            dssr_json.unlink()
            cached_dssr = None
        if cached_dssr is None:
            temporary_json = directory / "dssr.json.part"
            temporary_json.unlink(missing_ok=True)
            completed = subprocess.run([dssr, f"-i={cif}", "--json", "--more",
                                        f"-o={temporary_json}"], capture_output=True, text=True, timeout=600)
            (directory / "dssr.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode or not temporary_json.is_file():
                raise RuntimeError(f"DSSR failed: {completed.returncode}")
            cached_dssr = json.loads(temporary_json.read_text())
            if cached_dssr.get("warning") == "no nucleotides found":
                cached_dssr.update({"nts": [], "pairs": []})
            if not isinstance(cached_dssr.get("nts"), list):
                raise RuntimeError("DSSR output is missing the full nts list")
            temporary_json.replace(dssr_json)
        subprocess.run([python, "/home/rna_jev/tools/prepare_rna_complex_round49.py",
                        "--cif", str(cif), "--dssr-json", str(dssr_json),
                        "--dssr-binary-sha256", DSSR_SHA256, "--output", str(output)],
                       check=True, capture_output=True, text=True)
        result = json.loads(output.read_text())
        (directory / "error.txt").unlink(missing_ok=True)
        return pdb_id, {"accepted_chains": len(result["chains"]),
                        "rejected_chains": len(result["rejected_chains"]),
                        "labels_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            detail += "\n" + (error.stdout or "") + (error.stderr or "")
        (directory / "error.txt").write_text(detail + "\n")
        return pdb_id, {"error": detail}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-json", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dssr", default="/home/rna_jev/third_party/DSSR-v1.9.10/x3dna-dssr")
    args = parser.parse_args()
    document = json.loads(Path(args.ids_json).read_text())
    if isinstance(document, dict):
        ids = document.get("result_set", document.get("identifier", document.get("ids", [])))
    else:
        ids = document
    ids = [row["identifier"] if isinstance(row, dict) else row for row in ids]
    ids = sorted(set(ids))[:args.limit]
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect, pdb_id, root, args.python, args.dssr) for pdb_id in ids]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            pdb_id, result = future.result()
            results[pdb_id] = result
            print(json.dumps({"completed": index, "total": len(ids), "pdb_id": pdb_id,
                              "result": result}, sort_keys=True), flush=True)
    manifest = {"complete": True, "dssr_seq_id_mode": "observed_nts_v1",
                "n_entries": len(ids), "results": results}
    temporary = root / "collection.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "collection.json")


if __name__ == "__main__":
    main()
