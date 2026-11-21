#!/usr/bin/env python3
"""Independently recompute MMseqs hit counts from raw TSV and sequence lengths."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def independent_hits(path, query_map, query_lengths, target_lengths):
    output = {}
    with Path(path).open() as handle:
        for line in handle:
            f = line.rstrip().split("\t")
            if len(f) < 12:
                raise ValueError("invalid MMseqs TSV")
            q, t = f[0], f[1]
            identity = float(f[2])
            qcov = (int(f[7]) - int(f[6]) + 1) / int(query_lengths[q])
            tcov = (int(f[9]) - int(f[8]) + 1) / int(target_lengths[t])
            if identity < 0.8 - 1e-12 or qcov < 0.8 - 1e-12 or tcov < 0.8 - 1e-12:
                continue
            dataset = query_map[q]["dataset"]
            row = output.setdefault(dataset, {"keys": set(), "hashes": set(), "alignments": 0})
            row["keys"].add(q)
            row["hashes"].add(query_map[q]["sequence_sha256"])
            row["alignments"] += 1
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.document.read_text())
    work = Path(document["work_dir"])
    checks = {
        "not_clean_claim": document["not_clean_claim"] is True,
        "not_sota": document["not_sota"] is True,
        "not_model_selection": document["not_model_selection"] is True,
        "generator_hash": sha256_file(Path(__file__).resolve().with_name("audit_homology_mmseqs.py")) == document["generator_sha256"],
        "input_hashes": all(sha256_file(path) == expected for path, expected in document["input_files"].items()),
        "result_hashes": all(sha256_file(work / name) == expected for name, expected in document["result_files"].items()),
        "binary_hash": sha256_file(document["mmseqs"]["path"]) == document["mmseqs"]["sha256"],
        "binary_version": subprocess.check_output([document["mmseqs"]["path"], "version"], text=True).strip() == document["mmseqs"]["version"],
    }
    recomputed_tr = independent_hits(work / "vs_tr1c.tsv", document["query_map"], document["query_lengths"],
                                     document["target_tr1c_lengths"])
    recomputed_ch = independent_hits(work / "chanrg_vs_train_validation.tsv", document["query_map"],
                                     document["query_lengths"], document["target_chanrg_lengths"])
    count_checks = {}
    for field, recomputed in [("vs_tr1c", recomputed_tr), ("chanrg_vs_train_validation", recomputed_ch)]:
        for dataset, expected in document[field].items():
            actual = recomputed.get(dataset, {"keys": set(), "hashes": set(), "alignments": 0})
            count_checks[f"{field}:{dataset}"] = (
                expected["n_hit_records"] == len(actual["keys"]) and
                expected["n_hit_unique_sequences"] == len(actual["hashes"]) and
                expected["n_alignments_passing_postfilter"] == actual["alignments"] and
                expected["contract_status"] == ("HOMOLOGY" if actual["keys"] else "NOT_DETECTED_UNKNOWN"))
    checks["independent_hit_counts"] = all(count_checks.values())
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "document": str(args.document), "document_sha256": sha256_file(args.document),
              "verifier_sha256": sha256_file(Path(__file__).resolve()), "checks": checks,
              "count_checks": count_checks,
              "scope": "independent raw-MMseqs recomputation; NOT_DETECTED_UNKNOWN is not clean"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
