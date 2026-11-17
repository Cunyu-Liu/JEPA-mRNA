import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

TR1C = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
CLUSTERS = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/cl0_structure_contrast_round54/clusters.jsonl")
NCRNA = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/data/rinalmo/ncrna_official")
EXPECTED_NCRNA_OVERLAP = {"train.csv": 2000, "val.csv": 270, "test.csv": 304}
EVAL_FILES = ("bprna_vl0.jsonl", "bprna_new.jsonl", "bprna_ts0.jsonl", "archiveii_embok_clean.jsonl",
              "ref_pdb_ts1.jsonl", "ref_pdb_ts2.jsonl", "ref_pdb_ts3.jsonl",
              "ref_pdb_ts_hard.jsonl", "testsetb.jsonl", "rfam_fam.jsonl")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(sequence):
    return sequence.upper().replace("T", "U")


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    tr1c = list(read_jsonl(TR1C))
    sequences = [normalize(record["seq"]) for record in tr1c]
    sequence_set = set(sequences)
    sources = Counter(record["source"].split(":", 1)[0] for record in tr1c)
    clusters = list(read_jsonl(CLUSTERS))
    train_clusters = [record for record in clusters if record["split"] == "train"]
    cluster_ids = {record["cluster_id"] for record in train_clusters}
    clusters_by_split = {split: {record["cluster_id"] for record in clusters if record["split"] == split}
                         for split in {record["split"] for record in clusters}}
    pdb_entries = {record["pdb_entry"] for record in train_clusters if record["pdb_entry"]}
    if len(tr1c) != 42564 or len(cluster_ids) != 15521:
        raise AssertionError("TR1c or independent-cluster identity changed")
    attrition = {key: sum(int(record.get(key, 0)) for record in tr1c)
                 for key in ("n_pairs", "n_crossing", "n_dropped_crossing", "n_dropped_illegal")}
    duplicates = len(sequences) - len(sequence_set)
    evaluation = {}
    for name in EVAL_FILES:
        path = TR1C.parent / name
        if not path.exists():
            evaluation[name] = {"status": "BLOCKED_MISSING"}
            continue
        eval_sequences = {normalize(record["seq"]) for record in read_jsonl(path)}
        split = {"bprna_vl0": "vl0"}.get(path.stem, path.stem)
        eval_cluster_ids = clusters_by_split.get(split, set())
        evaluation[name] = {"sha256": sha256(path), "n_unique": len(eval_sequences),
                            "exact_overlap_with_tr1c": len(eval_sequences & sequence_set),
                            "n_80pct_clusters": len(eval_cluster_ids),
                            "cluster_overlap_with_train": len(eval_cluster_ids & cluster_ids)}
    clean_root = root / "ncrna_clean_v2"
    clean_root.mkdir(exist_ok=True)
    clean_manifest = {"source": str(NCRNA), "splits": {}, "rule": "remove every exact TR1c overlap from every split; originals preserved"}
    for filename, expected in EXPECTED_NCRNA_OVERLAP.items():
        source = NCRNA / filename
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        kept = [row for row in rows if normalize(row["sequence"]) not in sequence_set]
        removed = len(rows) - len(kept)
        removed_unique = len({normalize(row["sequence"]) for row in rows} - {normalize(row["sequence"]) for row in kept})
        if removed_unique != expected:
            raise AssertionError(f"unique overlap changed for {filename}: {removed_unique} != {expected}")
        destination = clean_root / filename
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("name", "width", "sequence", "class_id"))
            writer.writeheader()
            writer.writerows(kept)
        clean_sequences = {normalize(row["sequence"]) for row in kept}
        if clean_sequences & sequence_set:
            raise AssertionError(f"clean overlap remains for {filename}")
        clean_manifest["splits"][filename] = {
            "input_n": len(rows), "removed_exact_tr1c_rows": removed,
            "removed_exact_tr1c_unique": removed_unique, "output_n": len(kept),
            "n_classes": len({row["class_id"] for row in kept}), "sha256": sha256(destination),
        }
    report = {
        "status": "PASS_AUDIT_WITH_EXPLICIT_LIMITS",
        "scientific_scope": "data/provenance audit only; not a model accuracy or clean-OOD claim",
        "tr1c": {
            "path": str(TR1C), "sha256": sha256(TR1C), "n_sequences": len(tr1c),
            "n_unique_sequences": len(sequence_set), "duplicate_rows": duplicates,
            "n_80pct_clusters": len(cluster_ids), "clusters_sha256": sha256(CLUSTERS),
            "source_prefix_counts": dict(sources), "attrition_totals": attrition,
            "pdb_entry_count_with_identity": len(pdb_entries),
            "label_quality": "mixed bpRNA sources; source prefix is not an experimental-quality grade",
            "pdb_resolution_nmr_missing_residue": "BLOCKED_NOT_PRESENT_IN_TR1C_OR_CLUSTER_FIELDS",
        },
        "evaluation_exact_overlap": evaluation,
        "ncrna_clean": clean_manifest,
        "rinalmo_pretrain_exposure": {
            "status": "BLOCKED_FULL_PRETRAIN_CORPUS_NOT_FROZEN_LOCALLY",
            "claim": "no zero-contamination or clean-OOD assertion is permitted without a separate pretrain-overlap audit",
        },
        "limits": [
            "80% cluster count is not an independent-family count",
            "exact overlap is not approximate-homology or pretraining decontamination",
            "ncRNA cleaning removes exact TR1c overlap only; RiNALMo pretraining exposure remains open",
            "pseudoknot/noncanonical labels and 3D/complex environments are outside the canonical-nested training labels",
        ],
    }
    output = root / "data_exposure_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (root / "ncrna_clean_v2_manifest.json").write_text(json.dumps(clean_manifest, indent=2) + "\n")
    if args.verify:
        independent = json.loads(output.read_text())
        assert independent["tr1c"]["n_80pct_clusters"] == 15521
        assert independent["ncrna_clean"]["splits"] == clean_manifest["splits"]
        for filename in EXPECTED_NCRNA_OVERLAP:
            assert sha256(clean_root / filename) == clean_manifest["splits"][filename]["sha256"]
        print(json.dumps({"verification": "PASS_INDEPENDENT_REREAD", "audit_sha256": sha256(output)}, indent=2))
    else:
        print(json.dumps({"status": report["status"], "audit": str(output), "audit_sha256": sha256(output),
                          "ncrna_clean": report["ncrna_clean"]["splits"]}, indent=2))


if __name__ == "__main__":
    main()
