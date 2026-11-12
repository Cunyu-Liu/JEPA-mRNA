import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
TRAIN = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
VL0 = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl")
MMSEQS = Path("/home/rna_jev/tools/bin/mmseqs")
OUT = ROOT / "data/cl0_structure_contrast_round54"
TEST_SPLITS = {name: ROOT / "data/rinalmo_tr0_clean_round52" / f"{name}.jsonl"
               for name in ("bprna_ts0", "bprna_new", "archiveii_embok_clean", "ref_pdb_ts1",
                            "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb")}
TEST_SPLITS["rfam_fam"] = Path("/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise(sequence):
    return sequence.upper().replace("T", "U")


def pdb_entry(row):
    text = f"{row.get('name', '')} {row.get('source', '')}"
    if "pdb" not in text.lower():
        return None
    match = re.search(r"(?i)(?:^|[^0-9a-z])([0-9][a-z0-9]{3})(?:[^0-9a-z]|$)", text)
    return match.group(1).upper() if match else None


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def load_inputs():
    datasets = [("train", TRAIN), ("vl0", VL0)] + list(TEST_SPLITS.items())
    records = []
    for split, path in datasets:
        for index, row in enumerate(read_jsonl(path)):
            records.append({"split": split, "index": index, "name": row.get("name", ""),
                            "seq": normalise(row["seq"]), "pairs": row["pairs"],
                            "source": row.get("source", ""), "pdb_entry": pdb_entry(row)})
    assert len({r["seq"] for r in records if r["split"] == "train"}) == 42564
    return datasets, records


def commands(work):
    cluster = [str(MMSEQS), "easy-cluster", str(work / "all.fa"), str(work / "clusters"),
               "/tmp/cl0_round54_mmseqs", "--min-seq-id", "0.8", "--cov-mode", "0", "-c", "0.8",
               "--cluster-mode", "1", "--threads", "8"]
    nearest = [str(MMSEQS), "easy-search", str(work / "queries.fa"), str(work / "train.fa"),
               str(work / "nearest.tsv"), "/tmp/cl0_round54_search", "--max-seqs", "1",
               "--search-type", "3", "--threads", "8"]
    return cluster, nearest


def build_records(records, work):
    uf = UnionFind(len(records))
    exact = {}
    pdb = {}
    for index, row in enumerate(records):
        if row["seq"] in exact:
            uf.union(index, exact[row["seq"]])
        exact[row["seq"]] = index
        if row["pdb_entry"]:
            if row["pdb_entry"] in pdb:
                uf.union(index, pdb[row["pdb_entry"]])
            pdb[row["pdb_entry"]] = index
    cluster_edges = 0
    with (work / "clusters_cluster.tsv").open() as handle:
        for line in handle:
            if line.strip():
                representative, member = line.split()[:2]
                uf.union(int(representative), int(member))
                cluster_edges += 1
    roots = [uf.find(index) for index in range(len(records))]
    forbidden = {root for root, row in zip(roots, records) if row["split"] != "train"}
    output = []
    for index, (row, root) in enumerate(zip(records, roots)):
        sequence = row["seq"]
        output.append({"global_index": index, "split": row["split"], "index": row["index"],
                       "name": row["name"], "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                       "cluster_id": root, "pdb_entry": row["pdb_entry"],
                       "length": len(sequence), "gc": (sequence.count("G") + sequence.count("C")) / len(sequence),
                       "pair_density": len(row["pairs"]) / len(sequence),
                       "legal_negative": row["split"] == "train" and root not in forbidden})
    return output, cluster_edges, len(forbidden)


def cluster_stats(records):
    groups = collections.defaultdict(list)
    for row in records:
        if row["split"] == "train":
            groups[row["cluster_id"]].append(row)
    return [{"cluster_id": key, "n": len(rows),
             "length_mean": sum(row["length"] for row in rows) / len(rows),
             "gc_mean": sum(row["gc"] for row in rows) / len(rows),
             "pair_density_mean": sum(row["pair_density"] for row in rows) / len(rows),
             "legal_negative": all(row["legal_negative"] for row in rows)}
            for key, rows in sorted(groups.items())]


def nearest_records(records, work):
    train_n = sum(row["split"] == "train" for row in records)
    best = {}
    path = work / "nearest.tsv"
    if not path.exists():
        return []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            query, target = int(fields[0]), int(fields[1])
            identity = float(fields[2])
            alignment_length = int(fields[3])
            query_record, target_record = records[query], records[target]
            row = {"query_global_index": query, "query_split": query_record["split"],
                   "query_index": query_record["index"], "query_name": query_record["name"],
                   "train_global_index": target, "train_index": target_record["index"],
                   "train_name": target_record["name"], "identity": identity,
                   "alignment_length": alignment_length,
                   "query_coverage": alignment_length / query_record["length"],
                   "target_coverage": alignment_length / target_record["length"],
                   "same_cluster": query_record["cluster_id"] == target_record["cluster_id"]}
            key = (identity, row["query_coverage"], row["target_coverage"], -target)
            if query not in best or key > best[query][0]:
                best[query] = (key, row)
    output = [value[1] for value in sorted(best.values(), key=lambda value: value[1]["query_global_index"])]
    assert all(row["query_split"] != "train" and row["train_global_index"] < train_n for row in output)
    return output


def identity_summary(rows):
    return {bucket: sum(low <= row["identity"] < high for row in rows)
            for bucket, low, high in (("lt0.5", 0, 0.5), ("0.5_0.8", 0.5, 0.8),
                                      ("0.8_0.95", 0.8, 0.95), ("ge0.95", 0.95, 1.000001))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reuse-search", action="store_true")
    parser.add_argument("--qa-output")
    args = parser.parse_args()
    work = OUT / "work"
    datasets, raw = load_inputs()
    cluster_command, nearest_command = commands(work)
    if not args.verify_only:
        work.mkdir(parents=True, exist_ok=True)
        with (work / "all.fa").open("w") as all_handle, (work / "train.fa").open("w") as train_handle, (work / "queries.fa").open("w") as query_handle:
            for index, row in enumerate(raw):
                all_handle.write(f">{index}\n{row['seq']}\n")
                if row["split"] == "train":
                    train_handle.write(f">{index}\n{row['seq']}\n")
                else:
                    query_handle.write(f">{index}\n{row['seq']}\n")
        if not args.reuse_search:
            with (work / "cluster.log").open("w") as handle:
                subprocess.run(cluster_command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            with (work / "nearest.log").open("w") as handle:
                subprocess.run(nearest_command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    records, cluster_edges, forbidden_n = build_records(raw, work)
    nearest = nearest_records(records, work)
    stats = cluster_stats(records)
    train = [row for row in records if row["split"] == "train"]
    excluded = [row for row in train if not row["legal_negative"]]
    manifest = {"status": "PASS_CL0_HOMOLOGY_NEGATIVE_POOL_GATE",
                "protocol": "exact + MMseqs 80% identity/80% coverage connected components + explicit PDB entry when available",
                "mmseqs_version": subprocess.check_output([str(MMSEQS), "version"], text=True).strip(),
                "commands": {"cluster": cluster_command, "nearest": nearest_command},
                "sources": {split: {"path": str(path), "sha256": sha256(path)} for split, path in datasets},
                "train_n": len(train), "vl0_n": 196, "test_split_n": len(TEST_SPLITS),
                "train_clusters": len(stats), "cluster_edges": cluster_edges,
                "cross_split_clusters": forbidden_n,
                "legal_negative_train_n": len(train) - len(excluded),
                "excluded_negative_train_n": len(excluded),
                "pdb_annotated_n": sum(row["pdb_entry"] is not None for row in records),
                "pdb_policy": "Only explicit PDB identifiers are grouped; no identifier is inferred from sequence labels.",
                "nearest_by_split": {split: identity_summary([row for row in nearest if row["query_split"] == split])
                                     for split in ("vl0", *TEST_SPLITS)},
                "view_policy": "No structural truncation or suboptimal-structure positive; CL1 token-mask views are generated only after this gate.",
                "test_label_use": "No test labels or structures are used for negative-pool construction."}
    artifacts = {"clusters.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                 "excluded_negatives.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in excluded),
                 "train_cluster_stats.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in stats),
                 "nearest_neighbors.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in nearest),
                 "manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n"}
    if args.verify_only:
        for name, text in artifacts.items():
            assert (OUT / name).read_text() == text, name
    else:
        for name, text in artifacts.items():
            temporary = (OUT / name).with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(OUT / name)
    if args.qa_output:
        report = {"status": "PASS_CL0_INDEPENDENT_RECOMPUTE", "manifest": manifest,
                  "artifact_sha256": {name: sha256(OUT / name) for name in artifacts}}
        path = Path(args.qa_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "train_n", "train_clusters", "legal_negative_train_n",
                       "excluded_negative_train_n", "pdb_annotated_n")}, indent=2))


if __name__ == "__main__":
    main()
