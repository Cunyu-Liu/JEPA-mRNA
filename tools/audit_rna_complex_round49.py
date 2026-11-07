#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import gemmi

from audit_structrfm_clean_round48 import normalise, sha256

SPLITS = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam",
          "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fasta(records, path):
    with path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{normalise(sequence)}\n")


def source_metadata(cif_path, entity_ids):
    if not cif_path.is_file():
        return {"source_organisms": [], "partner_entity_types": []}
    block = gemmi.cif.read(str(cif_path)).sole_block()
    fields = [("_entity_src_gen", "pdbx_gene_src_scientific_name"),
              ("_entity_src_nat", "pdbx_organism_scientific"),
              ("_rcsb_entity_source_organism", "scientific_name")]
    organisms = set()
    for category, field in fields:
        for row in block.find(f"{category}.", ["entity_id", field]):
            if str(row[0]) in entity_ids and str(row[1]) not in {".", "?", ""}:
                organisms.add(str(row[1]).strip("'"))
    partners = {str(row[0]) for row in block.find("_entity_poly.", ["type"])}
    partners.discard("polyribonucleotide")
    return {"source_organisms": sorted(organisms), "partner_entity_types": sorted(partners)}


def group_rows(rows, cluster_path):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    entry_indices = {}
    for index, row in enumerate(rows):
        if row["pdb_id"] in entry_indices:
            union(index, entry_indices[row["pdb_id"]])
        else:
            entry_indices[row["pdb_id"]] = index
    cluster_indices = {}
    for line in cluster_path.open():
        representative, member = line.rstrip().split("\t")[:2]
        index = int(member)
        if representative in cluster_indices:
            union(index, cluster_indices[representative])
        else:
            cluster_indices[representative] = index
    roots = {index: find(index) for index in range(len(rows))}
    if len(set(roots.values())) == 0:
        raise RuntimeError("empty sequence grouping")
    names = {root: f"sequence_cluster_{root}" for root in set(roots.values())}
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(names[roots[index]], []).append(row)
    if sum(map(len, groups.values())) != len(rows):
        raise RuntimeError("entry+sequence grouping is not a partition")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries-root", type=Path, required=True)
    parser.add_argument("--pretrain-fasta", type=Path, required=True)
    parser.add_argument("--clean-splits", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--vl0-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, default=Path("/home/rna_jev/tools/bin/mmseqs"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--pretrain-sha256", default="739369d31df4a828adba040d9f88eb5c3bc6dbf010d7d22279333e694a9f3a40")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_pretrain = args.pretrain_sha256
    if sha256(args.pretrain_fasta) != expected_pretrain:
        raise ValueError("pretraining corpus identity mismatch")
    sources, chain_rows, seen = [], [], {}
    for path in sorted(args.entries_root.glob("*/labels.json")):
        document = json.loads(path.read_text())
        for chain in document["chains"]:
            sequence = normalise(chain["sequence"])
            row = dict(chain)
            metadata = source_metadata(path.with_name(f"{chain['pdb_id']}.cif"), set(chain.get("entity_ids", [])))
            row.update({"sequence": sequence, "source_json": str(path),
                        "source_json_sha256": sha256(path),
                        "experimental_method": document.get("experimental_method"),
                        "resolution_angstrom": document.get("resolution_angstrom"),
                        "deposition_date": document.get("deposition_date"), **metadata})
            if sequence in seen:
                seen[sequence]["duplicate_chain_ids"].append(chain["id"])
            else:
                seen[sequence] = row
                chain_rows.append(row)
    references = [(f"pretrain_{index}", line.strip())
                  for index, line in enumerate(args.pretrain_fasta.open()) if not line.startswith(">")]
    for path in [args.train_jsonl, args.vl0_jsonl]:
        rows = [json.loads(line) for line in path.open() if line.strip()]
        sources.append({"name": path.stem, "path": str(path), "sha256": sha256(path), "n": len(rows)})
        references.extend((f"{path.stem}_{i}", row["seq"]) for i, row in enumerate(rows))
    for name in SPLITS:
        path = args.clean_splits / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.open() if line.strip()]
        if not rows:
            raise ValueError(f"empty protected split: {name}")
        sources.append({"name": name, "path": str(path), "sha256": sha256(path), "n": len(rows)})
        references.extend((f"{name}_{i}", row["seq"]) for i, row in enumerate(rows))
    reference_path = args.out_dir / "protected_references.fasta"
    fasta(references, reference_path)
    query_path = args.out_dir / "candidate_chains.fasta"
    fasta([(str(i), row["sequence"]) for i, row in enumerate(chain_rows)], query_path)
    exact_references = {normalise(sequence) for _, sequence in references}
    exact = [i for i, row in enumerate(chain_rows) if row["sequence"] in exact_references]
    alignment = args.out_dir / "alignment.tsv"
    command = [str(args.mmseqs), "easy-search", str(query_path), str(reference_path),
               str(alignment), tempfile.mkdtemp(prefix="complex_mmseqs_"), "--min-seq-id", "0.8",
               "--cov-mode", "2", "-c", "0.8", "--max-seqs", "10000", "--threads",
               str(args.threads), "--search-type", "3"]
    subprocess.run(command, check=True)
    homology = collections.defaultdict(list)
    if alignment.exists():
        with alignment.open() as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if fields and not fields[0].startswith("#"):
                    homology[int(fields[0])].append({"reference": fields[1], "identity": float(fields[2]),
                                                     "alignment_length": int(fields[3])})
    kept, removed = [], []
    for index, row in enumerate(chain_rows):
        if index in exact or index in homology:
            removed.append({"id": row["id"], "sequence_sha256": hashlib.sha256(row["sequence"].encode()).hexdigest(),
                            "exact": index in exact, "homologous_matches": homology.get(index, [])})
        else:
            kept.append(dict(row, candidate_index=len(kept)))
    if len(chain_rows) != len(kept) + len(removed):
        raise RuntimeError("candidate attrition does not balance")
    unsupported_nested = [{"id": row["id"], "raw_pairs": row["raw_pairs"]}
                          for row in kept if not row["canonical_nested_pairs"]]
    n_decontaminated = len(kept)
    kept = [row for row in kept if row["canonical_nested_pairs"]]
    for candidate_index, row in enumerate(kept):
        row["candidate_index"] = candidate_index
    clean_path = args.out_dir / "decontaminated.fasta"
    fasta([(str(i), row["sequence"]) for i, row in enumerate(kept)], clean_path)
    cluster_command = [str(args.mmseqs), "easy-search", str(clean_path), str(clean_path),
                       str(args.out_dir / "self_alignment.tsv"), tempfile.mkdtemp(prefix="complex_self_"),
                       "--min-seq-id", "0.8", "--cov-mode", "2", "-c", "0.8",
                       "--max-seqs", str(max(1, len(kept))), "--threads", str(args.threads),
                       "--search-type", "3", "--mask", "0", "--comp-bias-corr", "0"]
    subprocess.run(cluster_command, check=True)
    with (args.out_dir / "clusters_cluster.tsv").open("w") as output:
        for index in range(len(kept)):
            output.write(f"{index}\t{index}\n")
        self_alignment = args.out_dir / "self_alignment.tsv"
        if self_alignment.exists():
            for line in self_alignment.open():
                query, target = line.rstrip().split("\t")[:2]
                if query != target:
                    output.write(f"{query}\t{target}\n")
    groups = group_rows(kept, args.out_dir / "clusters_cluster.tsv")
    if not kept:
        raise RuntimeError("no decontaminated complex chains; refusing to fabricate a holdout")
    assignments = {}
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest())
    n_holdout = max(1, round(len(kept) * 0.2))
    current = 0
    for group in ordered:
        if current < n_holdout:
            assignments[group] = "holdout"
            current += len(groups[group])
        else:
            assignments[group] = "train"
    if current == 0 or current == len(kept):
        raise RuntimeError("holdout split is empty or consumes all chains")
    manifest = {"seed": args.seed, "protocol": "normalized exact plus MMseqs2 identity>=0.8/query-coverage>=0.8 against TR1c, VL0, 13 clean splits and structRFM corpus; joint PDB entry+80% sequence clusters",
                "pretrain_sha256": expected_pretrain, "reference_sha256": sha256(reference_path),
                "search_command": command, "cluster_search_command": cluster_command, "sources": sources,
                "n_label_chains": sum(len(json.loads(path.read_text())["chains"]) for path in args.entries_root.glob("*/labels.json")),
                "n_unique_sequences": len(chain_rows), "n_removed": len(removed), "n_clean": len(kept),
                "n_decontaminated": n_decontaminated, "n_unsupported_nested": len(unsupported_nested),
                "supervision": "pairs=canonical_nested_pairs; raw_pairs retained for stratified analysis; no-crossing/no-shared-endpoint subset required",
                "unsupported_nested": unsupported_nested,
                "n_holdout": current, "groups": {name: {"assignment": assignments[name],
                  "ids": sorted(row["id"] for row in rows)} for name, rows in groups.items()},
                "removed": removed}
    atomic_json(args.out_dir / "decontamination.json", manifest)
    for assignment in ["train", "holdout"]:
        with (args.out_dir / f"complex_{assignment}.jsonl").open("w") as handle:
            for group in sorted(groups):
                if assignments[group] != assignment:
                    continue
                for row in groups[group]:
                    record = {"name": row["id"], "seq": row["sequence"],
                              "pairs": row["canonical_nested_pairs"], "raw_pairs": row["raw_pairs"],
                              "canonical_nested_pairs": row["canonical_nested_pairs"],
                              "group": group, "candidate_index": row["candidate_index"], "source": row}
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ["n_label_chains", "n_unique_sequences", "n_removed", "n_clean", "n_holdout"]}))


if __name__ == "__main__":
    main()
