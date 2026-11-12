import argparse
import bisect
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
TRAIN = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
CL0 = ROOT / "data/cl0_structure_contrast_round54/clusters.jsonl"
OUT = ROOT / "data/cl0_structure_contrast_round54"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pair_f1(left, right):
    left, right = set(map(tuple, left)), set(map(tuple, right))
    tp = len(left & right)
    precision = tp / len(left) if left else 0.0
    recall = tp / len(right) if right else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--qa-output")
    args = parser.parse_args()
    rows = read_jsonl(TRAIN)
    clusters = [row for row in read_jsonl(CL0) if row["split"] == "train"]
    assert len(rows) == len(clusters) == 42564
    assert all(row["split"] == "train" for row in clusters)
    gc = [row["gc"] for row in clusters]
    density = [row["pair_density"] for row in clusters]
    length_sd = statistics.pstdev([row["length"] for row in clusters]) or 1.0
    gc_sd = statistics.pstdev(gc) or 0.01
    density_sd = statistics.pstdev(density) or 0.01
    ordered = sorted(range(len(rows)), key=lambda i: clusters[i]["length"])
    lengths = [clusters[i]["length"] for i in ordered]
    mapping, weights, candidate_f1 = {}, {}, []
    coverage = Counter()
    for index, row in enumerate(rows):
        info = clusters[index]
        sequence_hash = info["sequence_sha256"]
        center = bisect.bisect_left(lengths, info["length"])
        neighborhood = ordered[max(0, center - 64):center + 64]
        candidates = []
        for other in neighborhood:
            if other == index or not clusters[other]["legal_negative"]:
                continue
            if clusters[other]["cluster_id"] == info["cluster_id"]:
                continue
            similarity = pair_f1(row["pairs"], rows[other]["pairs"])
            if similarity >= 0.5:
                continue
            distance = (abs(clusters[other]["length"] - info["length"]) / length_sd,
                        abs(clusters[other]["gc"] - info["gc"]) / gc_sd,
                        abs(clusters[other]["pair_density"] - info["pair_density"]) / density_sd)
            candidates.append((sum(distance), other, similarity))
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = candidates[:32]
        assert selected, f"no structure-hard negative for {sequence_hash}"
        mapping[sequence_hash] = [clusters[other]["sequence_sha256"] for _, other, _ in selected]
        weights[sequence_hash] = {clusters[other]["sequence_sha256"]: 1.0 - similarity
                                  for _, other, similarity in selected}
        candidate_f1.extend(similarity for _, _, similarity in selected)
        coverage[len(selected)] += 1
    manifest = {"status": "PASS_CL3_STRUCTURE_HARD_NEGATIVE_DATA", "n_anchors": len(mapping),
                "negatives_per_anchor": 32, "coverage": dict(sorted(coverage.items())),
                "structure_f1": {"min": min(candidate_f1), "median": statistics.median(candidate_f1),
                                 "max": max(candidate_f1)},
                "composition": "nearest length/GC/pair-density candidates; 64-neighbor length window",
                "structure_threshold": "train reference pair F1<0.5; same 80% identity/PDB-connected cluster excluded",
                "weights": "1-F1 for CL4 only; CL3 uses unweighted negatives",
                "train_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
                "cl0_sha256": hashlib.sha256(CL0.read_bytes()).hexdigest(),
                "test_label_use": False}
    artifacts = {"cl3_negative_map.json": json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n",
                 "cl4_negative_weights.json": json.dumps(weights, sort_keys=True, separators=(",", ":")) + "\n",
                 "cl3_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n"}
    if args.verify_only:
        for name, text in artifacts.items():
            assert (OUT / name).read_text() == text, name
    else:
        for name, text in artifacts.items():
            path = OUT / name
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(path)
    if args.qa_output:
        path = Path(args.qa_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {"status": "PASS_CL3_NEGATIVE_INDEPENDENT_RECOMPUTE", "manifest": manifest,
                  "sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in artifacts}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
