import argparse
import hashlib
import json
from pathlib import Path

from compare_structrfm_paired_round48 import holm_adjust, paired_bootstrap


SPLITS = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
          "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam",
          "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
MAIN8 = set(SPLITS[:8])
LONG_SPLITS = ("archiveii", "rfam_fam")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_rows(left, right):
    assert left["data_sha256"] == right["data_sha256"]
    assert left["raw_n"] == right["raw_n"]
    def keyed(rows):
        occurrences = {}
        keyed_rows = []
        for row in rows:
            name = row["name"]
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            keyed_rows.append(((name, occurrence), row))
        return keyed_rows

    left_rows = keyed(left["per_sequence"])
    right_rows = keyed(right["per_sequence"])
    right_by_key = dict(right_rows)
    missing = [key for key, _ in left_rows if key not in right_by_key]
    assert not missing, f"right arm lost supported sequences: {missing[:3]}"
    left_keys = {key for key, _ in left_rows}
    extra = [key for key, _ in right_rows if key not in left_keys]
    return [row for _, row in left_rows], [right_by_key[key] for key, _ in left_rows], extra


def subset(rows, scope):
    if scope == "length_200_400":
        return [row for row in rows if 200 < row["length"] <= 400]
    if scope == "length_over_400":
        return [row for row in rows if row["length"] > 400]
    if scope in {"span_30", "span_100"}:
        return [{**row, **row[scope]} for row in rows]
    return rows


def compare_scope(left_rows, right_rows, seed, bootstrap):
    if not left_rows:
        return {"status": "EMPTY_SUPPORTED_SUBSET", "n": 0}
    true_pairs = sum(row["tp"] + row["fn"] for row in left_rows + right_rows)
    if not true_pairs:
        return {"status": "EMPTY_TRUE_PAIR_SUBSET", "n": len(left_rows)}
    return {"status": "PASS", "n": len(left_rows),
            "small_sample_descriptive": len(left_rows) < 30,
            **paired_bootstrap(left_rows, right_rows, seed, bootstrap)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--manifest", default="audits/round48/ripe_preregistered.json")
    parser.add_argument("--status", default="PASS_RIPE_PAIRED_MICRO_F1_BOOTSTRAP")
    args = parser.parse_args()
    assert args.bootstrap >= 1000
    manifest_path = args.root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    baseline = manifest.get("selected") or manifest["baseline"]
    output = {"status": args.status,
              "n_bootstrap": args.bootstrap,
              "manifest_sha256": sha256(manifest_path),
              "baseline": {"experiment": baseline["experiment"], "run_prefix": baseline["run_prefix"]},
              "protocol": "right minus left within seed; same raw data and common supported names only; main8 Holm per seed; Rfam/CHANRG and long subsets unadjusted; n<30 descriptive",
              "unsupported_rule": "right-only supported sequences are counted separately and never imputed as zero left F1",
              "comparisons": {}, "long_subsets": {}, "source_hashes": {}}
    for arm in manifest["arms"]:
        if arm.get("reuse_baseline"):
            continue
        name = arm["arm"]
        comparison_rows = []
        long_rows = []
        for split in SPLITS:
            for seed in range(3):
                left_path = args.root / f"eval_decision/{baseline['experiment']}/s{seed}/{split}.json"
                right_path = args.root / f"eval_decision/{name}/s{seed}/{split}.json"
                for path in (left_path, right_path):
                    output["source_hashes"][str(path.relative_to(args.root))] = sha256(path)
                left = json.loads(left_path.read_text())
                right = json.loads(right_path.read_text())
                left_rows, right_rows, extra = common_rows(left, right)
                digest = hashlib.sha256(f"{name}:{split}:seed{seed}".encode()).digest()
                row = {"split": split, "seed": seed, "raw_n": left["raw_n"],
                       "left_supported_n": left["n"], "right_supported_n": right["n"],
                       "right_only_supported_n": len(extra),
                       "left_checkpoint_sha256": left["checkpoint_sha256"],
                       "right_checkpoint_sha256": right["checkpoint_sha256"],
                       **compare_scope(left_rows, right_rows,
                                       int.from_bytes(digest[:8], "little"), args.bootstrap)}
                comparison_rows.append(row)
                if split not in LONG_SPLITS:
                    continue
                for scope in ("length_200_400", "length_over_400", "span_30", "span_100"):
                    scope_digest = hashlib.sha256(f"{name}:{split}:seed{seed}:{scope}".encode()).digest()
                    long_rows.append({"split": split, "seed": seed, "scope": scope,
                                      "right_only_supported_n_full_split": len(extra),
                                      **compare_scope(subset(left_rows, scope), subset(right_rows, scope),
                                                      int.from_bytes(scope_digest[:8], "little"), args.bootstrap)})
        for seed in range(3):
            holm_adjust([row for row in comparison_rows
                         if row["seed"] == seed and row["split"] in MAIN8 and row["status"] == "PASS"])
        output["comparisons"][name] = comparison_rows
        output["long_subsets"][name] = long_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": output["status"], "out": str(args.out),
                      "sha256": sha256(args.out), "arms": list(output["comparisons"])}, indent=2))


if __name__ == "__main__":
    main()
