import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
SOURCE = ROOT / "data/rna_complex_round49"
OUT = SOURCE / "rinalmo_d1_length_round54"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sequence_set_hash(items):
    return hashlib.sha256("\n".join(sorted({r["seq"].upper().replace("T", "U") for r in items})).encode()).hexdigest()


def stats(items):
    lengths = [len(r["seq"]) for r in items]
    return {"n": len(items), "pairs": sum(len(r["pairs"]) for r in items),
            "max_length": max(lengths), "sequence_set_sha256": sequence_set_hash(items)}


def write_atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--qa-output")
    args = parser.parse_args()
    train = rows(SOURCE / "rinalmo_full_d1/train_d1.jsonl")
    holdout = rows(SOURCE / "rinalmo_full_d1/complex_holdout_0based.jsonl")
    assert len(train) == 43368 and len(holdout) == 202
    raw_train = rows(SOURCE / "decontamination/complex_train.jsonl")
    raw_holdout = rows(SOURCE / "decontamination/complex_holdout.jsonl")
    reports = {}
    for threshold in (1022, 2048, 3000):
        selected = train[:42564] + [r for r in train[42564:] if len(r["seq"]) <= threshold]
        selected_holdout = [r for r in holdout if len(r["seq"]) <= threshold]
        excluded = [{"name": r.get("name", ""), "length": len(r["seq"]),
                     "pairs": len(r["pairs"]),
                     "sequence_sha256": hashlib.sha256(r["seq"].upper().replace("T", "U").encode()).hexdigest(),
                     "reason": f"length>{threshold}"} for r in train[42564:] if len(r["seq"]) > threshold]
        raw_selected = [r for r in raw_train if len(r["seq"]) <= threshold]
        raw_selected_holdout = [r for r in raw_holdout if len(r["seq"]) <= threshold]
        assert len(selected) == 42564 + len(raw_selected)
        assert len(selected_holdout) == len(raw_selected_holdout)
        manifest = {
            "status": "PASS_D1_LENGTH_DATA",
            "threshold_nt": threshold,
            "train": stats(selected),
            "complex_train": stats(raw_selected),
            "holdout": stats(selected_holdout),
            "excluded_complex_train": len(excluded),
            "source_hashes": {name: sha256(SOURCE / path) for name, path in {
                "train_d1": "rinalmo_full_d1/train_d1.jsonl",
                "holdout": "rinalmo_full_d1/complex_holdout_0based.jsonl",
                "raw_train": "decontamination/complex_train.jsonl",
                "raw_holdout": "decontamination/complex_holdout.jsonl"}.items()},
            "pair_index": "existing canonical-nested 0-based rows preserved",
            "holdout_use": "not used for training, early stopping, calibration or arm selection",
        }
        directory = OUT / str(threshold)
        if not args.verify_only:
            directory.mkdir(parents=True, exist_ok=True)
            write_atomic(directory / "train_d1.jsonl",
                         "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in selected))
            write_atomic(directory / "complex_holdout.jsonl",
                         "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in selected_holdout))
            write_atomic(directory / "complex_excluded_by_length.jsonl",
                         "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in excluded))
            write_atomic(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            assert rows(directory / "train_d1.jsonl") == selected
            assert rows(directory / "complex_holdout.jsonl") == selected_holdout
            assert rows(directory / "complex_excluded_by_length.jsonl") == excluded
            assert json.loads((directory / "manifest.json").read_text()) == manifest
        manifest["hashes"] = {name: sha256(directory / name) for name in
                              ("train_d1.jsonl", "complex_holdout.jsonl",
                               "complex_excluded_by_length.jsonl", "manifest.json")}
        reports[str(threshold)] = manifest
    if args.qa_output:
        write_atomic(Path(args.qa_output), json.dumps({"status": "PASS_D1_LENGTH_INDEPENDENT_QA",
                     "thresholds": reports}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: {"train": value["complex_train"]["n"],
                           "holdout": value["holdout"]["n"],
                           "train_pairs": value["complex_train"]["pairs"]}
                      for key, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
