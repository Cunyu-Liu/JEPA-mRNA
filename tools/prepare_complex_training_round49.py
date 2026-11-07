#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination", type=Path, required=True)
    parser.add_argument("--teacher-complex", type=Path, required=True)
    parser.add_argument("--tr1c", type=Path, required=True)
    parser.add_argument("--teacher-tr1c", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verification = json.loads((args.decontamination / "verification.json").read_text())
    assert verification["status"] == "PASS_COMPLEX_DECONTAMINATION_INDEPENDENT_AUDIT"
    assert sha256(args.decontamination / "decontamination.json") == verification["decontamination_sha256"]
    assert sha256(args.decontamination / "complex_train.jsonl") == verification["train_sha256"]
    assert sha256(args.decontamination / "complex_holdout.jsonl") == verification["holdout_sha256"]
    complex_rows = [json.loads(line) for line in (args.decontamination / "complex_train.jsonl").open()]
    assert complex_rows
    for row in complex_rows:
        assert row["pairs"] and row["pairs"] == row["canonical_nested_pairs"]
        assert all(1 <= left < right <= len(row["seq"]) for left, right in row["pairs"])
        assert len({position for pair in row["pairs"] for position in pair}) == 2 * len(row["pairs"])
        assert not any(a < c < b < d for a, b in row["pairs"] for c, d in row["pairs"])
    decision = json.loads(args.decision.read_text())
    assert decision["complete"] and decision["allow_complex_data_route"]
    selected = decision["next_base"]
    metas = []
    for seed in range(3):
        path = Path("runs") / f"{selected['run_prefix']}{seed}" / "run_meta.json"
        if not path.is_absolute():
            path = args.decision.parents[2] / path
        meta = json.loads(path.read_text())
        assert meta["converged"] and meta["early_stopped"] and meta["status"] == "completed"
        assert meta["schedule"]["steps"] == 0 and meta["validation"]["patience"] == 6
        metas.append(meta)
    assert len({json.dumps(meta["objective"], sort_keys=True) for meta in metas}) == 1
    assert len({json.dumps(meta["schedule"], sort_keys=True) for meta in metas}) == 1
    teacher_new = json.loads((args.teacher_complex / "manifest.json").read_text())
    teacher_old = json.loads((args.teacher_tr1c / "manifest.json").read_text())
    assert teacher_new["n_sequences"] == len(complex_rows)
    assert teacher_old["n_sequences"] == sum(1 for _ in args.tr1c.open())
    assert teacher_new["teacher"] == teacher_old["teacher"] == "thermo:viennarna"
    assert teacher_new["version_lock"] == teacher_old["version_lock"] == "ViennaRNA 2.7.2"
    teacher_dir = args.out / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for source, manifest in [(args.teacher_tr1c, teacher_old), (args.teacher_complex, teacher_new)]:
        for shard in manifest["shards"]:
            target = source / shard["file"]
            assert target.is_file()
            item = dict(shard)
            item["index"] = len(shards)
            item["file"] = f"shard_{len(shards):05d}.npz"
            item["source_path"] = str(target)
            item["sha256"] = sha256(target)
            link = teacher_dir / item["file"]
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(os.path.relpath(target, teacher_dir))
            shards.append(item)
    atomic_json(teacher_dir / "manifest.json", {"teacher": "thermo:viennarna",
                "version_lock": "ViennaRNA 2.7.2", "n_sequences": len(complex_rows) + teacher_old["n_sequences"],
                "shards": shards, "sources": [str(args.teacher_tr1c), str(args.teacher_complex)]})
    train_path = args.out / "train_d1.jsonl"
    with train_path.open("w") as output:
        for line in args.tr1c.open():
            if line.strip():
                output.write(line if line.endswith("\n") else line + "\n")
        for row in complex_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    sequences = {json.loads(line)["seq"] for line in train_path.open()}
    assert len(sequences) == sum(1 for _ in train_path.open())
    config = {"status": "PASS_COMPLEX_D1_PRECONFIGURED",
              "d0": {"experiment": selected["experiment"], "run_prefix": selected["run_prefix"],
                     "reuse_existing_three_seed_results": True},
              "d1": {"data": str(train_path), "data_sha256": sha256(train_path),
                     "teacher_dir": str(teacher_dir), "teacher_manifest_sha256": sha256(teacher_dir / "manifest.json"),
                     "n_complex_train": len(complex_rows), "n_tr1c": teacher_old["n_sequences"],
                     "objective": metas[0]["objective"], "schedule": metas[0]["schedule"],
                     "scorer": metas[0]["scorer"], "pair_latent_dim": metas[0]["pair_latent_dim"],
                     "ripe": metas[0]["ripe"], "model_initialization": "same pretrained structRFM as D0; no D0 checkpoint fine-tuning"},
              "complex_holdout": str(args.decontamination / "complex_holdout.jsonl"),
              "complex_holdout_sha256": verification["holdout_sha256"],
              "decision_sha256": sha256(args.decision),
              "source_hashes": {str(args.tr1c): sha256(args.tr1c),
                                str(args.teacher_tr1c / "manifest.json"): sha256(args.teacher_tr1c / "manifest.json"),
                                str(args.teacher_complex / "manifest.json"): sha256(args.teacher_complex / "manifest.json")}}
    atomic_json(args.out / "d1_config.json", config)
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
