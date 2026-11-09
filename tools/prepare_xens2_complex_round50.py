import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SEEDS = {"plana_a": 42, "plana_b": 2026, "r2d": 3407}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data/rna_complex_round49/xens2_complex"
    manifest = json.loads((data / "data_manifest.json").read_text())
    assert manifest["status"] == "PASS_XENS2_COMPLEX_SUPPORTED_DATA"
    assert manifest["n_total_train"] == 42714
    assert sha256(data / "train_d1.jsonl") == manifest["train_d1_sha256"]
    new_path = data / "teacher_full"
    old_path = Path("/root/rna_jepa_data/ss_data/teacher/bprna_tr1c")
    new = json.loads((new_path / "manifest.json").read_text())
    old = json.loads((old_path / "manifest.json").read_text())
    assert new["n_sequences"] == 150 and old["n_sequences"] == 42564
    assert new["teacher"] == old["teacher"] == "thermo:viennarna"
    assert new["version_lock"] == old["version_lock"] == "ViennaRNA 2.7.2"
    out = data / "teacher_d1"
    out.mkdir(exist_ok=True)
    shards = []
    for source, teacher in [(old_path, old), (new_path, new)]:
        for shard in teacher["shards"]:
            src = source / shard["file"]
            assert src.is_file()
            if "sha256" in shard:
                assert sha256(src) == shard["sha256"]
            item = dict(shard)
            item["sha256"] = sha256(src)
            item["index"] = len(shards)
            item["file"] = f"shard_{len(shards):05d}.npz"
            item["source_path"] = str(src)
            dst = out / item["file"]
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            shards.append(item)
    teacher = {"teacher": "thermo:viennarna", "version_lock": "ViennaRNA 2.7.2",
               "n_sequences": 42714, "shards": shards,
               "sources": [str(old_path), str(new_path)]}
    (out / "manifest.json").write_text(json.dumps(teacher, indent=2) + "\n")
    recipe = {
        "status": "PASS_XENS2_COMPLEX_RECIPE",
        "ensemble": "Plan-A 42/2026 equal average, then 0.7 Plan-A + 0.3 R2D",
        "seeds": SEEDS,
        "data": str(data / "train_d1.jsonl"),
        "data_sha256": sha256(data / "train_d1.jsonl"),
        "teacher_dir": str(out),
        "teacher_manifest_sha256": sha256(out / "manifest.json"),
        "embedding_manifest": str(data / "embeddings/manifest.json"),
        "validation_data": "/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl",
        "validation_teacher": str(args.root / "data/teacher_vl0"),
        "rfam_data": "/root/rna_jepa_data/ss_data/jsonl/rfam_fam.jsonl",
        "rfam_embedding_split": "rfam_fam",
        "w_plana": 0.7,
        "plana_s1_weight": 0.5,
        "plana": {"steps": 0, "batch_size": 2, "patience": 6,
                  "head_lr": 0.0001, "backbone_lr": 1e-05,
                  "warmup_head_steps": 1600, "unfreeze_every": 800,
                  "unfreeze_per_step": 2, "n_blocks": 33, "scorer": "resnet2d"},
        "r2d": {"steps_safety_cap": 1000000, "batch_size": 1, "patience": 6,
                "lr": 0.0001, "scorer": "resnet2d", "resnet_blocks": 4,
                "d_z": 128, "hidden": 64, "head_chunk_size": 0},
        "members": {role: str(args.root / f"runs/xens2_complex_{role}_s{seed}")
                    for role, seed in SEEDS.items()},
        "source_hashes": {str(data / "data_manifest.json"): sha256(data / "data_manifest.json"),
                          str(new_path / "manifest.json"): sha256(new_path / "manifest.json"),
                          str(old_path / "manifest.json"): sha256(old_path / "manifest.json")},
    }
    (data / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps({"status": recipe["status"], "n_sequences": teacher["n_sequences"],
                      "teacher_sha256": recipe["teacher_manifest_sha256"],
                      "recipe_sha256": sha256(data / "recipe.json")}, indent=2))


if __name__ == "__main__":
    main()
