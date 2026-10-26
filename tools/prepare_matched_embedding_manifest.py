#!/usr/bin/env python3
"""Create a hash-locked embedding alias manifest for matched CHANRG experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = {
    "mixed_train": ["bprna_tr1c", "chanrg_train_train", "chanrg_train_validation",
                    "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"],
    "chanrg_train_train": ["chanrg_train_train"],
    "chanrg_train_validation": ["chanrg_train_validation"],
    "chanrg_test": ["chanrg_test"],
    "chanrg_gena": ["chanrg_gena"],
    "chanrg_genc": ["chanrg_genc"],
    "chanrg_genf": ["chanrg_genf"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_dir: Path) -> dict:
    source_manifest_path = source_dir / "manifest.json"
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    by_split = {}
    for entry in source["entries"]:
        by_split.setdefault(entry["split"], []).append(entry)
    hashes = {}
    entries = []
    for alias, source_splits in REQUIRED.items():
        for source_split in source_splits:
            if source_split not in by_split:
                raise ValueError(f"source split missing: {source_split}")
            for original in by_split[source_split]:
                path = Path(original["path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in hashes:
                    hashes[path] = sha256(path)
                item = dict(original)
                item["source_split"] = source_split
                item["split"] = alias
                item["path"] = str(path)
                item["sha256"] = hashes[path]
                entries.append(item)
    return {
        "manifest_type": "alias_with_required_sequence_filter",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"refusing to overwrite a different manifest: {output}")
        print(f"EMBEDDING_ALIAS_UNCHANGED {output}")
        return 0
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    assert len(checked["entries"]) == len(manifest["entries"])
    temporary.replace(output)
    print(f"EMBEDDING_ALIAS_PASS entries={len(manifest['entries'])} "
          f"unique_files={len({entry['path'] for entry in manifest['entries']})} "
          f"sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
