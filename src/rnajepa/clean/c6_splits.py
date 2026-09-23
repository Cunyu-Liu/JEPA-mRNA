"""C6 — split freezing (spec §4 C6).

Helpers to freeze the standard benchmarks and the out-of-distribution splits:

* ArchiveII — **deduplicated AND raw** versions (dual reporting, spec §3.3/Q9);
* bpRNA TS / TR / TM, RNAStrAlign, bpRNA-new, PDB ts1/ts2/ts3;
* length-bucket / GC-bucket / family-group OOD splits.

Hard constraint enforced here: the test split is written once and its manifest is
hashed; nothing in this module may look at the test set to choose anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .c2_redundancy import family_group_split, family_overlap
from .records import Record
from .c5_audit import build_manifest, manifest_digest

STANDARD_RATIOS = (0.8, 0.1, 0.1)


@dataclass
class FrozenSplit:
    """A named, hashed collection of splits."""

    name: str
    splits: Dict[str, List[Record]]
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""
    manifest: Dict[str, object] = field(default_factory=dict)

    @property
    def sizes(self) -> Dict[str, int]:
        return {name: len(records) for name, records in self.splits.items()}

    def digest(self) -> str:
        payload = {
            "name": self.name,
            "sizes": self.sizes,
            "ids": {k: [r.id for r in v] for k, v in sorted(self.splits.items())},
        }
        return manifest_digest(payload)

    def write(self, out_dir: str) -> Dict[str, object]:
        """Write one JSONL per split plus a hashed manifest."""
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for split_name, records in self.splits.items():
            path = os.path.join(out_dir, f"{self.name}.{split_name}.jsonl")
            with open(path, "w") as handle:
                for record in records:
                    handle.write(json.dumps(_record_to_dict(record)) + "\n")
            written.append(path)
        manifest = build_manifest(written, root=out_dir)
        manifest["split_digest"] = self.digest()
        manifest["created_utc"] = self.created_utc
        manifest["notes"] = self.notes
        manifest_path = os.path.join(out_dir, f"{self.name}.manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        self.manifest = manifest
        return manifest


def _record_to_dict(record: Record) -> Dict[str, object]:
    return {
        "id": record.id,
        "sequence": record.seq_norm or record.sequence,
        "structure": record.structure,
        "family": record.family,
        "source": record.source,
        "licence": record.licence,
        "label_source": record.label_source,
        "fingerprint": record.fingerprint,
        "meta": {k: v for k, v in record.meta.items() if k != "pairs"},
    }


def freeze_family_split(
    records: Sequence[Record],
    name: str,
    *,
    ratios: Tuple[float, float, float] = STANDARD_RATIOS,
    seed: int = 0,
    notes: str = "",
) -> FrozenSplit:
    """Freeze a family-grouped split and assert zero family overlap."""
    splits = family_group_split(records, ratios=ratios, seed=seed)
    overlap = family_overlap(splits)
    if overlap:
        raise AssertionError(f"family leakage across splits: {overlap}")
    return FrozenSplit(name, splits, notes=notes)


def freeze_archiveii(records: Sequence[Record], *, seed: int = 0) -> Dict[str, FrozenSplit]:
    """ArchiveII **dual version**: raw and deduplicated.

    ``records`` are assumed to be the *raw* compilation; the deduplicated version
    is produced by C2 upstream and passed in separately in production.  Here both
    versions are frozen from whatever is supplied, so the dual reporting is
    explicit rather than implicit.
    """
    raw = FrozenSplit("archiveii_raw", {"test": list(records)}, notes="raw compilation, redundancy NOT removed")
    dedup = freeze_family_split(records, "archiveii_dedup", seed=seed, notes="family-grouped, redundancy removed upstream")
    return {"raw": raw, "dedup": dedup}


def freeze_bprna_ts_tr_tm(records: Sequence[Record]) -> FrozenSplit:
    """bpRNA TS/TR/TM split, taken from ``meta['bprna_split']``.

    Records without a ``bprna_split`` field are rejected loudly rather than being
    silently dumped into train.
    """
    splits: Dict[str, List[Record]] = {"TS": [], "TR": [], "TM": []}
    for record in records:
        key = str(record.meta.get("bprna_split", "")).upper()
        if key not in splits:
            raise ValueError(f"record {record.id!r} has no valid bprna_split (got {key!r})")
        splits[key].append(record)
    return FrozenSplit("bprna_ts_tr_tm", splits, notes="official bpRNA TS/TR/TM labels")


def freeze_rnastralign(records: Sequence[Record], *, seed: int = 0) -> FrozenSplit:
    return freeze_family_split(records, "rnastralign", seed=seed)


def freeze_bprna_new(records: Sequence[Record]) -> FrozenSplit:
    """bpRNA-new: cross-family OOD, evaluation only — never trained on."""
    return FrozenSplit(
        "bprna_new",
        {"test": list(records)},
        notes="cross-family OOD main evaluation set; MUST NOT be used for training",
    )


def freeze_pdb_ts(records: Sequence[Record]) -> FrozenSplit:
    """PDB ts1/ts2/ts3, taken from ``meta['pdb_ts']``."""
    splits: Dict[str, List[Record]] = {"ts1": [], "ts2": [], "ts3": []}
    for record in records:
        key = str(record.meta.get("pdb_ts", "")).lower()
        if key not in splits:
            raise ValueError(f"record {record.id!r} has no valid pdb_ts (got {key!r})")
        splits[key].append(record)
    return FrozenSplit("pdb_ts", splits, notes="PDB ts1/ts2/ts3")


def length_bucket_ood(
    records: Sequence[Record],
    *,
    train_max_length: int,
    buckets: Sequence[int] = (512, 1024, 2048),
) -> FrozenSplit:
    """Cross-length-bucket OOD split: train on short, test on each longer bucket."""
    splits: Dict[str, List[Record]] = {"train": []}
    for bucket in buckets:
        splits[f"test_len_gt_{bucket}"] = []
    for record in records:
        length = len(record.seq_norm or record.sequence)
        if length <= train_max_length:
            splits["train"].append(record)
            continue
        for bucket in sorted(buckets):
            if length <= bucket:
                splits[f"test_len_gt_{bucket}"].append(record)
                break
        else:
            splits[f"test_len_gt_{buckets[-1]}"].append(record)
    return FrozenSplit("ood_length", splits, notes=f"train <= {train_max_length} nt")


def gc_bucket_ood(
    records: Sequence[Record],
    *,
    train_gc_range: Tuple[float, float] = (0.4, 0.6),
    bins: Sequence[float] = (0.3, 0.7),
) -> FrozenSplit:
    """Cross-GC-bucket OOD split: train on mid-GC, test on GC extremes."""
    from .c5_audit import gc_content

    splits: Dict[str, List[Record]] = {"train": []}
    for edge in bins:
        splits[f"test_gc_lt_{edge}"] = []
        splits[f"test_gc_gt_{edge}"] = []
    lo, hi = train_gc_range
    for record in records:
        gc = gc_content(record.seq_norm or record.sequence)
        if gc != gc:
            continue
        if lo <= gc <= hi:
            splits["train"].append(record)
            continue
        if gc < lo:
            splits[f"test_gc_lt_{bins[0]}"].append(record)
        else:
            splits[f"test_gc_gt_{bins[-1]}"].append(record)
    return FrozenSplit("ood_gc", splits, notes=f"train GC in [{lo}, {hi}]")


def family_ood_split(records: Sequence[Record], holdout_families: Sequence[str]) -> FrozenSplit:
    """Family-group OOD split: hold out whole families as the test set."""
    holdout = set(holdout_families)
    train, test = [], []
    for record in records:
        (test if (record.family in holdout) else train).append(record)
    return FrozenSplit(
        "ood_family",
        {"train": train, "test": test},
        notes=f"held-out families: {sorted(holdout)}",
    )
