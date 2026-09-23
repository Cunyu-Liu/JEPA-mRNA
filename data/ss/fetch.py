"""Acquisition scaffolding for the RNA secondary-structure datasets (spec §3.3).

This module does **not** download anything in this environment (there is no
network here; large artefacts live on the cluster at ``/mnt/cunyuliu/rna-jepa``).
It provides, instead:

* per-source fetch definitions -- URL(s) + mirror candidates, expected size,
  expected hash (left blank and marked ``待核验`` wherever it is not on record --
  **hashes are never invented**), licence, and a resumability strategy;
* a ``--dry-run`` planner that prints exactly what *would* be fetched, with no
  network access;
* a hash manifest writer so every fetched artefact is recorded with its
  expected/actual md5.

Reuse, not reinvention
----------------------
The proven multi-connection ranged-download + md5 verification lives in
``scripts/fetch_zenodo_parallel.sh`` (DNS-poisoned host + 15-20 KB/s per
connection => fixed-size byte ranges, a large worker pool, ``.parts/p%06d``
shards for resumability, then concat + md5).  This module **drives that script**
via subprocess rather than reimplementing it.  The resumability strategy for
every source is therefore exactly ``sharded_parts`` -- the same shard scheme,
so an interrupted fetch resumes from the completed shards.

Usage
-----
    python data/ss/fetch.py --dry-run                 # plan only, no network
    python data/ss/fetch.py --dry-run --manifest m.json
    python data/ss/fetch.py --execute --only bprna_1m # real fetch (needs network)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

#: Where the sharded-download helper lives (reused, not rewritten).
FETCH_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "scripts", "fetch_zenodo_parallel.sh")

#: Default root for fetched structure data (cluster path; absent on this machine).
DEFAULT_OUT_ROOT = os.environ.get("RNAJEPA_SS_ROOT", "/mnt/cunyuliu/rna-jepa/data/ss/raw")

HASH_UNKNOWN = "待核验"
RESUMABILITY = "sharded_parts"
MANIFEST_VERSION = "data/ss/fetch.py@v1"


@dataclass
class FetchSpec:
    """One fetchable artefact.

    ``expected_md5`` is ``""`` and ``hash_status == 待核验`` when the real hash is
    not known.  It must never be filled with a guessed value.
    """

    key: str
    name: str
    target_name: str
    urls: List[str] = field(default_factory=list)
    mirror_candidates: List[str] = field(default_factory=list)
    expected_size_bytes: Optional[int] = None
    expected_md5: str = ""
    hash_status: str = HASH_UNKNOWN
    licence: str = HASH_UNKNOWN
    resumability: str = RESUMABILITY
    shard_mb: int = 8
    jobs: int = 128
    ss_annotations: str = "yes"
    notes: str = ""


def structure_specs() -> List[FetchSpec]:
    """Structure datasets to acquire (spec §3.3).

    URLs and hashes are left blank + ``待核验`` wherever they are not on record.
    The bpRNA site is the only URL the spec itself names (§3.3); even it is
    marked unprobed because reachability has not been measured.
    """
    bprna_url = "https://bprna.cgrb.oregonstate.edu/"
    return [
        FetchSpec(
            key="bprna_1m", name="bpRNA-1m（含 TR0/TS0/VL0）", target_name="bpRNA_1m.tar.gz",
            urls=[bprna_url], mirror_candidates=[HASH_UNKNOWN + "：spec §3.3 未指定镜像"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN,
            notes="spec §3.3 记为 ~102,318 结构；URL/大小/哈希均待核验，站点可用性未实测",
        ),
        FetchSpec(
            key="bprna_new", name="bpRNA-new", target_name="bpRNA_new.tar.gz",
            urls=[bprna_url], mirror_candidates=[HASH_UNKNOWN],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN,
            notes="跨家族 OOD 主评测集；仅评测，不进训练",
        ),
        FetchSpec(
            key="archiveii", name="ArchiveII", target_name="archiveii.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN + "：spec §3.3 记为『多镜像』，未给 URL"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, notes="~3,975；原始版冗余严重，须去冗余",
        ),
        FetchSpec(
            key="rnastralign", name="RNAStrAlign", target_name="rnastralign.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, notes="~30,451；与 ArchiveII 须交叉去冗余",
        ),
        FetchSpec(
            key="pdb_derived", name="PDB 衍生（RNAsolo / RCSB）", target_name="pdb_derived.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN + "：spec §3.3 仅列名 RNAsolo / RCSB"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, notes="需分辨率与方法过滤",
        ),
        FetchSpec(
            key="pseudobase_pp", name="PseudoBase++", target_name="pseudobase_pp.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, notes="假结集；规模小",
        ),
        FetchSpec(
            key="rna_puzzles_casp", name="RNA-Puzzles / CASP15-16", target_name="rna_puzzles.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN + "：官方站点未记录"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, notes="仅评测，绝不进训练",
        ),
        FetchSpec(
            key="probing_data", name="SHAPE / DMS / icSHAPE / PARS", target_name="probing.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN + "：各原始论文补充材料"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, ss_annotations="partial",
            notes="间接/噪声标签，须分级",
        ),
        FetchSpec(
            key="rnacentral", name="RNAcentral", target_name="rnacentral.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN + "：FTP 主机未记录"],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, ss_annotations="no",
            notes="数十 GB 量级；网络受限（spec §3.4），须单独评估吞吐",
        ),
        FetchSpec(
            key="rfam", name="Rfam", target_name="rfam.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, ss_annotations="partial",
            notes="网络受限（spec §3.4）",
        ),
        FetchSpec(
            key="ensembl_gencode", name="Ensembl / GENCODE", target_name="ensembl_gencode.tar.gz",
            urls=[], mirror_candidates=[HASH_UNKNOWN],
            expected_size_bytes=None, expected_md5="", hash_status=HASH_UNKNOWN,
            licence=HASH_UNKNOWN, ss_annotations="no",
            notes="网络受限（spec §3.4）；序列/注释，非结构标注",
        ),
    ]


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
class FetchBlockedError(RuntimeError):
    """Raised when a spec cannot be fetched yet (unknown URL or size)."""


def build_plan(specs: List[FetchSpec], out_root: str) -> List[Dict[str, object]]:
    """Return the plan: what would be fetched, and why some entries are blocked.

    No network access happens here.
    """
    plan: List[Dict[str, object]] = []
    for spec in specs:
        out_path = os.path.join(out_root, spec.target_name)
        entry: Dict[str, object] = {
            "key": spec.key,
            "name": spec.name,
            "out_path": out_path,
            "resumability": spec.resumability,
            "shard_mb": spec.shard_mb,
            "jobs": spec.jobs,
            "licence": spec.licence,
            "hash_status": spec.hash_status,
            "expected_size_bytes": spec.expected_size_bytes,
            "expected_md5": spec.expected_md5,
        }
        if not spec.urls:
            entry["action"] = "blocked"
            entry["blocked_reason"] = "URL 待核验（未记录，禁止编造）"
        elif spec.expected_size_bytes is None:
            entry["action"] = "blocked"
            entry["blocked_reason"] = "expected size 待核验：sharded ranged download 需要总大小"
        else:
            entry["action"] = "fetch"
            entry["url"] = spec.urls[0]
            entry["command"] = [
                "bash", os.path.normpath(FETCH_HELPER),
                spec.urls[0], out_path, str(spec.expected_size_bytes),
                spec.expected_md5 or HASH_UNKNOWN, str(spec.jobs), str(spec.shard_mb),
            ]
        plan.append(entry)
    return plan


def render_plan(plan: List[Dict[str, object]]) -> str:
    lines = ["# 二级结构数据获取计划（dry-run，无网络访问）", ""]
    for e in plan:
        lines.append(f"## {e['key']} — {e['name']}")
        lines.append(f"- action: **{e['action']}**")
        lines.append(f"- out: `{e['out_path']}`")
        lines.append(f"- resumability: `{e['resumability']}`（分片 `.parts/p%06d`，可断点续传）")
        lines.append(f"- size: {e['expected_size_bytes'] if e['expected_size_bytes'] is not None else '待核验'}"
                     f"  md5: {e['expected_md5'] or '待核验'} ({e['hash_status']})")
        lines.append(f"- licence: {e['licence']}")
        if e["action"] == "fetch":
            lines.append(f"- url: {e['url']}")
            lines.append(f"- cmd: `{' '.join(e['command'])}`")  # type: ignore[arg-type]
        else:
            lines.append(f"- BLOCKED: {e['blocked_reason']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def new_manifest() -> Dict[str, object]:
    return {"manifest_version": MANIFEST_VERSION, "entries": []}


def upsert_entry(manifest: Dict[str, object], entry: Dict[str, object]) -> Dict[str, object]:
    """Insert or replace the manifest entry with the same ``key``."""
    entries = manifest.setdefault("entries", [])
    for i, existing in enumerate(entries):  # type: ignore[union-attr]
        if existing.get("key") == entry.get("key"):
            entries[i] = entry  # type: ignore[index]
            return manifest
    entries.append(entry)  # type: ignore[union-attr]
    return manifest


def write_manifest(path: str, manifest: Dict[str, object]) -> None:
    manifest = dict(manifest)
    manifest["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).replace(
        microsecond=0, tzinfo=None).isoformat() + "Z"
    with open(path, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def load_manifest(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return new_manifest()
    with open(path) as fh:
        return json.load(fh)


def verify_artefact(path: str, expected_md5: str) -> Dict[str, object]:
    """Verify a downloaded artefact.  Unknown expected hash -> cannot verify."""
    if not os.path.exists(path):
        return {"status": "missing", "actual_md5": "", "verified": False}
    actual = md5_file(path)
    if not expected_md5 or expected_md5 == HASH_UNKNOWN:
        return {"status": "unverifiable", "actual_md5": actual, "verified": False}
    ok = actual == expected_md5
    return {"status": "ok" if ok else "md5_mismatch", "actual_md5": actual, "verified": ok}


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def fetch_spec(spec: FetchSpec, out_root: str) -> Dict[str, object]:
    """Fetch one spec by delegating to the proven sharded downloader.

    Raises :class:`FetchBlockedError` when the spec is not yet resolvable
    (unknown URL or size) -- it never invents a URL or a hash.
    """
    if not spec.urls:
        raise FetchBlockedError(f"{spec.key}: URL 待核验 -- cannot fetch without a recorded URL")
    if spec.expected_size_bytes is None:
        raise FetchBlockedError(f"{spec.key}: expected size 待核验 -- sharded download needs total size")
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, spec.target_name)
    cmd = ["bash", os.path.normpath(FETCH_HELPER), spec.urls[0], out_path,
           str(spec.expected_size_bytes), spec.expected_md5 or HASH_UNKNOWN,
           str(spec.jobs), str(spec.shard_mb)]
    subprocess.run(cmd, check=True)
    verification = verify_artefact(out_path, spec.expected_md5)
    return {
        "key": spec.key,
        "name": spec.name,
        "url": spec.urls[0],
        "out_path": out_path,
        "expected_size_bytes": spec.expected_size_bytes,
        "expected_md5": spec.expected_md5,
        "hash_status": spec.hash_status,
        "licence": spec.licence,
        "resumability": spec.resumability,
        "fetched_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0, tzinfo=None).isoformat() + "Z",
        "tool": MANIFEST_VERSION,
        **verification,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RNA SS data acquisition scaffolding")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the fetch plan without touching the network (default)")
    ap.add_argument("--execute", action="store_true",
                    help="actually fetch (requires network; blocked specs are skipped)")
    ap.add_argument("--only", default="", help="restrict to one spec key")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--manifest", default="data/ss/hash_manifest.json")
    args = ap.parse_args(argv)

    specs = structure_specs()
    if args.only:
        specs = [s for s in specs if s.key == args.only]
        if not specs:
            print(f"[fetch] no spec with key {args.only!r}", file=sys.stderr)
            return 2

    plan = build_plan(specs, args.out_root)
    if args.dry_run or not args.execute:
        sys.stdout.write(render_plan(plan))
        n_blocked = sum(1 for e in plan if e["action"] == "blocked")
        print(f"[fetch] dry-run: {len(plan)} specs, {n_blocked} blocked (URL/size 待核验); no network",
              file=sys.stderr)
        return 0

    manifest = load_manifest(args.manifest)
    for spec in specs:
        try:
            entry = fetch_spec(spec, args.out_root)
        except FetchBlockedError as exc:
            print(f"[fetch] SKIP {spec.key}: {exc}", file=sys.stderr)
            continue
        upsert_entry(manifest, entry)
        write_manifest(args.manifest, manifest)
        print(f"[fetch] {spec.key}: {entry['status']} md5={entry['actual_md5']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
