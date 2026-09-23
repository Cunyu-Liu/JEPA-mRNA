"""Executable real-data inventory for the RNA secondary-structure workstream.

This is a **measured audit**, not a plan.  Its job is to make one thing
impossible: leaving any data source in an *assumed-available* state.  Every
source must carry a status drawn from exactly four categories

    已落盘 (LANDED) | 待获取 (TO_FETCH) | 网络受限 (NETWORK_LIMITED) | 不可得 (UNAVAILABLE)

and an evidence tag saying *how* we know that status:

    recorded_fact       -- a measured fact already on record (md5/size from a
                           completed fetch, or a spec-§3 measurement);
    live_probe          -- this process actually probed it (file exists / HTTP
                           reachable / tool importable) and cached the result;
    requires_live_probe -- the status is the best *recorded expectation* and has
                           NOT been probed in this environment.

The honesty rule enforced in code (:func:`assert_no_assumed_available`): a
``requires_live_probe`` entry is never reported as ``verified``.  Offline mode
therefore reports ``verified=False`` for everything it did not itself probe,
and only a saved live-probe cache can promote an entry to ``verified=True``.

Why this file exists (spec §3.2 / §10.2, the project's largest data risk)
------------------------------------------------------------------------
The assets already on disk (Zenodo 17786045, Zenodo 12516160, mRNABERT) contain
**zero secondary-structure annotations**.  17786045 is 145 mRNA *functional*
tasks; 12516160 is unlabelled sequence.  Structure prediction therefore needs an
entirely new acquisition workstream, and the first artefact of that workstream
is this inventory.

Network constraints are real and measured (spec §3.4): from the cluster
``zenodo.org`` DNS does not resolve (returns ``::``) so an IP must be hard-coded,
and a single connection is throttled to ~15-20 KB/s (aggregate ~4 MB/s at 168
workers).  Large files must therefore be fetched as sharded byte ranges with md5
verification -- the capability already proven in
``scripts/fetch_zenodo_parallel.sh`` and reused by :mod:`data.ss.fetch`.

Usage
-----
    # No network here: build from recorded state, mark everything unprobed.
    python data/ss/inventory.py --offline --md data/ss/INVENTORY.md

    # On a networked host: actually probe every source, cache the results.
    python data/ss/inventory.py --cache data/ss/probe_cache.json \
        --md data/ss/INVENTORY.md --json data/ss/inventory.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# the four status categories (exactly four -- nothing else is a valid status)
# ---------------------------------------------------------------------------
STATUS_LANDED = "已落盘"
STATUS_TO_FETCH = "待获取"
STATUS_NETWORK_LIMITED = "网络受限"
STATUS_UNAVAILABLE = "不可得"
STATUSES = (STATUS_LANDED, STATUS_TO_FETCH, STATUS_NETWORK_LIMITED, STATUS_UNAVAILABLE)

# evidence tags
EVIDENCE_RECORDED = "recorded_fact"
EVIDENCE_LIVE = "live_probe"
EVIDENCE_NEEDS_PROBE = "requires_live_probe"
EVIDENCES = (EVIDENCE_RECORDED, EVIDENCE_LIVE, EVIDENCE_NEEDS_PROBE)

# probe modes
PROBE_LOCAL = "local_file"
PROBE_HTTP = "http"
PROBE_FTP = "ftp"
PROBE_TOOL = "tool_import"
PROBE_MANUAL = "manual"

# SS-annotation tri-state
SS_YES = "yes"
SS_NO = "no"
SS_PARTIAL = "partial"
SS_NA = "n/a"

#: Cluster root that holds the landed artefacts.  NOT mounted on the code-only
#: machine this repository is developed on, so local probes will miss here and
#: the entry is reported as recorded-fact / unprobed rather than verified.
DEFAULT_DATA_ROOT = os.environ.get("RNAJEPA_DATA_ROOT", "/mnt/cunyuliu/rna-jepa/data/raw")


@dataclass
class Source:
    """One audited data source (or teacher tool)."""

    key: str
    name: str
    kind: str                      # dataset | teacher | landed_asset
    probe: str                     # PROBE_*
    urls: List[str] = field(default_factory=list)
    mirror_candidates: List[str] = field(default_factory=list)
    expected_size_bytes: Optional[int] = None
    expected_md5: str = ""
    hash_status: str = "待核验"      # "known" once a real hash is recorded
    licence: str = "待核验"
    local_path: str = ""
    ss_annotations: str = SS_NO
    recorded_status: str = STATUS_TO_FETCH
    recorded_basis: str = ""        # where the recorded status comes from
    reason_code: str = ""
    notes: str = ""
    tool_import: str = ""           # for PROBE_TOOL
    version_lock: str = ""


# ---------------------------------------------------------------------------
# registry: landed assets (must be re-reported, spec §3.1)
# ---------------------------------------------------------------------------
def landed_assets() -> List[Source]:
    """The assets already on disk.  md5s are recorded from the completed fetch
    (``scripts/fetch_all_zenodo.sh``); they are measured facts, not plans."""
    api = "https://zenodo.org/api/records"
    root = DEFAULT_DATA_ROOT
    out: List[Source] = []
    zenodo_17786045 = [
        ("full_length", 281158, "3652178c257341010800e2d241a9c258"),
        ("Spliceator", 24191813, "d80c393eec09728c57e2c66c361e90e9"),
        ("protein", 30702614, "e8f6b277303959a2010d0047830b2c83"),
        ("te_ultra_full_length", 32611876, "939b495793687db362d4b9464a5df570"),
        ("CDS", 35586644, "dbb145edd36a67b63e7184da04dab8c4"),
        ("5UTR", 60237546, "6d36a52b06b6d493e900d60590e881da"),
        ("3UTR", 64124658, "6edf8dffcb2ee63560e276a65a5f5e9f"),
    ]
    for name, size, md5 in zenodo_17786045:
        out.append(Source(
            key=f"zenodo_17786045::{name}",
            name=f"Zenodo 17786045 / {name}.zip",
            kind="landed_asset",
            probe=PROBE_LOCAL,
            urls=[f"{api}/17786045/files/{name}.zip/content"],
            expected_size_bytes=size,
            expected_md5=md5,
            hash_status="known",
            licence="见 Zenodo 记录 17786045",
            local_path=f"{root}/{name}.zip",
            ss_annotations=SS_NO,
            recorded_status=STATUS_LANDED,
            recorded_basis="md5 已核（fetch_all_zenodo.sh 完成记录）",
            notes="mRNA 功能任务包；不含二级结构标注",
        ))
    out.append(Source(
        key="zenodo_12516160",
        name="Zenodo 12516160 / mRNAdataset.zip",
        kind="landed_asset",
        probe=PROBE_LOCAL,
        urls=[f"{api}/12516160/files/mRNAdataset.zip/content"],
        expected_size_bytes=20021485711,
        expected_md5="bf8bc5c946a0bd3b07716b1c7f785d54",
        hash_status="known",
        licence="见 Zenodo 记录 12516160",
        local_path=f"{root}/mRNAdataset.zip",
        ss_annotations=SS_NO,
        recorded_status=STATUS_LANDED,
        recorded_basis="md5 已核（20.02 GB 预训练语料）",
        notes="无标签序列语料；不含二级结构标注",
    ))
    out.append(Source(
        key="mrnabert_weights",
        name="mRNABERT 官方权重 YYLY66/mRNABERT",
        kind="landed_asset",
        probe=PROBE_LOCAL,
        urls=[],
        mirror_candidates=["huggingface.co/YYLY66/mRNABERT（待核验：镜像未实测）"],
        expected_size_bytes=None,
        expected_md5="",
        hash_status="待核验",
        licence="见 HuggingFace 模型卡",
        local_path=f"{root}/../weights/mRNABERT",
        ss_annotations=SS_NO,
        recorded_status=STATUS_LANDED,
        recorded_basis="86.6M 参数已落盘（74 词表，自定义 bert_layers）",
        notes="权重哈希未记录 -> 待核验；不含二级结构标注",
    ))
    return out


# ---------------------------------------------------------------------------
# registry: structure-annotation datasets to acquire (spec §3.3)
# ---------------------------------------------------------------------------
def structure_sources() -> List[Source]:
    """Sources whose reachability is UNPROBED here.  URLs/hashes are left blank
    and marked 待核验 wherever they are not on record -- never invented."""
    # The only URL the spec itself names (§3.3) is the bpRNA site.
    bprna_url = "https://bprna.cgrb.oregonstate.edu/"
    needs = "recorded_from_spec §3.3；站点可用性未经实测"
    return [
        Source(
            key="bprna_1m", name="bpRNA-1m（含 TR0/TS0/VL0 划分）", kind="dataset",
            probe=PROBE_HTTP, urls=[bprna_url],
            mirror_candidates=["待核验：spec §3.3 未指定镜像"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：~102,318 结构；含多来源汇总，冗余高",
            notes=needs + "；主训练集，须尽早实测",
        ),
        Source(
            key="bprna_new", name="bpRNA-new（低同源性子集）", kind="dataset",
            probe=PROBE_HTTP, urls=[bprna_url],
            mirror_candidates=["待核验"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：跨家族 OOD 主评测集",
            notes="仅评测，不可用于训练；" + needs,
        ),
        Source(
            key="archiveii", name="ArchiveII", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验：spec §3.3 记为『多镜像』，未指定具体 URL"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：~3,975；原始版冗余严重",
            notes="须去冗余并双版本（去冗余 + 原始）并列报告",
        ),
        Source(
            key="rnastralign", name="RNAStrAlign", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验：spec §3.3 记为『多镜像』"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：~30,451",
            notes="与 ArchiveII 有重叠，须交叉去冗余",
        ),
        Source(
            key="pdb_derived", name="PDB 衍生（RNAsolo / RCSB）", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验：spec §3.3 仅列名 RNAsolo / RCSB，未给 URL"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：数千",
            notes="需分辨率与方法过滤",
        ),
        Source(
            key="pseudobase_pp", name="PseudoBase++（假结集）", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：规模小，仅够 T2",
            notes="假结；规模小",
        ),
        Source(
            key="rna_puzzles_casp", name="RNA-Puzzles / CASP15-16 RNA", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验：官方站点未记录"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_YES,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：数十靶标",
            notes="仅评测，绝不进训练",
        ),
        Source(
            key="probing_data", name="SHAPE / DMS / icSHAPE / PARS", kind="dataset",
            probe=PROBE_HTTP, urls=[],
            mirror_candidates=["待核验：各原始论文补充材料，未记录"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_PARTIAL,
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.3：间接、噪声标签，须分级",
            notes="间接/噪声标签，须分级使用，不可当作结构真值",
        ),
        Source(
            key="rnacentral", name="RNAcentral", kind="dataset",
            probe=PROBE_FTP, urls=[],
            mirror_candidates=["待核验：FTP 主机未记录"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_NO,
            recorded_status=STATUS_NETWORK_LIMITED,
            recorded_basis="spec §3.4：数十 GB 量级，须单独评估；DNS/吞吐受限",
            reason_code="NETWORK_THROUGHPUT",
            notes="规模（数十 GB）远超结构数据集 1-2 GB；须先做可达性与吞吐实测",
        ),
        Source(
            key="rfam", name="Rfam", kind="dataset",
            probe=PROBE_FTP, urls=[],
            mirror_candidates=["待核验：FTP 主机未记录"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_PARTIAL,
            recorded_status=STATUS_NETWORK_LIMITED,
            recorded_basis="spec §3.3/§3.4：大，FTP；网络受限",
            reason_code="NETWORK_THROUGHPUT",
            notes="含一致性二级结构（covariance models），非逐碱基结构真值",
        ),
        Source(
            key="ensembl_gencode", name="Ensembl / GENCODE", kind="dataset",
            probe=PROBE_FTP, urls=[],
            mirror_candidates=["待核验：FTP 主机未记录"],
            expected_size_bytes=None, expected_md5="", hash_status="待核验",
            licence="待核验", ss_annotations=SS_NO,
            recorded_status=STATUS_NETWORK_LIMITED,
            recorded_basis="spec §3.3/§3.4：大，FTP；网络受限",
            reason_code="NETWORK_THROUGHPUT",
            notes="序列/注释，非二级结构标注",
        ),
        # ---- teacher tools (spec §3.5) ----
        Source(
            key="viennarna", name="ViennaRNA（McCaskill 配分函数）", kind="teacher",
            probe=PROBE_TOOL, tool_import="RNA", version_lock="ViennaRNA 2.x（Turner 参数随版本变化，须锁定）",
            urls=[], ss_annotations=SS_NA, licence="ViennaRNA 许可（待核验具体条款）",
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.5：需安装；本环境未安装",
            notes="System-2 教师 + 低置信回退求解器；安装需网络",
        ),
        Source(
            key="rnastructure", name="RNAstructure", kind="teacher",
            probe=PROBE_TOOL, tool_import="RNAstructure", version_lock="RNAstructure（版本锁定）",
            urls=[], ss_annotations=SS_NA, licence="待核验",
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.5：需安装；本环境未安装",
            notes="教师集成成员；安装需网络",
        ),
        Source(
            key="linearpartition", name="LinearPartition", kind="teacher",
            probe=PROBE_TOOL, tool_import="linearpartition", version_lock="LinearPartition（版本锁定）",
            urls=[], ss_annotations=SS_NA, licence="待核验",
            recorded_status=STATUS_TO_FETCH,
            recorded_basis="spec §3.5：需安装；本环境未安装",
            notes="教师集成成员；安装需网络",
        ),
    ]


def all_sources() -> List[Source]:
    return landed_assets() + structure_sources()


# ---------------------------------------------------------------------------
# measured network constraints (spec §3.4)
# ---------------------------------------------------------------------------
NETWORK_CONSTRAINTS: List[Dict[str, str]] = [
    {"constraint": "zenodo.org DNS 不可解析（返回 ::）",
     "consequence": "必须硬编码 IP（scripts/fetch_zenodo_parallel.sh 已实现 --resolve）"},
    {"constraint": "单连接吞吐被限速至 ~15-20 KB/s",
     "consequence": "单流下载不可行；须多连接分块"},
    {"constraint": "168 并发实测约 4 MB/s（32 conn ~460 KB/s，64 conn ~950 KB/s）",
     "consequence": "大文件走多连接 ranged download + md5 校验"},
    {"constraint": "本代码机无网络访问（大文件在集群 /mnt/cunyuliu，本机不可达）",
     "consequence": "本仓库只产出代码/清单；不执行下载"},
]


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _probe_local(src: Source, *, deep: bool = False) -> Dict[str, str]:
    path = src.local_path
    if not path or not os.path.exists(path):
        return {"status": STATUS_TO_FETCH, "detail": f"local path missing: {path}"}
    size = os.path.getsize(path)
    if src.expected_size_bytes is not None and size != src.expected_size_bytes:
        return {"status": STATUS_LANDED,
                "detail": f"present but size {size} != expected {src.expected_size_bytes}"}
    if deep and src.expected_md5:
        got = md5_file(path)
        if got != src.expected_md5:
            return {"status": STATUS_LANDED, "detail": f"size ok but md5 {got} != {src.expected_md5}"}
        return {"status": STATUS_LANDED, "detail": f"size+md5 verified ({got})"}
    return {"status": STATUS_LANDED, "detail": f"present, size {size} (md5 not checked)"}


def _probe_tool(src: Source) -> Dict[str, str]:
    try:
        importlib.import_module(src.tool_import)
        return {"status": STATUS_LANDED, "detail": f"import {src.tool_import!r} ok (version lock: {src.version_lock})"}
    except Exception as exc:  # noqa: BLE001 - any import failure means "not installed"
        return {"status": STATUS_TO_FETCH,
                "detail": f"import {src.tool_import!r} failed ({type(exc).__name__}); 需安装 + 版本锁定"}


def _probe_network(src: Source, *, timeout: float = 5.0) -> Dict[str, str]:
    url = src.urls[0] if src.urls else ""
    if not url:
        return {"status": STATUS_TO_FETCH, "detail": "no recorded URL to probe -> 待核验"}
    host = url.split("//", 1)[-1].split("/", 1)[0]
    try:
        socket.gethostbyname(host)
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_NETWORK_LIMITED,
                "detail": f"DNS for {host} failed ({type(exc).__name__}) -> 网络受限"}
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": STATUS_LANDED if resp.status < 400 else STATUS_TO_FETCH,
                    "detail": f"HTTP HEAD {url} -> {resp.status}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_NETWORK_LIMITED,
                "detail": f"HTTP probe failed ({type(exc).__name__}) -> 网络受限"}


def probe_source(src: Source, *, deep: bool = False, timeout: float = 5.0) -> Dict[str, str]:
    """Live-probe one source.  Only used in online mode."""
    if src.probe == PROBE_LOCAL:
        return _probe_local(src, deep=deep)
    if src.probe == PROBE_TOOL:
        return _probe_tool(src)
    if src.probe in (PROBE_HTTP, PROBE_FTP):
        return _probe_network(src, timeout=timeout)
    return {"status": src.recorded_status, "detail": "manual probe mode -> 使用 recorded 状态"}


# ---------------------------------------------------------------------------
# inventory build
# ---------------------------------------------------------------------------
def build_inventory(*, offline: bool = True, cache_path: Optional[str] = None,
                    deep: bool = False, timeout: float = 5.0,
                    data_root: Optional[str] = None) -> Dict[str, object]:
    """Build the inventory.

    Offline mode uses recorded state only and marks everything it did not probe
    as ``requires_live_probe`` / ``verified=False``.  A saved live-probe cache
    (written by a previous online run) may promote entries to ``live_probe`` /
    ``verified=True`` -- nothing else can.
    """
    cache: Dict[str, Dict[str, str]] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)

    sources = all_sources()
    if data_root:
        for src in sources:
            if src.probe == PROBE_LOCAL and src.local_path.startswith(DEFAULT_DATA_ROOT):
                src.local_path = src.local_path.replace(DEFAULT_DATA_ROOT, data_root, 1)

    entries: List[Dict[str, object]] = []
    for src in sources:
        entry = asdict(src)
        if offline:
            cached = cache.get(src.key)
            if cached:
                entry["status"] = cached["status"]
                entry["evidence"] = EVIDENCE_LIVE
                entry["verified"] = True
                entry["detail"] = cached.get("detail", "from probe cache")
            else:
                entry["status"] = src.recorded_status
                entry["evidence"] = (EVIDENCE_RECORDED if src.recorded_basis
                                     else EVIDENCE_NEEDS_PROBE)
                entry["verified"] = False
                entry["detail"] = src.recorded_basis or "no recorded basis -> requires live probe"
        else:
            probed = probe_source(src, deep=deep, timeout=timeout)
            entry["status"] = probed["status"]
            entry["evidence"] = EVIDENCE_LIVE
            entry["verified"] = True
            entry["detail"] = probed["detail"]
            cache[src.key] = {"status": probed["status"], "evidence": EVIDENCE_LIVE,
                              "detail": probed["detail"]}
        entries.append(entry)

    if not offline and cache_path:
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)

    inv: Dict[str, object] = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0, tzinfo=None).isoformat() + "Z",
        "mode": "offline" if offline else "online",
        "data_root": data_root or DEFAULT_DATA_ROOT,
        "statuses": list(STATUSES),
        "evidence_tags": list(EVIDENCES),
        "network_constraints": NETWORK_CONSTRAINTS,
        "ss_annotation_warning": (
            "已落盘资产（Zenodo 17786045 / 12516160 / mRNABERT）**不含任何二级结构标注**。"
            "17786045 是 145 个 mRNA 功能任务，12516160 是无标签序列。"
            "二级结构预测需要一条全新的数据获取工作流。"
        ),
        "entries": entries,
    }
    assert_no_assumed_available(inv)
    return inv


def assert_no_assumed_available(inv: Dict[str, object]) -> bool:
    """Enforce: no source is left in an assumed-available state.

    Every entry must carry a status from the four categories and an evidence
    tag from the three known tags.  An offline entry that was not probed must
    not be marked ``verified``.
    """
    entries = inv.get("entries", [])
    if not entries:
        raise AssertionError("inventory has no entries")
    for entry in entries:  # type: ignore[assignment]
        key = entry.get("key", "<unknown>")
        status = entry.get("status")
        if status not in STATUSES:
            raise AssertionError(f"source {key!r} has non-category status {status!r}")
        evidence = entry.get("evidence")
        if evidence not in EVIDENCES:
            raise AssertionError(f"source {key!r} has unknown evidence tag {evidence!r}")
        if entry.get("verified") and evidence != EVIDENCE_LIVE:
            raise AssertionError(
                f"source {key!r} marked verified but evidence is {evidence!r} (must be live_probe)")
    return True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _size(n: Optional[int]) -> str:
    if n is None:
        return "待核验"
    if n >= 10 ** 9:
        return f"{n/10**9:.2f} GB"
    if n >= 10 ** 6:
        return f"{n/10**6:.1f} MB"
    if n >= 10 ** 3:
        return f"{n/10**3:.1f} KB"
    return f"{n} B"


def render_markdown(inv: Dict[str, object]) -> str:
    entries = inv["entries"]  # type: ignore[index]
    lines: List[str] = []
    lines.append("# 二级结构数据盘查清单（实测状态，非计划）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：`{inv['generated_at_utc']}`")
    lines.append(f"- 模式：**{inv['mode']}**"
                 + ("（offline：仅使用 recorded 状态，未做任何探测）" if inv["mode"] == "offline" else "（online：逐源实测）"))
    lines.append(f"- 数据根目录：`{inv['data_root']}`")
    lines.append(f"- 状态类别（恰好四档）：{' / '.join(inv['statuses'])}")  # type: ignore[arg-type]
    lines.append("")
    lines.append("> ## ⚠️ 关键发现（必须置顶）")
    lines.append("> ")
    lines.append(f"> {inv['ss_annotation_warning']}")
    lines.append(">")
    lines.append("> **已落盘资产含二级结构标注数 = 0。** 不得复用现有语料充当结构标签。")
    lines.append("")

    lines.append("## 证据说明（recorded-fact vs live-probe）")
    lines.append("")
    lines.append("| evidence | 含义 | 是否可称『已核验』 |")
    lines.append("|---|---|---|")
    lines.append("| `recorded_fact` | 已记录在案的实测事实（如已完成 fetch 的 md5/大小） | 否（本环境未复测） |")
    lines.append("| `live_probe` | 本次进程真实探测（文件存在 / HTTP 可达 / 工具可导入） | 是 |")
    lines.append("| `requires_live_probe` | 仅有记录性预期，**尚未探测** | 否 |")
    lines.append("")

    def table(rows: List[Dict[str, object]], title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| key | 状态 | evidence | verified | 规模 | md5 | 许可 | SS 标注 | 备注 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for e in rows:
            md5 = e["expected_md5"] or "待核验"
            lines.append("| `{k}` | **{s}** | {ev} | {v} | {sz} | {m} | {l} | {ss} | {n} |".format(
                k=e["key"], s=e["status"], ev=e["evidence"],
                v="✅" if e["verified"] else "—",
                sz=_size(e["expected_size_bytes"]), m=md5, l=e["licence"],
                ss=e["ss_annotations"], n=(e["notes"] or e["detail"]),
            ))
        lines.append("")

    table([e for e in entries if e["kind"] == "landed_asset"], "1. 已落盘资产（可直接使用，均无结构标注）")
    table([e for e in entries if e["kind"] == "dataset"], "2. 待获取的结构标注数据（本项目必需）")
    table([e for e in entries if e["kind"] == "teacher"], "3. 教师模型依赖（非数据）")

    lines.append("## 4. 网络约束（实测）")
    lines.append("")
    lines.append("| 约束 | 后果 |")
    lines.append("|---|---|")
    for c in inv["network_constraints"]:  # type: ignore[union-attr]
        lines.append(f"| {c['constraint']} | {c['consequence']} |")
    lines.append("")

    counts: Dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    lines.append("## 5. 状态统计与断言")
    lines.append("")
    lines.append("| 状态 | 数量 |")
    lines.append("|---|---|")
    for s in STATUSES:
        lines.append(f"| {s} | {counts.get(s, 0)} |")
    lines.append("")
    lines.append("**断言（代码强制）**：每个 source 的状态必属四档之一，且必有 evidence 标签；"
                 "offline 模式下未探测的条目 `verified=False`，**不存在『假定可用』的 source**。")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RNA SS real-data inventory (measured audit)")
    ap.add_argument("--offline", action="store_true",
                    help="build from recorded state only (no network probes)")
    ap.add_argument("--cache", default="", help="live-probe cache JSON (read in offline, written in online)")
    ap.add_argument("--md", default="", help="write the markdown inventory to this path")
    ap.add_argument("--json", default="", help="write the machine-readable inventory to this path")
    ap.add_argument("--deep", action="store_true", help="online: also md5-verify local files (slow)")
    ap.add_argument("--timeout", type=float, default=5.0, help="online probe timeout (s)")
    ap.add_argument("--data-root", default="", help="override the landed-artefact root")
    args = ap.parse_args(argv)

    inv = build_inventory(offline=args.offline, cache_path=(args.cache or None),
                          deep=args.deep, timeout=args.timeout,
                          data_root=(args.data_root or None))
    md = render_markdown(inv)

    if args.md:
        with open(args.md, "w") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(inv, fh, ensure_ascii=False, indent=2)

    n = len(inv["entries"])  # type: ignore[arg-type]
    print(f"[inventory] mode={inv['mode']} sources={n} -> "
          f"md={args.md or '(stdout)'} json={args.json or '-'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
