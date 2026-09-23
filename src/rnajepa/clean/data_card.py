"""``data_card.md`` template generator (spec §4 C5 / Task 9 SubTask 9.4).

The generated card records provenance, the cleaning stages actually applied,
scale, the licence check, version/hash information and — most importantly — the
**known defects** section, which is mandatory rather than optional.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .c5_audit import manifest_digest
from .records import Record

KNOWN_DEFECTS_HEADER = (
    "现有 145 个下游任务与 20 GB 预训练语料**不含任何二级结构标注**；"
    "本数据卡描述的数据集为**全新获取**，不得与既有 mRNA 语料混同。"
)


def render_data_card_template(
    *,
    dataset_name: str,
    sources: Sequence[str],
    kept: Sequence[Record],
    pseudoknot_count: int = 0,
    attrition_markdown: str = "",
    licence_report: Optional[Dict[str, object]] = None,
    split_sizes: Optional[Dict[str, int]] = None,
    split_distances: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, object]] = None,
    known_defects: Optional[Sequence[str]] = None,
    version_digest: Optional[str] = None,
    extraction_date: str = "",
) -> str:
    """Render the ``data_card.md`` template for one dataset.

    Returns a markdown string; callers decide where to write it.
    """
    defects: List[str] = [KNOWN_DEFECTS_HEADER]
    defects.extend(known_defects or [])

    lines: List[str] = [
        f"# Data card — {dataset_name}",
        "",
        f"- 抽取日期: {extraction_date or '（待填）'}",
        f"- 版本摘要 (version digest): {version_digest or '（待填：manifest digest）'}",
        f"- 记录数 (kept): {len(kept)}",
        f"- 假结集 (routed out): {pseudoknot_count}",
        "",
        "## 1. 来源 (Provenance)",
        "",
    ]
    for source in sources:
        lines.append(f"- {source}")

    lines += ["", "## 2. 清洗流程与配置 (Cleaning pipeline)", ""]
    if config:
        lines.append("```yaml")
        for key, value in sorted(config.items()):
            lines.append(f"{key}: {value}")
        lines.append("```")
    else:
        lines.append("（待填：C1-C6 各级实际使用的阈值与工具版本）")

    lines += ["", "### 逐级衰减表 (Attrition)", ""]
    lines.append(attrition_markdown or "（待填）")

    lines += ["", "## 3. 规模 (Scale)", ""]
    if split_sizes:
        for name, size in sorted(split_sizes.items()):
            lines.append(f"- {name}: {size}")
    else:
        lines.append("（待填：各 split 记录数）")

    lines += ["", "## 4. 许可合规 (Licence check)", ""]
    if licence_report:
        per_source = licence_report.get("per_source", {})
        for source, licences in sorted(per_source.items()):
            lines.append(f"- {source}: {', '.join(licences)}")
        unknown = licence_report.get("sources_with_unknown_licence", [])
        lines.append("")
        lines.append(
            f"- ⚠️ 许可未知的来源: {', '.join(unknown) if unknown else '无'}"
        )
    else:
        lines.append("（待填）")

    lines += ["", "## 5. 分布与 split 间分布距离 (Distribution)", ""]
    if split_distances:
        for key, value in sorted(split_distances.items()):
            lines.append(f"- {key}: {value:.4f}")
    else:
        lines.append("（待填：长度/GC/家族/配对密度分布，以及 split 间分布距离）")

    lines += ["", "## 6. 已知缺陷 (Known defects)", ""]
    for defect in defects:
        lines.append(f"- {defect}")

    lines += [
        "",
        "## 7. 使用限制 (Usage restrictions)",
        "",
        "- 测试集**不参与任何超参数选择**（spec 执行边界红线 1）。",
        "- 跨家族 OOD 集（bpRNA-new）**不得用于训练**。",
        "- RNA-Puzzles / CASP 靶标**仅用于评测**。",
        "",
    ]
    return "\n".join(lines)


def render_data_card_from_result(
    result,
    *,
    dataset_name: str,
    sources: Sequence[str],
    known_defects: Optional[Sequence[str]] = None,
    extraction_date: str = "",
) -> str:
    """Convenience wrapper around :func:`render_data_card_template`."""
    stats = result.stats or {}
    return render_data_card_template(
        dataset_name=dataset_name,
        sources=sources,
        kept=result.kept,
        pseudoknot_count=len(result.pseudoknots),
        attrition_markdown=result.attrition.to_markdown(),
        licence_report=stats.get("licences"),
        split_sizes={k: len(v) for k, v in (result.splits or {}).items()},
        split_distances=stats.get("split_distances"),
        config=result.config,
        known_defects=known_defects,
        version_digest=manifest_digest(
            {"dataset": dataset_name, "n": len(result.kept), "sources": list(sources)}
        ),
        extraction_date=extraction_date,
    )
