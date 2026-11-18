from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "T-A233"
ROW = "> | T-A233 | 时间外 RNA 结构候选的数据、标签与暴露门 | 固定日期、官方元数据、独立计数、mmCIF/DSSR 守恒、精确/同源暴露与最终样本量齐全 | 候选池元数据已通过独立QA：1805入口、4087实体、1391唯一序列；16–8192nt为3584实体/1131唯一序列，≤1022nt为2588实体/841唯一序列；2条方法未知保留。完整mmCIF/DSSR标签提取进行中，未启动模型推理；候选数不等于独立测试集。metadata SHA256=`b79ecc8f…289c`，QA SHA256=`54f01470…8fea` |\n"
CHECK = "> - [ ] T-A233 时间外候选元数据独立QA已通过：1805入口、4087实体、1391唯一序列；长度过滤后1131唯一序列（16–8192nt），≤1022nt为841。完整DSSR标签、坐标映射、配对和同源门未闭环，不启动精度评测。\n"
SPEC = "> **时间外候选数据门（2026-11-18）**：固定initial_release_date≥2025-07-01，不按模型结果选样本。官方Search＋GraphQL冻结1805入口、4087 RNA实体，独立QA确认1391唯一序列；16–8192nt为3584实体/1131唯一序列，≤1022nt为2588实体/841唯一序列，2条实验方法未知保留。日期过滤不是历史预训练截止证明，唯一序列不是独立家族。完整mmCIF/DSSR标签、缺失坐标、配对守恒和同源审计未完成前禁止模型推理。metadata SHA256=`b79ecc8f3501966417738d85f94559da917ca20a26bd3a16e9f07b130c1d289c`，QA SHA256=`54f014707e285a2ed5f1e56e52780b23c5c389f7a23df72b599e91ebae5f8fea`。\n>\n"
CONTRACT = "> **时间外结构候选补充门（2026-11-18）**：为检验现有bpRNA-new同源未检出n=0造成的评估缺口，固定RCSB initial_release_date≥2025-07-01，不以任何模型表现调整日期或筛选候选。元数据独立QA通过1805入口、4087 RNA实体、1391唯一序列；16–8192nt为3584实体/1131唯一序列，≤1022nt为2588实体/841唯一序列。日期只是保守过滤，不证明历史预训练未接触；canonical polymer字符串不是坐标观测序列或配对标签。必须完成mmCIF哈希、polymer/label_seq映射、DSSR配对与nested删除守恒、缺失残基和精确/80%同源审计后才能冻结评测集，未通过不得启动模型推理或新训练。DSSR v1.9.10非商业使用，二进制SHA256=`c7261c0a…c39a`；完整标签任务在独立`/teams`目录运行。\n>\n"
LOG = """

## §15.957（2026-11-18）交接历史恢复与时间外 RNA 候选元数据门

- 发现本地交接目录为不完整镜像，整文件同步commit=`5b50129`曾将远端16096行旧台账替换为296行。已从`b223645`恢复全部旧文本并追加§15.925–956，最终16391行，SHA256=`e3ade160ae7e5bce388a9a0dcdae2076b9a29d41db2c811e61f57f3309a14cd4`；旧前缀、追加段和三处镜像逐字节通过。一次错误使用不存在系统python3及一次多算一行的断言未产生额外内容删除；以显式良好commit替代错误的HEAD~2基线后复核通过。问题与恢复保留在`history_recovery_qa.json`。
- 新版稿件3346词、SHA256=`5ec63380…6a1`，deck=29页、SHA256=`5a6e87ab…19a`，PDF=6页、SHA256=`09ba5a7a…3bab`；独立重建文本/表格/页数一致，首/中/末页视觉复核通过。
- 时间外候选固定initial_release_date≥2025-07-01，不使用模型表现。官方Search字段错误、GraphQL字段错误及方法缺失断言均保留并纠正；2条实验方法未知显式保留。最终1805入口、4087 RNA实体、1391唯一序列；16–8192nt为3584实体/1131唯一序列，≤1022nt为2588实体/841唯一序列。元数据SHA256=`b79ecc8f…289c`，独立QA SHA256=`54f01470…8fea`，代码commit=`a1c9a41`。
- DSSR v1.9.10二进制SHA256=`c7261c0a…c39a`，非商业使用；既有工具16项回归通过。4条固定工程样本均完成：前三条无可接受链内配对，10FZ_A保留1542nt、430个canonical-nested配对；未知HSX不强行映射。完整1805入口标签提取在独立目录运行，尚未通过完整标签/同源门，未启动模型推理或训练。
"""


def prepend(path, text):
    content = path.read_text()
    if MARKER in content or "时间外候选数据门" in content:
        return
    marker = "> **历史执行覆盖（2026-11-16"
    assert marker in content, path
    path.write_text(content.replace(marker, SPEC + marker, 1))


def main():
    for relative in ["spec.md", "spec/spec.md", "spec_sync/spec.md"]:
        prepend(ROOT / relative, SPEC)
    for relative in ["tasks.md", "spec/tasks.md", "spec_sync/tasks.md"]:
        path = ROOT / relative
        content = path.read_text()
        if MARKER not in content:
            lines = content.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if MARKER.replace("233", "232") in line) + 1
            lines.insert(index, ROW)
            path.write_text("".join(lines))
    for relative in ["checklist.md", "spec/checklist.md", "spec_sync/checklist.md"]:
        path = ROOT / relative
        content = path.read_text()
        if MARKER not in content:
            lines = content.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if MARKER.replace("233", "232") in line) + 1
            lines.insert(index, CHECK)
            path.write_text("".join(lines))
    path = ROOT / "docs/plans/2026-11-16-dual-view-jepa-accuracy-contract.md"
    content = path.read_text()
    if "时间外结构候选补充门" not in content:
        anchor = "> **H2 编码器身份限制"
        assert anchor in content
        path.write_text(content.replace(anchor, CONTRACT + anchor, 1))
    log = ROOT / "records/DECISION_TRAINING_LOG.md"
    content = log.read_text()
    assert "## §15.895" in content and "## §15.925" in content
    if "## §15.957" not in content:
        log.write_text(content.rstrip() + LOG + "\n")
    for name in ["spec.md", "tasks.md", "checklist.md"]:
        assert (ROOT / name).read_bytes() == (ROOT / "spec" / name).read_bytes() == (ROOT / "spec_sync" / name).read_bytes()
    print("PASS: temporal candidate gate added without deleting historical content")


if __name__ == "__main__":
    main()
