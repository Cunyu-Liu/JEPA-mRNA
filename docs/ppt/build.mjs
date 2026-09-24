/**
 * RNA-JEPA project progress deck — generator.
 *
 * Design brief (consulting-research recipe, "more academic" remix)
 * ---------------------------------------------------------------
 * Visual thesis: analytical, compressed, calm — a decision document, not a
 * keynote. Design tension: rigorous *and* honest, so the one slide that reports
 * a failed criterion carries the same visual weight as the ones reporting gains.
 *
 * The deck has to survive being read rather than watched: every page states a
 * claim in its title, puts the evidence directly under it, and keeps the
 * implication and the source visible at the bottom. Numbers are always paired
 * with the split / step count they came from, because a bare F1 in this project
 * has already been misread once.
 *
 * Structure follows the four-layer split the PptxGenJS guidance asks for:
 * content model -> design tokens -> primitives -> compositions.
 */

import { createRequire } from "node:module";
import { addSingleLineToken } from "/Users/liucunyu/.trae-cn/plugins/trae-remote-official/jingmei-ppt/1.0.0/skills/jingmei-ppt/scripts/pptx-text-guards.mjs";

const require = createRequire(import.meta.url);
const PptxGenJS = require("pptxgenjs");

// ---------------------------------------------------------------- role tokens
const T = {
  ink: "0F1B2D",          // primary text
  muted: "5A6472",        // secondary text
  faint: "8A93A0",        // annotations
  accent: "1F3A5F",       // navy: structure, table headers
  evidence: "0E7C86",     // teal: section labels, evidence emphasis
  positive: "1B7F4B",
  caution: "B26A00",
  risk: "B3261E",
  surface: "F4F6F8",
  rule: "D8DEE6",
  white: "FFFFFF",
};

const FONT_CN = "PingFang SC";
const FONT_NUM = "Arial";

const M = { left: 0.62, right: 0.62, contentW: 13.333 - 0.62 * 2 };
const BAND = { label: 0.34, claim: 0.60, rule: 1.30, evidence: 1.52, foot: 7.02 };
const PAGE = { w: 13.333, h: 7.5 };

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_16x9";
pptx.author = "RNA-JEPA";
pptx.title = "RNA-JEPA 项目进展 2026-09-24";

const FOOTER_TEXT = "RNA-JEPA 决策模型线 · 2026-09-24 · 提交 3bb5186";

// ----------------------------------------------------------------- primitives
function rule(slide, x, y, w, color = T.rule, h = 0.01) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color }, line: { color, width: 0 } });
}

function token(slide, value, opts) {
  return addSingleLineToken(slide, value, { fontFace: FONT_NUM, ...opts });
}

/** Section label + claim title + hairline + footer. The claim is a sentence. */
function chrome(slide, { label, claim, page, claimColor = T.ink }) {
  slide.addText(label, {
    x: M.left, y: BAND.label, w: M.contentW, h: 0.24,
    fontFace: FONT_CN, fontSize: 11.5, bold: true, color: T.evidence,
    charSpacing: 1.2, margin: 0, valign: "middle",
  });
  slide.addText(claim, {
    x: M.left, y: BAND.claim, w: M.contentW, h: 0.62,
    fontFace: FONT_CN, fontSize: 21, bold: true, color: claimColor,
    margin: 0, valign: "middle", wrap: true,
  });
  rule(slide, M.left, BAND.rule, M.contentW);
  slide.addText(FOOTER_TEXT, {
    x: M.left, y: BAND.foot, w: M.contentW - 0.8, h: 0.22,
    fontFace: FONT_CN, fontSize: 9.5, color: T.faint, margin: 0, valign: "middle",
  });
  token(slide, String(page).padStart(2, "0"), {
    x: PAGE.w - M.right - 0.7, y: BAND.foot, w: 0.7, h: 0.22,
    fontSize: 10.5, bold: true, color: T.accent, align: "right",
  });
  rule(slide, M.left, BAND.foot - 0.14, M.contentW, T.rule);
}

function sourceNote(slide, text, y = 6.62) {
  slide.addText(text, {
    x: M.left, y, w: M.contentW, h: 0.30,
    fontFace: FONT_CN, fontSize: 9.5, color: T.muted, margin: 0, valign: "top", wrap: true,
  });
}

/** One row of key statistics. Values go through the single-line guard.
 *
 * Heights are set so a 30pt value and a two-line caption both fit: a single 30pt
 * line needs 30*1.34/72 = 0.56in, so the value box is 0.60in rather than the
 * 0.48in that looked right and overflowed (caught by the layout gate below).
 */
function stats(slide, items, { x = M.left, y, w = M.contentW, valueSize = 30, cardH = 1.40 } = {}) {
  const gap = 0.22;
  const cw = (w - gap * (items.length - 1)) / items.length;
  items.forEach((it, i) => {
    const cx = x + i * (cw + gap);
    slide.addShape(pptx.ShapeType.rect, {
      x: cx, y, w: cw, h: cardH,
      fill: { color: it.fill || T.surface }, line: { color: T.rule, width: 0.75 },
    });
    token(slide, it.value, {
      x: cx + 0.14, y: y + 0.12, w: cw - 0.28, h: 0.60,
      fontSize: valueSize, bold: true, color: it.color || T.accent,
    });
    slide.addText(it.label, {
      x: cx + 0.14, y: y + 0.74, w: cw - 0.28, h: cardH - 0.88,
      fontFace: FONT_CN, fontSize: 10.5, color: T.muted, margin: 0, valign: "top", wrap: true,
    });
  });
}

/** Native PowerPoint table: editable, exact comparison, no flattened images. */
function table(slide, { x = M.left, y, w = M.contentW, colW, header, rows, rowH = 0.30,
                         fontSize = 11, headerSize = 10.5, flagCol = -1, flagMap = {} }) {
  const total = colW.reduce((a, b) => a + b, 0);
  const scaled = colW.map((c) => (c / total) * w);
  const head = header.map((h) => ({
    text: h,
    options: {
      fontFace: FONT_CN, fontSize: headerSize, bold: true, color: T.white,
      fill: { color: T.accent }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08],
    },
  }));
  const body = rows.map((r) =>
    r.map((cell, ci) => {
      const isObj = cell && typeof cell === "object" && !Array.isArray(cell);
      const text = isObj ? cell.text : cell;
      const color = isObj && cell.color ? cell.color
        : (flagCol === ci && flagMap[text] ? flagMap[text] : T.ink);
      const bold = Boolean((isObj && cell.bold) || (flagCol === ci && flagMap[text]));
      return {
        text: String(text),
        options: {
          // numerals get Arial's reliable metrics
          fontFace: /^[\d.,%+\-–—/\s]+$/.test(String(text)) ? FONT_NUM : FONT_CN,
          fontSize, bold, color,
          valign: "middle", margin: [0.03, 0.08, 0.03, 0.08],
        },
      };
    }));
  slide.addTable([head, ...body], {
    x, y, w, colW: scaled, rowH,
    border: { type: "solid", color: T.rule, pt: 0.5 },
    fill: { color: T.white },
    autoPage: false,
    valign: "middle",
  });
  return y + rowH * (rows.length + 1);
}

function bullets(slide, items, { x = M.left, y, w = M.contentW, size = 12.5, gapY = 0.06 }) {
  const step = size / 72 + 0.155 + gapY;
  items.forEach((it, i) => {
    const isObj = typeof it === "object";
    const text = isObj ? it.text : it;
    const color = isObj && it.color ? it.color : T.ink;
    slide.addText([
      { text: isObj && it.mark ? `${it.mark}  ` : "—  ",
        options: { bold: true, color: isObj && it.markColor ? it.markColor : T.evidence } },
      { text, options: { color } },
    ], {
      x, y: y + i * step, w,
      fontFace: FONT_CN, fontSize: size, margin: 0, valign: "top", wrap: true,
    });
  });
}

// ------------------------------------------------- programmatic overflow gate
// The guidance asks for a cheap structural check in place of a full render. Every
// text box is recorded and its wrapped height estimated, so a box whose copy
// cannot fit is reported rather than discovered by eye in a thumbnail. Wrapping
// is estimated with full-width characters at 1em and ASCII at 0.55em, which is
// the ratio that actually governs this deck (Chinese prose carrying Latin
// figures such as "micro F1" and "6,022,538").
const LAYOUT = [];
const _firstSlide = pptx.addSlide();
const SlideProto = Object.getPrototypeOf(_firstSlide);
const _origAddText = SlideProto.addText;
function flattenText(t) {
  if (Array.isArray(t)) return t.map((r) => (typeof r === "object" ? r.text : r)).join("");
  return String(t ?? "");
}
SlideProto.addText = function (text, opts = {}) {
  LAYOUT.push({
    text: flattenText(text), x: opts.x, y: opts.y, w: opts.w, h: opts.h,
    fontSize: opts.fontSize ?? 18, wrap: opts.wrap !== false,
  });
  return _origAddText.call(this, text, opts);
};

// --------------------------------------------------------------------- slides
// S1 cover
{
  const s = _firstSlide;
  s.background = { color: T.ink };
  for (let i = 0; i < 14; i += 1) {
    rule(s, 0.62 + i * 0.30, 0.62, 0.14, "2C4059", 0.055);
  }
  s.addText("RNA-JEPA", {
    x: 0.62, y: 2.42, w: 11, h: 0.5,
    fontFace: FONT_NUM, fontSize: 15, bold: true, color: T.evidence, charSpacing: 3, margin: 0,
  });
  s.addText("项目进展汇报", {
    x: 0.62, y: 2.86, w: 11, h: 1.0,
    fontFace: FONT_CN, fontSize: 44, bold: true, color: T.white, margin: 0, valign: "middle",
  });
  s.addText("决策模型线 · 训练数据 / 测评基准 / 基线三条线的全面复核", {
    x: 0.62, y: 3.98, w: 11, h: 0.4,
    fontFace: FONT_CN, fontSize: 16, color: "B9C4D2", margin: 0, valign: "middle",
  });
  rule(s, 0.62, 4.62, 4.2, T.evidence, 0.03);
  s.addText(
    "本轮推翻了三条已写进记录的旧结论，并得到一条否证本项目核心主张的实测结果。"
    + "以下所有数字均标注 split、步数与口径。",
    {
      x: 0.62, y: 4.84, w: 9.6, h: 0.8,
      fontFace: FONT_CN, fontSize: 14, color: "D6DEE8", margin: 0, valign: "top", wrap: true,
    });
  s.addText("2026-09-24", {
    x: 0.62, y: 6.42, w: 4, h: 0.3,
    fontFace: FONT_NUM, fontSize: 14, bold: true, color: T.white, margin: 0,
  });
  s.addText("git@github.com:Cunyu-Liu/JEPA-mRNA.git  ·  提交 3bb5186", {
    x: 0.62, y: 6.76, w: 11, h: 0.28,
    fontFace: FONT_NUM, fontSize: 10, color: "8A93A0", margin: 0,
  });
}

// S2 headline findings
{
  const s = pptx.addSlide();
  chrome(s, { label: "总览", claim: "三条旧结论被推翻，一条核心主张被自己的测量否证", page: 2 });
  stats(s, [
    { value: "4.29×", label: "训练集扩容（10,682 → 45,865）\n实测去重后，不是 8.5×", color: T.accent },
    { value: "0.162", label: "与最强基线的 F1 差距\nRNAformer 0.7578 / 0.5958", color: T.risk },
    { value: "0.0015", label: "RNAformer 概率的 ECE\n无需重标定，优于 0.0031", color: T.risk },
    { value: "FAIL", label: "C1-a 判据\n已加写作红线，禁「首次」", color: T.risk, fill: "FDF1F0" },
  ], { y: BAND.evidence });

  bullets(s, [
    { mark: "旧结论 1", text: "训练数据只盘了 BPfold_data。集群另有 RNAformer 参考发布（biophysical 416,479 行等），此前完全未被调研。", markColor: T.risk },
    { mark: "旧结论 2", text: "「UFold / RNAformer / EternaFold 权重不可得」是错的：权重一直就在集群本地，源码本轮 clone 成功。", markColor: T.risk },
    { mark: "旧结论 3", text: "「学习型基线都弱」只在 bprna_ts0 成立；按 split 会反转（pdb_ts1 上 ViennaRNA 0.7299 > UFold 0.6434）。", markColor: T.risk },
    { mark: "新反证", text: "C1 原表述「首次给出经校准验证的免 DP 配对概率」被 RNAformer 的实测否证。", markColor: T.risk },
  ], { y: 3.16, size: 12.5, gapY: 0.04 });

  // implication band: fills the lower third and states the consequence, rather
  // than leaving an empty strip above the source line
  s.addShape(pptx.ShapeType.rect, {
    x: M.left, y: 5.02, w: M.contentW, h: 1.36,
    fill: { color: "FDF1F0" }, line: { color: T.risk, width: 1 },
  });
  s.addText("这意味着什么", {
    x: M.left + 0.20, y: 5.14, w: 3.0, h: 0.26,
    fontFace: FONT_CN, fontSize: 11.5, bold: true, color: T.risk, margin: 0,
  });
  s.addText(
    "本轮的价值不在新增了多少数字，而在两条线被同时纠正：数据线的旧结论是调研不足，"
    + "基线线的旧结论是把网络探测失败当成了资产不存在。"
    + "而纠正之后暴露出的 0.162 F1 差距和 C1 的反证，才是真正需要决策的东西。",
    {
      x: M.left + 0.20, y: 5.42, w: M.contentW - 0.40, h: 0.86,
      fontFace: FONT_CN, fontSize: 12, color: T.ink, margin: 0, valign: "top", wrap: true,
    });

  sourceNote(s, "口径：F1 为 micro（汇总 TP/FP/FN），GT 限制为规范+嵌套配对；ECE 为同 split、同 6,022,538 个候选对。详见 records/DECISION_TRAINING_LOG.md §14.29–§14.42。");
}

// S3 training data
{
  const s = pptx.addSlide();
  chrome(s, { label: "训练数据", claim: "扩容是 4.29×，不是 8.5×：一半以上的「新数据」是重复的", page: 3 });
  table(s, {
    y: BAND.evidence,
    colW: [3.5, 1.6, 1.6, 1.4, 4.4],
    header: ["来源", "输入条数", "去重后保留", "保留率", "说明"],
    rowH: 0.335,
    flagCol: 3,
    flagMap: { "68.7%": T.ink, "22.7%": T.caution, "5.9%": T.caution, "0.0%": T.risk, "100.0%": T.ink, "46.8%": T.risk },
    rows: [
      ["bprna_tr0（既有语料）", "10,682", "10,682", "100.0%", "本轮扩容的基线"],
      ["ref_tr_bprna", "36,865", "25,320", "68.7%", "bpRNA-1m 的另一份发布，与既有语料大量重合"],
      ["ref_tr_experimental", "42,283", "9,582", "22.7%", "实验结构；多数已在上面两行中出现过"],
      ["ref_tr_intra_family", "4,803", "281", "5.9%", "几乎全部冗余"],
      ["ref_tr_inter_family", "3,462", "0", "0.0%", "完全冗余，贡献 0 条"],
      ["合计", "98,095", "45,865", "46.8%", "报告 98K 会虚报 2.1 倍"],
    ],
  });
  bullets(s, [
    { mark: "方法", text: "tools/concat_corpora.py：只做逐序列精确去重，绝不自称做了同源去冗余（那是 mmseqs 的下一步）。" },
    { mark: "教师", text: "45,865 条 ViennaRNA 2.7.2 软标签已完成：12 进程 892 s（实测吞吐 L=100 为 34 seq/s，L=500 为 0.6 seq/s）。" },
  ], { y: 4.72, size: 12.5 });
  sourceNote(s, "Manifest：ss_data/jsonl/bprna_tr1.manifest.json（含逐来源 n_in / n_kept / sha256）。");
}

// S4 data-quality traps
{
  const s = pptx.addSlide();
  chrome(s, { label: "数据质量", claim: "质量是好的，但有三个必须记录的缺陷，它们各自设了一个上界", page: 4 });
  const y0 = BAND.evidence;
  const cw = (M.contentW - 0.44) / 3;
  const cards = [
    {
      title: "标注自相矛盾",
      value: "5.5%",
      body: "5,393 / 98,095 条：同一序列带着不同结构出现，其中 992 条是同一个文件内部冲突。\n"
          + "→ 构成与训练量无关的 precision 上界，加数据消不掉。",
      color: T.caution,
    },
    {
      title: "合成集不是泛化基准",
      value: "MFE 1.0000",
      body: "ref_synthetic_test 的真值就是 ViennaRNA 的 MFE 结构（MFE micro F1 = 1.0000）。\n"
          + "→ 蒸馏一致性集，不得与 bprna_ts0 并列进主表。",
      color: T.caution,
    },
    {
      title: "配对口径不一致",
      value: "9.1–23.7%",
      body: "参考发布原始配对集中非规范配对的占比，而我们既有 .bpseq 语料是 0.000。\n"
          + "→ 两个来源的 F1 不是同一个地面，必须显式说明删了多少对。",
      color: T.caution,
    },
  ];
  cards.forEach((c, i) => {
    const x = M.left + i * (cw + 0.22);
    s.addShape(pptx.ShapeType.rect, {
      x, y: y0, w: cw, h: 2.55, fill: { color: "FDFAF3" }, line: { color: T.rule, width: 0.75 },
    });
    rule(s, x, y0, cw, c.color, 0.045);
    s.addText(c.title, {
      x: x + 0.16, y: y0 + 0.14, w: cw - 0.32, h: 0.28,
      fontFace: FONT_CN, fontSize: 12.5, bold: true, color: T.ink, margin: 0,
    });
    token(s, c.value, {
      x: x + 0.16, y: y0 + 0.46, w: cw - 0.32, h: 0.46,
      fontSize: 24, bold: true, color: c.color,
    });
    s.addText(c.body, {
      x: x + 0.16, y: y0 + 0.98, w: cw - 0.32, h: 1.45,
      fontFace: FONT_CN, fontSize: 11, color: T.muted, margin: 0, valign: "top", wrap: true,
    });
  });
  bullets(s, [
    { mark: "已通过", text: "既有语料的内部一致性仍然成立：内部精确重复 0、发夹环违规 0、非经典配对 0.000。" },
    { mark: "新规则", text: "参考发布新增三条显式规则并各带测试：pk 栏精确删假结 / 非规范配对删除 / 多联体解析。" },
  ], { y: 4.40, size: 12.5 });
  sourceNote(s, "工具：tools/concat_corpora.py（常驻计数 conflicting_structure_kept_first）、data/ss/prepare_decision_data.py（21 项新测试）。");
}

// S5 benchmark expansion
{
  const s = pptx.addSlide();
  chrome(s, { label: "测评基准", claim: "benchmark 从 2 个 split 扩到 9 个，且用的是已发表数字的同一批集合", page: 5 });
  table(s, {
    y: BAND.evidence,
    colW: [3.0, 1.2, 1.1, 5.4],
    header: ["新增 split", "条数", "被拒", "用途"],
    rowH: 0.305,
    rows: [
      ["ref_pdb_ts1", "63", "4", "实验标签（PDB）——唯一能回答「噪声标签上界」的集合"],
      ["ref_pdb_ts2", "39", "0", "实验标签"],
      ["ref_pdb_ts3", "19", "0", "实验标签（难例）"],
      ["ref_pdb_ts_hard", "28", "0", "对抗子集"],
      ["ref_synthetic_test", "3,344", "0", "合成集；真值 = ViennaRNA MFE，仅作蒸馏一致性"],
      ["ref_synthetic_valid", "2,727", "0", "新的选择用验证集，替代组成偏斜的 VL0"],
      ["ref_bprna_ts0", "1,291", "14", "与既有 TS0（1,288）同源，两个独立解析器的交叉核验"],
    ],
  });
  bullets(s, [
    { mark: "为什么", text: "这些就是 RNAformer 已发表数字所用的集合，因此 F1 与该基线在同一地面上可比，不需要「看起来相似」的换算。" },
    { mark: "待解释", text: "ref_bprna_ts0（1,291）与既有 TS0（1,288）差 3 条，来源可能是 N 碱基拒绝（14 条）；解释清楚之前两个 split 不混用。" },
  ], { y: 4.68, size: 12.5 });
  sourceNote(s, "构建：scripts/build_ref_corpus.sh，与既有语料走同一条投影/校验代码路径；manifest 记录删除了多少配对。");
}

// S6 strong baselines
{
  const s = pptx.addSlide();
  chrome(s, { label: "基线实测", claim: "最强基线是 RNAformer（0.7578），我们差 0.162——比此前以为的大一倍", page: 6 });
  table(s, {
    y: BAND.evidence,
    colW: [4.6, 2.0, 2.0, 3.6],
    header: ["模型（同一 split：ref_bprna_ts0，1,291 条）", "micro F1", "macro F1", "备注"],
    rowH: 0.35,
    flagCol: 1,
    flagMap: { "0.7578": T.risk, "0.6598": T.caution, "0.5958": T.accent },
    rows: [
      ["RNAformer（32M, bprna checkpoint）", "0.7578", "0.7454", "最强已发布基线，本轮首次跑通"],
      ["UFold", "0.6598", "0.7221", "权重一直就在集群本地"],
      ["我们（rinalmo_ff @step 20000, w=-1）", "0.5958", "0.5970", "无任何选择；高于 ViennaRNA centroid"],
      ["ViennaRNA centroid", "0.5393", "0.5288", "纯物理基线"],
      ["ViennaRNA MFE", "0.5056", "—", "—"],
    ],
  });
  stats(s, [
    { value: "0.162", label: "与 RNAformer 的差距\n占其绝对水平的 21%", color: T.risk, fill: "FDF1F0" },
    { value: "0.064", label: "与 UFold 的差距\n只比 UFold 会低估差距 2.5 倍", color: T.caution },
    { value: "+0.057", label: "高出 ViennaRNA centroid\n方向正确，幅度不足", color: T.positive },
  ], { y: 4.52, valueSize: 26 });
  sourceNote(s, "合法性：非法结构率 0、发夹环违规率 0。全部基线走同一份 JSONL 与同一套 ss.metrics 实现。");
}

// S7 undertraining
{
  const s = pptx.addSlide();
  chrome(s, { label: "核心结果", claim: "欠训练是此前低分的主因：同一配置跑满 20,000 步，TS0 从 0.4953 升到 0.5956", page: 7 });
  table(s, {
    y: BAND.evidence,
    colW: [2.4, 2.0, 2.0, 2.0, 4.0],
    header: ["训练步数", "20（未训练）", "1,000", "3,500", "6,000 → 10,000 → 20,000"],
    rowH: 0.40,
    rows: [
      ["micro F1", "0.2167", "0.4554", "0.4953", "0.5369 → 0.5587 → 0.5956"],
      ["macro F1", "0.2372", "0.4592", "0.4858", "0.5384 → 0.5549 → 0.5965"],
      ["ECE（裸头）", "0.5547", "0.1837", "0.1373", "0.2126 → 0.1965 → 0.1957"],
    ],
  });
  bullets(s, [
    { mark: "+0.1003", text: "3,500 → 20,000 步的 F1 增益。3,500 步时只训练了 0.56 epoch（batch 4 × 3,500 ÷ 10,682）。", markColor: T.positive },
    { mark: "旧 headline", text: "0.4953 必须停止引用：它停在「低于两个参数化热力学基线」的位置，而原因是训练不足，不是方法不行。", markColor: T.risk },
    { mark: "注意", text: "ECE 不随训练单调改善（0.1373 → 0.1957）：F1 上升、校准变差，这正是 C1 需要单独立论的理由。", markColor: T.caution },
  ], { y: 4.28, size: 12.5, gapY: 0.02 });
  sourceNote(s, "单 seed、单 checkpoint。协议要求 ≥5 seed；seed 1–5 目前只跑到 2,125–4,850 步，因此暂不能报告 mean ± std。");
}

// S8 calibration table
{
  const s = pptx.addSlide();
  chrome(s, { label: "校准（C1）", claim: "同一 split、同一 6,022,538 个候选对、同一套指标实现下的完整对比", page: 8 });
  table(s, {
    y: BAND.evidence,
    colW: [4.2, 1.5, 1.5, 1.5, 2.3],
    header: ["概率来源", "ECE", "Brier", "NLL", "需要后处理重标定？"],
    rowH: 0.345,
    flagCol: 4,
    flagMap: { "不需要": T.positive, "需要（2 参数拟合）": T.caution },
    rows: [
      [{ text: "RNAformer（最强基线）", bold: true }, "0.0015", "0.0025", "0.0132", "不需要"],
      [{ text: "我们的 CRF 精确边际", bold: true }, "0.0004", "0.0030", "0.0137", "不需要"],
      [{ text: "我们 免 DP 仿射重标定", bold: true }, "0.0031", "0.0057", "0.0320", "需要（2 参数拟合）"],
      ["ViennaRNA 精确 BPP", "0.0048", "0.0051", "0.0259", "不需要"],
      ["UFold", "0.0147", "0.0108", "0.0437", "不需要"],
      [{ text: "我们 裸头", bold: true }, "0.1955", "0.1015", "0.3310", "—"],
    ],
  });
  bullets(s, [
    { mark: "独立复核", text: "UFold 的概率在期望意义上高估配对数 4.10 倍（平均预测 112.9 对 vs 真值 27.5 对）——这点为真，但它撑不起「既有方法都没校准」。" },
    { mark: "反例", text: "RNAformer 就是单次前向出 L×L 概率 + 非交叉贪心解码（免 DP），且概率实测已校准，无需任何后处理。", color: T.risk },
  ], { y: 4.62, size: 12.5 });
  sourceNote(s, "eval/ss/reference_calibration.py 新增 --external-probs，使外部基线的概率与我们的头走同一份 pooled_pair_calibration。");
}

// S9 C1 verdict (risk page)
{
  const s = pptx.addSlide();
  chrome(s, { label: "C1 判定", claim: "C1-a 判 FAIL，且核心新颖性主张被否证——已加写作红线", page: 9, claimColor: T.risk });
  s.addShape(pptx.ShapeType.rect, {
    x: M.left, y: BAND.evidence, w: M.contentW, h: 1.62,
    fill: { color: "FDF1F0" }, line: { color: T.risk, width: 1 },
  });
  s.addText("判据原文", {
    x: M.left + 0.20, y: BAND.evidence + 0.12, w: 2.0, h: 0.26,
    fontFace: FONT_CN, fontSize: 11.5, bold: true, color: T.risk, margin: 0,
  });
  s.addText(
    "我们的 System-1 头在 TS0 / ArchiveII / PDB ts1 上 ECE 与 Brier 优于或持平 SPOT-RNA / UFold 的 sigmoid 概率。",
    {
      x: M.left + 0.20, y: BAND.evidence + 0.40, w: M.contentW - 0.40, h: 0.34,
      fontFace: FONT_CN, fontSize: 12, color: T.ink, margin: 0, wrap: true,
    });
  s.addText(
    "实测：对 UFold 成立（0.0031 vs 0.0147），但判据的本意是「免 DP 概率的校准由我们首次做到」——"
    + "RNAformer 实测 ECE 0.0015，比我们重标定后的 0.0031 还好一倍，且它不做任何后处理重标定。"
    + "只与 UFold 比是换了更弱的对手，不构成通过。",
    {
      x: M.left + 0.20, y: BAND.evidence + 0.76, w: M.contentW - 0.40, h: 0.74,
      fontFace: FONT_CN, fontSize: 12, color: T.ink, margin: 0, valign: "top", wrap: true,
    });

  s.addText("写作红线（新增，与原红线并列）", {
    x: M.left, y: 3.32, w: M.contentW, h: 0.26,
    fontFace: FONT_CN, fontSize: 12, bold: true, color: T.ink, margin: 0,
  });
  s.addText([
    { text: "不得出现", options: { bold: true, color: T.risk } },
    { text: "「首次给出经校准验证的免 DP 配对概率」。", options: { color: T.ink } },
    { text: "可以写", options: { bold: true, color: T.positive } },
    { text: "「首次系统评测 RNA 配对概率的校准」——覆盖 6 个来源、同 split 同候选对的表确实是领域空白。", options: { color: T.ink } },
  ], {
    x: M.left, y: 3.60, w: M.contentW, h: 0.54,
    fontFace: FONT_CN, fontSize: 12, margin: 0, valign: "top", wrap: true,
  });

  s.addText("仍然站得住的部分", {
    x: M.left, y: 4.26, w: M.contentW, h: 0.26,
    fontFace: FONT_CN, fontSize: 12, bold: true, color: T.positive, margin: 0,
  });
  bullets(s, [
    { mark: "C1-c", text: "免 DP 头经重标定后与「我们自己的」CRF 精确边际 gap = 0.0028（阈值 0.02），且 Brier/NLL 同时改善。这是自洽性结果。", markColor: T.positive },
    { mark: "评测贡献", text: "系统性校准评测确为空白；我们的精确边际 ECE 0.0004 是全场最低，并给出同一模型的三档一致校准谱。", markColor: T.positive },
    { mark: "结论调整", text: "C1 从「方法新颖性」降级为「评测 + 自洽性」；主对标改为 RNAformer（0.7578）。", markColor: T.caution },
  ], { y: 4.54, size: 11.5, gapY: 0.02 });
}

// S10 method decisions
{
  const s = pptx.addSlide();
  chrome(s, { label: "方法决策", claim: "参考语料的三条规则与教师问题：每条都有实测理由，不是默认值", page: 10 });
  table(s, {
    y: BAND.evidence,
    colW: [3.1, 3.0, 5.9],
    header: ["决策", "取值", "实测理由"],
    rowH: 0.42,
    rows: [
      ["假结处理", "pk 栏精确删除",
       "参考发布给每个配对打了假结等级标签。用贪心交叉删除会连带删掉被卷入交叉的无辜嵌套配对。"],
      ["配对类型", "删除非规范配对",
       "参考发布非规范配对占 9.1%–23.7%，我们既有语料是 0.000。头只能输出 AU/GC/GU，拿它无法表达的配对当 GT 没有意义。"],
      ["多联体（一个碱基多个配对）", "保留跨度最大者",
       "这条不是冗余保护：贪心交叉只捕捉「共享左端」，共享右端完全漏掉，而 pairs_to_dotbracket 会静默写出不平衡括号。"],
      ["教师模型", "扩为集成（ViennaRNA + EternaFold）",
       "EternaFold 二进制已在盘上，--posteriors 直出配对后验概率，集成不必新装任何软件。「为什么原选 ViennaRNA」≠「它是最好的教师」。"],
    ],
  });
  sourceNote(s, "「ViennaRNA 不是性能最高的、为什么当教师」这条质疑成立：它当初被选中的唯一理由是 C1-c 需要一个精确参照（McCaskill 配分函数，版本锁 2.7.2）。");
}

// S11 reproduction credibility
{
  const s = pptx.addSlide();
  chrome(s, { label: "复现可信度", claim: "基线的复现能力已被独立证据验证，不是「跑起来就算」", page: 11 });
  stats(s, [
    { value: "0.728", label: "RNAformer 已发表 TS0 F1\narXiv:2307.10073 表1，已核实", color: T.accent },
    { value: "0.7154", label: "我们的复现 · 发布口径\n取下界", color: T.positive },
    { value: "0.7578", label: "我们的复现 · 项目口径\n取上界，恰好夹住已发表值", color: T.positive },
    { value: "156/156", label: "UFold 权重重载\nstrict 全键匹配，0 缺失", color: T.positive },
  ], { y: BAND.evidence, valueSize: 26 });

  table(s, {
    y: 2.98,
    colW: [3.4, 2.4, 2.4, 3.4],
    header: ["split", "项目口径 micro F1", "发布口径 micro F1", "被移除配对"],
    rowH: 0.335,
    rows: [
      ["ref_bprna_ts0", "0.7578", "0.7154", "5,119 / 40,067（12.8%）"],
      ["ref_pdb_ts2", "0.8590", "0.7203", "249 / 904（27.5%）"],
      ["ref_pdb_ts3", "0.9410", "0.7826", "227 / 644（35.2%）"],
    ],
  });
  bullets(s, [
    { mark: "结论", text: "移除非规范/假结配对会显著抬高任何模型的 F1，假结越多的集抬得越多（ts3 上 +0.158）。所以 F1 必须与「删了多少对」同列报告。" },
    { mark: "记录", text: "UFold 自带 seq2dot 用逐位 argmax、不保证对称，34/1291 条序列括号不平衡；改为只保留互为最优的对后 F1 由 0.6584 → 0.6598。" },
  ], { y: 4.52, size: 12, gapY: 0.02 });
}

// S12 status
{
  const s = pptx.addSlide();
  chrome(s, { label: "当前状态", claim: "TR1 训练已在运行；其余队列与已到位的外部资产", page: 12 });
  table(s, {
    y: BAND.evidence,
    colW: [4.4, 2.6, 4.9],
    header: ["任务", "状态", "说明"],
    rowH: 0.335,
    flagCol: 1,
    flagMap: { "运行中": T.evidence, "已完成": T.positive, "可用": T.positive, "待做": T.caution },
    rows: [
      ["TR1 训练（4.29× 数据对照）", "运行中", "rinalmo_ff_tr1 @step 475/20000；rinalmo_len_tr1 @step 500/20000"],
      ["TR1 教师软标签", "已完成", "45,865 条，12 进程 892 s，version_lock = ViennaRNA 2.7.2"],
      ["benchmark 新增 7 个 split", "已完成", "含基线跑完；synthetic 两集已标记不得进主表"],
      ["UFold / RNAformer 适配器", "已完成", "dot-bracket + sigmoid 概率矩阵均已产出并通过校验"],
      ["EternaFold", "可用", "二进制在盘上，--posteriors 输出已验证"],
      ["mmseqs 真实 identity 去冗余", "待做", "此前记「工具未安装」，实际在盘上"],
      ["seed 补满 20,000 步（≥5 seed）", "待做", "当前只能报告单点，不能报 mean ± std"],
    ],
  });
  sourceNote(s, "GPU：8×A100-40GB 共享，GPU 6/7 为 MIG 切片（CUDA_VISIBLE_DEVICES=6/7 拿到的是 4.75 GiB 的 1g.5gb，已两次导致 OOM）。");
}

// S13 next steps
{
  const s = pptx.addSlide();
  chrome(s, { label: "下一步", claim: "三条可能缩小 0.162 差距的路径，以及缩不回来时的诚实退路", page: 13 });
  table(s, {
    y: BAND.evidence,
    colW: [3.0, 2.0, 1.4, 5.6],
    header: ["路径", "做法", "状态", "为什么可能有效"],
    rowH: 0.44,
    flagCol: 2,
    flagMap: { "运行中": T.evidence, "未试": T.caution },
    rows: [
      ["增加训练量", "TR1（4.29× 数据）同配置重训", "运行中", "已证实 3,500 → 20,000 步带来 +0.1003；数据量是同一杠杆"],
      ["改解码口径", "校准后再送势（sigmoid 之前先做单调重标定）", "未试", "§14.11 已证「送势 > 送概率」，但从未试过「先校准再送势」"],
      ["架构 / 容量", "rinalmo_big（4.8× 参数）臂", "运行中", "回答平铺头的平台是容量限制还是数据限制"],
    ],
  });

  s.addShape(pptx.ShapeType.rect, {
    x: M.left, y: 3.42, w: M.contentW, h: 1.28,
    fill: { color: "FDFAF3" }, line: { color: T.caution, width: 1 },
  });
  s.addText("如果三条都缩不回来", {
    x: M.left + 0.20, y: 3.52, w: M.contentW - 0.40, h: 0.26,
    fontFace: FONT_CN, fontSize: 12, bold: true, color: T.caution, margin: 0,
  });
  s.addText(
    "论文按「校准系统评测 + 诚实的负结果」来写，把「免 DP 头在准确率上落后已发表 SOTA 0.16」"
    + "作为主结果如实报告，而不是继续把 C1 包装成正面贡献。这个决定等 TR1 结果出来再定，不预先写结论。",
    {
      x: M.left + 0.20, y: 3.80, w: M.contentW - 0.40, h: 0.80,
      fontFace: FONT_CN, fontSize: 12, color: T.ink, margin: 0, valign: "top", wrap: true,
    });

  s.addText("不能省略的诚实声明", {
    x: M.left, y: 4.86, w: M.contentW, h: 0.26,
    fontFace: FONT_CN, fontSize: 12, bold: true, color: T.ink, margin: 0,
  });
  bullets(s, [
    { mark: "1", text: "全部 F1 为单 seed、单 checkpoint；≥5 seed 未完成前不得报告 mean ± std。", markColor: T.risk },
    { mark: "2", text: "UFold 的训练集与 TS0 的同源重叠仍未核实（§5.2），引用其数字时必须声明。", markColor: T.risk },
    { mark: "3", text: "跨家族短板最大：bprna_new 上我们 0.3536，UFold 0.6106，差距 0.257，是 TS0 的 4 倍。", markColor: T.risk },
  ], { y: 5.12, size: 11.5, gapY: 0.02 });
}

// ------------------------------------------------------------ structural gate
const FULLWIDTH = /[\u1100-\u115F\u2E80-\u303E\u3041-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uA000-\uA4CF\uAC00-\uD7A3\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFF60\uFFE0-\uFFE6]/;
function estHeight(rec) {
  if (!rec.w || !rec.h) return 0;
  const em = rec.fontSize / 72;
  const usable = rec.w - 0.18;                 // PptxGenJS default insets
  let lines = 0;
  for (const seg of rec.text.split("\n")) {
    if (!rec.wrap) { lines += 1; continue; }
    let width = 0;
    for (const ch of seg) {
      const cw = (FULLWIDTH.test(ch) ? 1 : 0.55) * em;
      if (width + cw > usable) { lines += 1; width = cw; } else { width += cw; }
    }
    lines += 1;
  }
  return (lines * rec.fontSize * 1.34) / 72;
}

const problems = [];
LAYOUT.forEach((r) => {
  const need = estHeight(r);
  const tag = r.text.slice(0, 42).replace(/\n/g, "⏎");
  if (r.h && need > r.h * 1.06 + 0.02) {
    problems.push(`overflow  need ${need.toFixed(2)}in  box h ${r.h}in  | ${tag}`);
  }
  if (r.w && r.x != null && r.x + r.w > PAGE.w - 0.30) {
    problems.push(`right     edge ${(r.x + r.w).toFixed(2)}in  | ${tag}`);
  }
  if (r.h && r.y != null && r.y + r.h > PAGE.h - 0.10) {
    problems.push(`bottom    edge ${(r.y + r.h).toFixed(2)}in  | ${tag}`);
  }
});
console.log(`[layout] ${LAYOUT.length} text boxes, ${problems.length} flagged`);
problems.forEach((p) => console.log(`   ! ${p}`));

const out = process.argv[2] || "RNA-JEPA_项目进展_20260924.pptx";
await pptx.writeFile({ fileName: out });
console.log("deck written ->", out);