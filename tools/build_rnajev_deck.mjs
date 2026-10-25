import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const PptxGenJS = require("pptxgenjs");
const ROOT = process.env.RNAJEV_DECK_ROOT || "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03";
const FIG = path.join(ROOT, "figures");
const OUT = process.argv[2] || path.join(ROOT, "RNA-JEV_论文逻辑重构_20261025.pptx");
const D = JSON.parse(fs.readFileSync(path.join(FIG, "source_data.json"), "utf8"));
const AB_JSON = path.join(FIG, "main_ablation_5x8.json");
const AB_FIG_A = path.join(FIG, "fig03a_ablation_matrix_3840x1920.png");
const AB_FIG_B = path.join(FIG, "fig03b_ood_signature_3840x1920.png");
const MRL_JSON = path.join(FIG, "mrl_suite_result.json");
const MATCHED_JSON = path.join(FIG, "matched_30k_statistics.json");
const HAS_AB = fs.existsSync(AB_JSON) && fs.existsSync(AB_FIG_A) && fs.existsSync(AB_FIG_B);
const HAS_MRL = fs.existsSync(MRL_JSON);
const HAS_MATCHED = fs.existsSync(MATCHED_JSON);
const HAS_ALL = HAS_AB && HAS_MRL && HAS_MATCHED;
if (!HAS_ALL && !OUT.includes("draft")) {
  throw new Error(`正式版证据门未闭合：ablation=${HAS_AB} mrl=${HAS_MRL} matched=${HAS_MATCHED}；只能生成 draft。`);
}

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "RNA-JEV";
pptx.company = "Zhejiang University";
pptx.subject = "去污染 benchmark、OOD 机制、校准与跨骨干迁移";
pptx.title = "RNA-JEV 论文逻辑版项目进展";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "微软雅黑",
  bodyFontFace: "微软雅黑",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const SLIDE_W = 13.333, SLIDE_H = 7.5;
const MARGIN = 0.52, CONTENT_X = MARGIN, CONTENT_W = SLIDE_W - 2 * MARGIN;
const FONT = "微软雅黑";
const T = {
  ink: "17243A", muted: "667085", faint: "98A2B3", blue: "0F4D92",
  blue2: "3775BA", teal: "0E7C86", green: "287A5B", red: "B64342",
  gold: "C98A22", surface: "F5F7FA", rule: "DDE3EA", white: "FFFFFF",
  dark: "101B31", paleBlue: "EAF1F8", paleTeal: "E9F5F4", paleRed: "FBEFEF",
};

function calculateScaledImageOpts(opts) {
  const { path: p, w: targetW, h: targetH, x = 0, y = 0, mode = "cover", ...rest } = opts;
  if (!p || !targetW || !targetH) return opts;
  return { path: p, x, y, w: targetW, h: targetH, sizing: { type: mode, w: targetW, h: targetH }, ...rest };
}
function createVirtualNode(type, data, parentX = 0, parentY = 0) {
  const opts = data.opts || {};
  const node = { type, data, absX: parentX + (opts.x || 0), absY: parentY + (opts.y || 0), w: opts.w || 0, h: opts.h || 0, children: [] };
  node.addShape = function (shapeType, o = {}) { const c = createVirtualNode("shape", { shapeType, opts: o }, node.absX, node.absY); node.children.push(c); return c; };
  node.addText = function (text, o = {}) { const c = createVirtualNode("text", { text, opts: { fit: "shrink", ...o } }, node.absX, node.absY); node.children.push(c); return c; };
  node.addImage = function (o = {}) { const c = createVirtualNode("image", { opts: calculateScaledImageOpts(o) }, node.absX, node.absY); node.children.push(c); return c; };
  node.addTable = function (tableData, o = {}) { const c = createVirtualNode("table", { tableData, opts: o }, node.absX, node.absY); node.children.push(c); return c; };
  return node;
}
const LAYOUT = [];
let CURRENT_SLIDE = 0;
function flattenNode(node, realSlide) {
  const o = { ...node.data.opts, x: node.absX, y: node.absY };
  if (node.type === "text") {
    const plain = Array.isArray(node.data.text) ? node.data.text.map(r => typeof r === "object" ? r.text : r).join("") : String(node.data.text ?? "");
    LAYOUT.push({ slide: CURRENT_SLIDE, text: plain, ...o });
    realSlide.addText(node.data.text, o);
  } else if (node.type === "shape") realSlide.addShape(node.data.shapeType, o);
  else if (node.type === "image") realSlide.addImage(o);
  else if (node.type === "table") realSlide.addTable(node.data.tableData, o);
  node.children.forEach(c => flattenNode(c, realSlide));
}
const originalAddSlide = pptx.addSlide.bind(pptx);
pptx.addSlide = function (options) {
  const realSlide = originalAddSlide(options);
  CURRENT_SLIDE += 1;
  return {
    children: [], _realSlide: realSlide,
    set background(v) { realSlide.background = v; }, get background() { return realSlide.background; },
    addShape(t, o = {}) { const n = createVirtualNode("shape", { shapeType: t, opts: o }); this.children.push(n); return n; },
    addText(t, o = {}) { const n = createVirtualNode("text", { text: t, opts: { fit: "shrink", ...o } }); this.children.push(n); return n; },
    addImage(o = {}) { const n = createVirtualNode("image", { opts: calculateScaledImageOpts(o) }); this.children.push(n); return n; },
    addTable(t, o = {}) { const n = createVirtualNode("table", { tableData: t, opts: o }); this.children.push(n); return n; },
    addChart(t, d, o) { realSlide.addChart(t, d, o); },
    render() { this.children.forEach(n => flattenNode(n, realSlide)); },
  };
};

// 全文统一微软雅黑与 1.5 倍行距；表格单元格也走同一规则。
let spacingInstalled = false;
function installSpacing(sample) {
  if (spacingInstalled) return;
  spacingInstalled = true;
  const proto = Object.getPrototypeOf(sample);
  const addText = proto.addText;
  proto.addText = function (text, opts = {}) {
    const base = opts.fontSize || 12;
    if (opts.lineSpacing === undefined) opts.lineSpacing = Math.round(base * 1.5 * 100) / 100;
    return addText.call(this, text, opts);
  };
  const addTable = proto.addTable;
  proto.addTable = function (rows, opts = {}) {
    rows.forEach(row => Array.isArray(row) && row.forEach(cell => {
      if (cell && typeof cell === "object" && cell.options) {
        const fs = cell.options.fontSize || opts.fontSize || 10;
        if (cell.options.lineSpacing === undefined) cell.options.lineSpacing = Math.round(fs * 1.5 * 100) / 100;
      }
    }));
    return addTable.call(this, rows, opts);
  };
}
const firstSlide = pptx.addSlide();
installSpacing(firstSlide._realSlide);

function bounds() {
  for (const r of LAYOUT) {
    if (r.x < 0 || r.y < 0 || r.x + r.w > SLIDE_W + 0.001 || r.y + r.h > SLIDE_H + 0.001) {
      throw new Error(`越界 slide=${r.slide} ${r.text.slice(0, 36)} (${r.x},${r.y},${r.w},${r.h})`);
    }
  }
}
function notes(slide, text) { if (slide._realSlide.addNotes) slide._realSlide.addNotes(text); }
function token(slide, value, opts) {
  const fontSize = opts.fontSize || 12;
  const lines = String(value).split(/\r?\n/);
  const lineSpacing = opts.lineSpacing || fontSize * 1.5;
  const lineHeight = opts.h || (lineSpacing / 72 + 0.03);
  lines.forEach((lineText, index) => {
    slide.addText(lineText, { ...opts, fontFace: FONT, fontSize, y: (opts.y || 0) + index * lineHeight,
      h: lineHeight, margin: opts.margin ?? 0, wrap: false, fit: "shrink", breakLine: false });
  });
}
function rect(slide, x, y, w, h, fill = T.surface, line = T.rule, radius = true) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, { x, y, w, h, rectRadius: radius ? 0.08 : 0, fill: { color: fill }, line: { color: line, width: 0.7 } });
}
function text(slide, value, x, y, w, h, size = 12, color = T.ink, opts = {}) {
  slide.addText(value, { x, y, w, h, fontFace: FONT, fontSize: size, color, margin: 0, breakLine: false, fit: "shrink", valign: "top", ...opts });
}
function line(slide, x, y, w, h = 0, color = T.rule, width = 1, arrow = false) {
  slide.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, width, endArrowType: arrow ? "triangle" : "none" } });
}
function chrome(slide, section, title, subtitle, page, source = "") {
  text(slide, section.toUpperCase(), MARGIN, 0.24, CONTENT_W, 0.26, 10.5, T.teal, { bold: true, charSpacing: 1.1 });
  text(slide, title, MARGIN, 0.57, CONTENT_W, 0.55, 20.5, T.ink, { bold: true, valign: "middle" });
  if (subtitle) text(slide, subtitle, MARGIN, 1.12, CONTENT_W, 0.30, 10.5, T.muted);
  line(slide, MARGIN, 7.08, CONTENT_W, 0, T.rule, 0.7);
  if (source) text(slide, source, MARGIN, 7.16, CONTENT_W - 0.7, 0.18, 8, T.faint);
  token(slide, String(page).padStart(2, "0"), { x: 12.18, y: 7.13, w: 0.62, h: 0.23, fontSize: 9.5, bold: true, color: T.blue, align: "right" });
}
function figureSlide(page, section, title, subtitle, filename, caption, source) {
  const s = pptx.addSlide();
  chrome(s, section, title, subtitle, page, source);
  const img = path.join(FIG, filename);
  if (!fs.existsSync(img)) throw new Error(`缺图: ${img}`);
  s.addImage({ path: img, x: 1.3665, y: 1.48, w: 10.6, h: 5.30 });
  rect(s, MARGIN, 6.79, CONTENT_W, 0.24, T.surface, T.surface, false);
  text(s, caption, MARGIN + 0.10, 6.82, CONTENT_W - 0.20, 0.20, 9.4, T.ink);
  notes(s, `${title}\n${caption}\n来源: ${source}`);
  s.render();
}
function table(slide, rows, { x, y, w, colW, rowH = 0.35, fontSize = 9, headerSize = fontSize, headerFill = T.blue }) {
  const head = rows[0].map(v => ({ text: String(v), options: { fontFace: FONT, fontSize: headerSize, bold: true, color: T.white, fill: { color: headerFill }, margin: [0.03, 0.06, 0.03, 0.06], valign: "middle" } }));
  const body = rows.slice(1).map((row, ri) => row.map((v, ci) => {
    const obj = v && typeof v === "object" ? v : { text: v };
    return { text: String(obj.text ?? ""), options: { fontFace: FONT, fontSize, color: obj.color || T.ink, bold: !!obj.bold, fill: { color: ri % 2 ? T.surface : T.white }, margin: [0.025, 0.055, 0.025, 0.055], valign: "middle", align: ci === 0 ? "left" : "center" } };
  }));
  const total = colW.reduce((a, b) => a + b, 0);
  slide.addTable([head, ...body], { x, y, w, colW: colW.map(v => v / total * w), rowH, border: { type: "solid", color: T.rule, pt: 0.45 }, autoPage: false, valign: "middle" });
}

// 01 封面
{
  const s = firstSlide;
  s.background = { color: T.dark };
  text(s, "RNA-JEV / RESEARCH DECK", 0.68, 0.56, 7.2, 0.35, 12, "73D4D0", { bold: true, charSpacing: 1.8 });
  text(s, "结构决策头", 0.68, 1.55, 7.1, 0.70, 34, T.white, { bold: true });
  text(s, "去污染 benchmark、OOD 机制与跨骨干迁移", 0.68, 2.36, 7.25, 1.10, 23, T.white, { bold: true });
  text(s, "论文逻辑版项目进展｜从评测卫生到机制与边界", 0.68, 3.74, 6.8, 0.44, 13, "C4CEDD");
  rect(s, 0.68, 4.62, 6.70, 1.22, "182944", "182944");
  text(s, "核心原则", 0.92, 4.83, 1.5, 0.28, 11, "73D4D0", { bold: true });
  text(s, "模型身份、数据划分、评分器与硬件协议必须同时匹配；不重复展示同一证据。", 0.92, 5.18, 6.12, 0.46, 12, T.white);
  text(s, "2026-10-25 · Round 32", 0.68, 6.66, 4.5, 0.30, 11, "AAB6C8");
  // 原生 L×L 决策矩阵，呼应主题而非装饰性图片。
  const x0 = 8.65, y0 = 1.46, cell = 0.39;
  rect(s, 8.28, 1.08, 4.25, 4.55, "162740", "2B3C56");
  text(s, "L × L DECISION MAP", 8.62, 1.25, 3.45, 0.30, 10.5, "73D4D0", { bold: true, charSpacing: 1.1 });
  const pairs = new Set(["0,8", "1,6", "2,4", "3,5", "4,2", "5,3", "6,1", "8,0"]);
  for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
    const key = `${i},${j}`;
    const fill = pairs.has(key) ? (i + j === 8 ? "0E7C86" : "3775BA") : (i === j ? "33445C" : "1D2E47");
    s.addShape(pptx.ShapeType.rect, { x: x0 + j * cell, y: y0 + i * cell, w: 0.30, h: 0.30, fill: { color: fill }, line: { color: "162740", width: 0.3 } });
  }
  text(s, "frozen representation → 2D scorer → legal structure + confidence", 8.50, 5.20, 3.75, 0.28, 9.5, "C4CEDD", { align: "center" });
  s.render();
}

// 02 目录与论文逻辑
{
  const s = pptx.addSlide();
  chrome(s, "论文行文", "从评测卫生到机制、迁移与边界", "五条创新线共用同一条证据链；每一图只回答一个问题。", 2, "结构：主文 10 图 + 4 个附录");
  const steps = [
    ["01", "评测卫生与主结果", "泄露审计 → 去污染主 benchmark", "03–04"],
    ["02", "效率与机制", "速度 → OOD 消融 → ECE → 精度-速度", "05–09"],
    ["03", "架构拆解", "冻结表征、二维决策、结构先验解码", "10"],
    ["04", "迁移与外部验证", "NucleicBERT → RiNALMo 下游 → CHANRG → 长度", "11–14"],
    ["05", "结论与复现", "贡献边界、全板、协议、演进与当前队列", "15–20"],
  ];
  steps.forEach((r, i) => {
    const y = 1.58 + i * 1.03;
    rect(s, 0.58, y, 7.35, 0.84, i === 0 ? T.paleBlue : T.surface, T.rule);
    token(s, r[0], { x: 0.78, y: y + 0.17, w: 0.55, h: 0.36, fontSize: 18, bold: true, color: T.blue });
    text(s, r[1], 1.52, y + 0.11, 4.65, 0.28, 12.5, T.ink, { bold: true });
    text(s, r[2], 1.52, y + 0.45, 5.55, 0.23, 9.5, T.muted);
    token(s, r[3], { x: 6.93, y: y + 0.25, w: 0.65, h: 0.25, fontSize: 9.5, bold: true, color: T.teal, align: "right" });
  });
  text(s, "五条创新线", 8.35, 1.50, 4.35, 0.35, 13, T.blue, { bold: true });
  const innovations = [
    ["主下游 benchmark", "统一单模型与集成模型的参照合同"],
    ["推理速度 benchmark", "端到端吞吐，明确模型与硬件"],
    ["OOD 表现", "主 split 消融与 CHANRG 外部验证"],
    ["ECE 指标", "校准误差与结构质量排序并列"],
    ["兼容与迁移", "NucleicBERT 和 RiNALMo 下游任务"],
  ];
  innovations.forEach((r, i) => {
    const y = 1.98 + i * 0.94;
    rect(s, 8.35, y, 4.35, 0.76, i === 4 ? T.paleTeal : T.surface, T.rule);
    text(s, r[0], 8.56, y + 0.10, 3.88, 0.24, 11, T.ink, { bold: true });
    text(s, r[1], 8.56, y + 0.40, 3.88, 0.22, 8.7, T.muted);
  });
  s.render();
}

// 03 泄露审计
{
  const s = pptx.addSlide();
  chrome(s, "01 / 评测卫生", "泄露审计：先证明比较对象干净，再讨论 SOTA", "保留原表；0 表示规范化序列精确重叠，不等同于同源或预训练层面的绝对保证。", 3, "来源：overlap audit、leak_full_audit、clean-181 head-to-head");
  const rows = [
    ["模型（上板口径）", "训练语料", "TS0", "new", "TS1", "TS2", "TS3", "hard", "TSB", "判定"],
    ["RNAformer-bprna", "ref_tr_bprna (34.9k)", "0", "0", "38/63", "28/39", "15/19", "20/28", "339/428", "TS1-3/hard/TSB 污染"],
    ["RNAformer-interfam", "ref_tr_inter (2.2k)", "0", "0", "0", "0", "0", "0", "1/428", "主判定参照（TSB 除外）"],
    ["RiNALMo-ft 等 4 家", "bprna_tr0 (10.7k)", "0", "0", "0", "0", "0", "0", "247/428", "仅 TSB 污染"],
    ["eFold（零样本）", "bpRNA子集+ArchII留出", "0", "0", "0", "0", "0", "0", "247/428*", "TSB 下界命中"],
    ["xens2（我们）", "bprna_tr1c (42.6k)", "0", "0", "0", "0", "0", "0", "0/428", "结构监督语料全板零重叠"],
  ];
  table(s, rows, { x: MARGIN, y: 1.53, w: CONTENT_W, colW: [2.1, 2.0, 0.55, 0.55, 0.60, 0.60, 0.60, 0.60, 0.75, 2.15], rowH: 0.47, fontSize: 8.2, headerSize: 8.2 });
  const actions = [
    ["01 撤回", "撤回 TR1/TS0 与 TR0/TestSetB 的污染读数，不沿用虚高结果。"],
    ["02 去重", "以 blocklist 和规范化序列精确去重，独立第二遍审计。"],
    ["03 冻结", "固定 split、GT、评分器和 SHA256；参照 checkpoint 逐 split 选择。"],
    ["04 复评", "TestSetB 改用 clean-181：xens2 0.8370 vs RiNALMo-ft 0.8177。"],
  ];
  actions.forEach((r, i) => {
    const x = MARGIN + i * 3.08;
    rect(s, x, 4.66, 2.88, 1.34, i === 3 ? T.paleTeal : T.surface, T.rule);
    text(s, r[0], x + 0.16, 4.82, 2.55, 0.27, 11, i === 3 ? T.teal : T.blue, { bold: true });
    text(s, r[1], x + 0.16, 5.17, 2.55, 0.63, 9.1, T.ink);
  });
  text(s, "与后文关系：只有通过这页审计的参照，才进入下一页的统一主 benchmark。", MARGIN, 6.32, CONTENT_W, 0.28, 10.5, T.blue, { bold: true });
  s.render();
}

// 04–05 主结果与速度
figureSlide(4, "02 / 主 benchmark", "主图 1｜统一模型口径：最佳单 checkpoint 4/8，xens2 为 6/8", "预注册去污染参照；同一 split、同一评分器、同一模型身份。", "fig01_main_benchmark_3840x1920.png", "为何不是旧 7/8：bpRNA-new 的 xens2=0.5532；0.6132 属于另一组 r2d_tr1 集成。全板干净敏感性 xens2=5/8。", "主表与 metrics matrix；模型身份第二轮复核");
figureSlide(5, "03 / 推理速度", "主图 2｜r2d 单模型 124.78 seq/s，F1 逐位不变", "RTX 4090、TS0、端到端含 frozen encoder + 9.3M head + C decode。", "fig02_speed_benchmark_3840x1920.png", "124.78 属于 r2dtr1c-s0（F1=0.667971），不与 xens2 的 0.8039 拼接；A100 与 forward-only 参照分别标注。", "台账 §15.79–15.82；speed_v3/v4 JSON");

// 06 消融：正式版必须有完整矩阵，draft 仅展示执行状态。
if (HAS_AB) {
  figureSlide(6, "04 / OOD 机制", "主图 3a｜五类消融 × 8 split：全矩阵性能变化", "控制模型、评分协议和缺失格全部一致；不以单一 OOD 均值掩盖任务差异。", "fig03a_ablation_matrix_3840x1920.png", "每格均为消融−控制的 strict micro-F1 变化；五类消融与八 split 全部实测。", "main_ablation_5x8.json；40/40 + 控制 8/8 原始结果");
  figureSlide(7, "04 / OOD 机制", "主图 3b｜核心三模块：ID、外部集与 PDB-OOD 效应分解", "DP、Turner 和 2D scorer 单独对照；保留原第 37 页的统一签名读法。", "fig03b_ood_signature_3840x1920.png", "ID=TS0/ArchiveII-clean；外部集=bpRNA-new/TestSetB；PDB-OOD=TS1/TS2/TS3/TS-hard。旧 DP OOD=0.43 代理值已撤回。", "main_ablation_5x8.json；核心三模块的分组均值与 PDB/ID 比值");
} else {
  const s = pptx.addSlide();
  chrome(s, "04 / OOD 机制", "主图 3｜五类消融 × 8 split：等待完整实测矩阵", "DRAFT 状态页，不是科学结论；正式版生成器会在缺格时拒绝输出。", 6, "T-A128 / complete_main_ablations.sh");
  const rows = [
    ["消融", "历史已有", "当前缺口", "闭环方式"],
    ["DP → greedy", "TS0", "其余 7 split", "同一 r2d 权重全 8 split 直接评测"],
    ["Turner = 0", "6/8", "ArchII / TestSetB", "同一 r2d 权重、prior_weight=0"],
    ["2D scorer → pair MLP", "6/8", "ArchII / TestSetB", "复用 fftr1c 实体权重"],
    ["No distillation", "6/8", "ArchII / TestSetB", "复用 nodistill 实体权重"],
    ["No RLCD", "6/8（历史）", "统一新 ckpt 的 8/8", "唯一重训臂；VL0 early-stopping"],
  ];
  table(s, rows, { x: MARGIN, y: 1.60, w: CONTENT_W, colW: [2.6, 1.5, 2.4, 4.6], rowH: 0.58, fontSize: 10 });
  rect(s, MARGIN, 5.48, CONTENT_W, 0.92, T.paleRed, T.paleRed);
  text(s, "禁止放行：缺一格即不出主图；不得复用旧 DP OOD=0.43，也不得把 smoke/proxy 当成最终结果。", MARGIN + 0.20, 5.73, CONTENT_W - 0.40, 0.38, 11, T.red, { bold: true });
  s.render();
  const s2 = pptx.addSlide();
  chrome(s2, "04 / OOD 机制", "主图 3b｜核心三模块：等待 ID/OOD 分组统计", "DRAFT 状态页，不是科学结论；完整 5×8 矩阵验收后自动生成。", 7, "T-A128 / ID、External、PDB-OOD 分组");
  const groups = [["分组", "包含 split", "分析用途"], ["ID", "TS0、ArchiveII-clean", "近似分布内保持与通用结构能力"], ["External", "bpRNA-new、TestSetB", "外部序列集合的泛化"], ["PDB-OOD", "TS1、TS2、TS3、TS-hard", "实验结构 PDB 分布外泛化"]];
  table(s2, groups, { x: MARGIN, y: 1.68, w: CONTENT_W, colW: [1.4, 4.3, 5.5], rowH: 0.64, fontSize: 10 });
  rect(s2, MARGIN, 4.85, CONTENT_W, 1.10, T.surface, T.rule);
  text(s2, "核心三模块：DP → greedy、Turner prior = 0、2D scorer → pair MLP。\n只有矩阵完整后才计算组均值与 PDB/ID 比值，不预先填写效应方向。", MARGIN + 0.22, 5.10, CONTENT_W - 0.44, 0.62, 11, T.ink);
  s2.render();
}

// 07–08 校准与精度-速度
figureSlide(8, "05 / 校准与置信度", "主图 4｜大样本 split 校准改善，小 PDB split 不作过度承诺", "VL0 拟合两参数 logit 仿射，测试 split 不重拟合；ECE 与排序能力并列。", "fig04_ece_confidence_3840x1920.png", "TS0/TestSetB/bpRNA-new/ArchiveII 均改善；TS1 与 TS-hard 小幅改善，TS2/TS3 过校正，必须保留 n 与限制。", "台账 §15.80；structure confidence board");
figureSlide(9, "06 / 精度-速度", "主图 5｜真实帕累托集：A100 上是 RNAformer 与 UFold", "同节点、同 split、strict micro-F1；r2d 在 A100 被 RNAformer 同时以更高 F1 和吞吐支配。", "fig05_accuracy_speed_pareto_3840x1920.png", "4090 的 r2d=124.78 seq/s 仅作硬件更新，不混入 A100 前沿；不虚构 xens2 的吞吐或联合最优点。", "A100 baseline board + §15.79；Pareto 程序逐点判定");

// 09 原生三层架构图
{
  const s = pptx.addSlide();
  chrome(s, "07 / 架构拆解", "主图 6｜三层架构：冻结表征、二维决策、结构先验解码", "每个模块都标出输入、输出、可训练参数和验证路径；不是概念拼贴。", 10, "decision_head.py / harness.py / evaluate_decision.py");
  const cols = [0.55, 4.74, 8.93];
  const headers = ["A  冻结序列表征", "B  二维配对决策", "C  结构先验与输出"];
  cols.forEach((x, i) => {
    rect(s, x, 1.55, 3.85, 4.96, i === 0 ? T.paleBlue : (i === 1 ? T.surface : T.paleTeal), T.rule);
    text(s, headers[i], x + 0.20, 1.75, 3.45, 0.34, 13, i === 2 ? T.teal : T.blue, { bold: true });
  });
  // A：输入与冻结骨干
  rect(s, 0.84, 2.35, 3.27, 0.74, T.white, T.rule);
  text(s, "RNA sequence", 1.02, 2.51, 2.90, 0.28, 13, T.ink, { bold: true, align: "center" });
  line(s, 2.47, 3.10, 0, 0.28, T.muted, 1.2, true);
  rect(s, 0.84, 3.39, 3.27, 1.06, T.blue, T.blue);
  text(s, "RiNALMo-giga · 650M", 1.02, 3.55, 2.90, 0.28, 13, T.white, { bold: true, align: "center" });
  text(s, "frozen backbone · 0 trainable params", 1.02, 3.94, 2.90, 0.24, 9.3, T.white, { align: "center" });
  line(s, 2.47, 4.46, 0, 0.28, T.muted, 1.2, true);
  rect(s, 0.84, 4.75, 3.27, 0.74, T.white, T.rule);
  text(s, "embeddings  L × 1280", 1.02, 4.92, 2.90, 0.28, 12, T.ink, { bold: true, align: "center" });
  text(s, "下游结构训练只用 tr1c；骨干不在 benchmark 上微调。", 0.84, 5.78, 3.27, 0.48, 9.2, T.muted, { align: "center" });
  // B：配对表示与 2D 头
  rect(s, 5.03, 2.35, 3.27, 0.94, T.white, T.rule);
  text(s, "hᵢ+hⱼ  |  hᵢ⊙hⱼ  |  |hᵢ−hⱼ|", 5.18, 2.50, 2.97, 0.28, 11.3, T.ink, { bold: true, align: "center" });
  text(s, "permutation-invariant pair features", 5.18, 2.88, 2.97, 0.20, 8.7, T.muted, { align: "center" });
  line(s, 6.66, 3.30, 0, 0.28, T.muted, 1.2, true);
  rect(s, 5.03, 3.59, 3.27, 1.06, T.blue2, T.blue2);
  text(s, "ResNet2D scorer · 9.3M", 5.18, 3.75, 2.97, 0.28, 12.5, T.white, { bold: true, align: "center" });
  text(s, "symmetric L × L pair logits · one forward", 5.18, 4.15, 2.97, 0.24, 8.9, T.white, { align: "center" });
  rect(s, 5.03, 4.96, 3.27, 0.82, T.white, T.rule);
  text(s, "NLL + distillation + RLCD", 5.18, 5.12, 2.97, 0.27, 11, T.ink, { bold: true, align: "center" });
  text(s, "训练目标；消融逐项关闭", 5.18, 5.47, 2.97, 0.20, 8.7, T.muted, { align: "center" });
  // C：双输出与先验栈
  rect(s, 9.22, 2.35, 3.27, 0.84, T.white, T.rule);
  text(s, "pair logits + Turner prior", 9.38, 2.53, 2.95, 0.28, 11.5, T.ink, { bold: true, align: "center" });
  line(s, 10.85, 3.20, 0, 0.28, T.muted, 1.2, true);
  rect(s, 9.22, 3.49, 3.27, 0.94, T.teal, T.teal);
  text(s, "non-crossing max-product DP", 9.38, 3.64, 2.95, 0.28, 11.3, T.white, { bold: true, align: "center" });
  text(s, "C-batched exact decode · legal by construction", 9.38, 4.03, 2.95, 0.22, 8.6, T.white, { align: "center" });
  rect(s, 9.22, 4.72, 3.27, 0.94, T.white, T.rule);
  text(s, "sigmoid → VL0 affine → SC / ECE", 9.38, 4.88, 2.95, 0.28, 10.8, T.ink, { bold: true, align: "center" });
  text(s, "dot-bracket structure + per-structure confidence", 9.38, 5.27, 2.95, 0.22, 8.6, T.muted, { align: "center" });
  text(s, "先验注入位置可审计；DP 仍为 O(L³)，只免配分函数。", 9.22, 5.91, 3.27, 0.36, 9.0, T.muted, { align: "center" });
  // 跨列数据流箭头
  line(s, 4.40, 3.92, 0.34, 0, T.blue, 1.6, true);
  line(s, 8.59, 3.92, 0.34, 0, T.teal, 1.6, true);
  text(s, "与后文关系：迁移实验复用同一决策头与先验栈，只替换骨干或下游头。", MARGIN, 6.70, CONTENT_W, 0.26, 10.2, T.blue, { bold: true });
  s.render();
}

// 10–13 迁移与外部验证
figureSlide(11, "08 / 跨骨干迁移", "主图 7｜Jev-on-NucleicBERT：8/8 split 均正增益", "Δ +0.055 至 +0.234，平均 +0.134；OOD/家族 split 增益更大。", "fig07_nucleicbert_transfer_3840x1920.png", "回答兼容性问题：同一 Jev 决策头不依赖 RiNALMo 特有表征；ArchII 使用匹配的 trunc598 口径。", "ROUND26 / T-A99；NB-ft 与 Jev-on-NB 8-split");
const NEW_POSITIVE = HAS_MRL ? Object.entries(D.downstream.additional_mrl || {}).filter(([, r]) => r.comparison.significant_positive).map(([name]) => name) : [];
const DOWNSTREAM_NOTE = HAS_MRL ? (NEW_POSITIVE.length ? `新增显著正结果：${NEW_POSITIVE.join("、")}；其余数据留档。` : "新增三组 MRL 已完成但未达显著门，全部数据留档。") : "新增任务必须完成预注册与统计后再加入。";
figureSlide(12, "09 / RiNALMo 下游", "主图 8｜下游头迁移：ncRNA 三指标提升，回归任务小幅为正", "ncRNA 保留 accuracy / macro-F1 / balanced accuracy；MRL 与 TE-EL 保留 PCC。", "fig08_rinalmo_downstream_3840x1920.png", `合并原第 25、28a、36 页信息；splice 数据留档不进主文。${DOWNSTREAM_NOTE}`, "ROUND26 / ROUND28 / T-A129；frozen backbone 双头对照");
figureSlide(13, "10 / 外部 OOD", "主图 9｜CHANRG：重点看 OOD mean，而不是回避 GenF", "仅保留 RiNALMo-Giga ft、OFoldNA-SS、xens2、r2d 四个系统；其余数据留档。", "fig09_chanrg_ood_3840x1920.png", "xens2 OOD mean 0.4434 vs OFoldNA-SS 0.4410，仅写并列；GenF=0.3066 是弱项，不用均值掩盖。", "OFoldNA Table 3 同口径；ROUND26 CHANRG board");
figureSlide(14, "11 / 长度边界", "主图 10｜短中段领先，>400 nt 仍是开放问题", "TS0 与 ArchiveII-clean 双面板；桶样本量和 strict micro-F1 全部保留。", "fig10_length_generalization_3840x1920.png", "长度分层用于界定部署边界，不把 retention 写成绝对长链优势；重复的原第 35 页已删除。", "ROUND29 length-bucket result.json 重算");

// 14 结论与边界
{
  const s = pptx.addSlide();
  chrome(s, "12 / 结论", "五条创新线成立，但每条都必须带边界", "结论不再靠重复图表强化；用 claim–evidence–limitation 对齐。", 15, "结论口径：Round 32 统一复核");
  const rows = [
    ["创新线", "可支持的结论", "必须同时展示的限制"],
    ["主 benchmark", "统一 xens2 对预注册 clean 参照为 6/8；最佳单 checkpoint 4/8", "旧 7/8 是混合模型组合；全板干净敏感性 xens2=5/8；TS2 未胜"],
    ["推理速度", "r2d 在 4090 达 124.78 seq/s，批大小改变不改预测", "A100 r2d 不在 Pareto 前沿；不把该速度与 xens2 F1 拼成联合点"],
    ["OOD", "先验栈消融 + CHANRG 外部 OOD 共同支撑鲁棒性", "GenF 弱、TS2 未胜；xens2 与 OFoldNA-SS 的 OOD mean 仅并列"],
    ["ECE", "四个大样本 split 均改善，结构置信度排序全正", "TS2/TS3 过校正；小样本噪声与族敏感性不可隐藏"],
    ["兼容与迁移", "NucleicBERT 8/8 正增益；ncRNA 三指标提升", "MRL/TE-EL 增益较小；新任务未完成前不得扩写为普遍优势"],
  ];
  table(s, rows, { x: MARGIN, y: 1.55, w: CONTENT_W, colW: [1.75, 5.0, 5.1], rowH: 0.74, fontSize: 9.2, headerSize: 9.5 });
  rect(s, MARGIN, 6.18, CONTENT_W, 0.54, T.paleBlue, T.paleBlue);
  text(s, "论文闭环：干净比较 → 精度与速度 → OOD 机制 → 置信度 → 跨骨干与部署边界。", MARGIN + 0.18, 6.34, CONTENT_W - 0.36, 0.26, 11, T.blue, { bold: true, align: "center" });
  s.render();
}

// 15 附录 A：完整基线，修正模型身份但保留全量数据。
{
  const s = pptx.addSlide();
  chrome(s, "附录 A / 全板", "全板敏感性：xens2=5/8，最佳单 checkpoint=4/8", "完整保留基线；统一 xens2 与历史 r2d_tr1 集成单列，避免旧表混用。", 16, "全板含 Vienna / Kirigami；TS3/hard 采用 clean inter-fam 参照");
  const rows = [
    ["系统", "TS0", "new", "TS1", "TS2", "TS3", "hard", "ArchII", "TSB", "源"],
    ["ViennaRNA mfe", "0.5056", "0.6379", "0.7209", "0.8980", "0.8206", "0.7663", "0.5752", "0.5225", "CPU"],
    ["ViennaRNA centroid", "0.5393", "0.6770", "0.7299", "0.8930", "0.8235", "0.7958", "0.6181", "0.5482", "CPU"],
    ["ViennaRNA mea", "0.5241", "0.6620", "0.7192", "0.8938", "0.8237", "0.7864", "0.6054", "0.5393", "CPU"],
    ["Nussinov+Turner", "0.2126", "0.3015", "0.3730", "0.4187", "0.4131", "0.4067", "0.2000", "0.2164", "CPU"],
    ["MXfold2", "0.5651", "0.6688", "0.7967", "0.8991", "0.8127", "0.8028", "0.7400", "0.6504", "GPU"],
    ["EternaFold", "0.5400", "0.6854", "0.7426", "0.8635", "0.8005", "0.7658", "0.6157", "0.5850", "CPU"],
    ["UFold", "0.6598", "0.6106", "0.6455", "0.8792", "0.6326", "0.6388", "0.4111", "0.5858", "GPU"],
    ["RNAformer (bprna)", "0.7578", "0.4936", "0.7658", "0.8590", "0.9410‡", "0.7845‡", "—", "—", "GPU"],
    ["RNAformer (inter-fam)", "—", "—", "0.8150", "0.9043", "0.8245", "0.7709", "—", "—", "GPU"],
    ["RNAformer (biophys)", "—", "—", "—", "—", "0.8265", "—", "—", "—", "GPU"],
    ["RiNALMo-ft (Zenodo)", "0.7214", "0.4489", "0.7139", "0.6702", "0.6055", "0.5556", "0.7613", "0.8177†", "GPU"],
    ["NucleicBERT frozen+官方头", "0.1509", "0.0955", "0.2029", "0.1541", "0.1439", "0.1343", "0.2372", "0.1210", "GPU"],
    ["NucleicBERT fine-tuned（实测）", "0.4609", "0.3453", "0.6049", "0.6435", "0.5175", "0.4845", "0.4735", "0.4862", "GPU"],
    ["eFold (Sci Adv 26)", "0.5612", "0.6640", "0.7682", "0.8862", "0.7354", "0.7632", "0.6618", "0.5661†", "GPU"],
    ["DEPfold (ICLR 25)", "0.4430", "0.1736", "0.3735", "0.3270", "0.2670", "0.2409", "0.6216", "0.5835†", "GPU"],
    ["RFold (ICML 24, TR0 重训)", "0.5520", "0.5313", "0.6578", "0.7002", "0.6495", "0.6220", "0.6811", "0.5740†", "GPU"],
    ["Kirigami (arXiv 2406)", "0.7386", "0.5800", "0.7366", "0.7061", "0.7436", "0.6782", "0.7872", "0.7514†", "GPU"],
    ["xens2（统一板协议）", "0.8039", "0.5532", "0.8755", "0.8604", "0.8965", "0.8732", "0.7760", "0.8370†", "GPU"],
    ["r2d_tr1 2-seed（历史集成）", "0.6733", "0.6132", "—", "—", "—", "—", "0.7664", "—", "GPU"],
  ];
  table(s, rows, { x: MARGIN, y: 1.48, w: CONTENT_W, colW: [2.65, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.48], rowH: 0.238, fontSize: 7.0, headerSize: 7.2 });
  text(s, "第 7 页问题的准确回答：最佳单 checkpoint 4/8 成立；但原页混合全集、clean 子集与多套参照，不能直接据此写 xens2 7/8。†=clean-181；‡=不用于 clean 判定的旧参照。", MARGIN, 6.42, CONTENT_W, 0.36, 8.6, T.muted);
  s.render();
}

// 16 附录 B：速度协议
{
  const s = pptx.addSlide();
  chrome(s, "附录 B / 速度协议", "模型、硬件、计时范围必须同时写", "速度测试串行执行；不把跨硬件或 forward-only 读数写成同口径。", 17, "r2dtr1c-s0 / TS0 n=1288 / strict micro-F1");
  const rows = [["配置", "seq/s", "F1", "协议说明"]];
  D.speed.ladder.forEach(p => rows.push([p.label, p.throughput.toFixed(2), D.speed.f1.toFixed(6), p.label === "B=256" ? "终值；峰值" : (["B=384", "B=512"].includes(p.label) ? "饱和后复测" : "批轴/工程阶梯")]));
  table(s, rows, { x: MARGIN, y: 1.55, w: 7.10, colW: [1.4, 1.0, 1.1, 2.4], rowH: 0.44, fontSize: 9 });
  const rules = [
    ["模型身份", "124.78 属于 r2dtr1c-s0，不是 xens2；F1=0.667971。"],
    ["端到端范围", "含 frozen RiNALMo 前向、9.3M 头与 C 解码；不是 encoder-cache 口径。"],
    ["硬件边界", "UFold/RNAformer/MXfold2 为 A100 板；4090 点不与其混算 Pareto。"],
    ["参照限制", "RiNALMo-ft 14.5 seq/s 为 forward-only，单列，不能冒充完整系统延迟。"],
    ["有效性", "tp/fp/fn=24406/13778/10485；批大小不改变预测，B=384/512 只作饱和检查。"],
  ];
  rules.forEach((r, i) => {
    const y = 1.55 + i * 0.97;
    rect(s, 7.93, y, 4.88, 0.79, i === 0 ? T.paleBlue : T.surface, T.rule);
    text(s, r[0], 8.12, y + 0.10, 4.46, 0.23, 10.5, T.blue, { bold: true });
    text(s, r[1], 8.12, y + 0.39, 4.46, 0.30, 8.6, T.ink);
  });
  s.render();
}

// 17 附录 C：演进与机制映射
{
  const s = pptx.addSlide();
  chrome(s, "附录 C / 演进", "模型演进与 Jev 机制映射", "保留演进与竞争定位，但不把历史读数重新当成当前主结果。", 18, "模型演进表 + spec J1/J2/J3/J4/J6");
  const rows = [
    ["阶段", "关键变化", "TS0 F1", "决策"],
    ["r2d / tr0", "frozen RiNALMo + ResNet2D", "0.6629", "二维上下文有效；旧语料不用于最终主表"],
    ["r2d / tr1c", "去污染语料与冻结 split", "0.6679", "撤回虚高值，保留干净控制臂"],
    ["plana / tr1c", "渐进解冻，精度家族形成", "0.7866", "ID 提升；OOD 种子方差单列"],
    ["xens2", "plana 两种子 × r2d，w=0.7", "0.8039", "家族互补；统一模型身份"],
    ["Round 32", "参照与模型身份再审", "6/8 clean", "旧 7/8 混合口径撤回；全板敏感性单列"],
  ];
  table(s, rows, { x: MARGIN, y: 1.55, w: 7.30, colW: [1.55, 2.8, 0.9, 3.0], rowH: 0.62, fontSize: 8.7 });
  const maps = [
    ["J2 · 单次前向", "L×L pair logits；无自回归路径"],
    ["J3 · 合法输出", "非交叉 DP；非法结构率 0"],
    ["J4 · 置信度", "VL0 仿射 + SC/ECE；测试集不重拟合"],
    ["J6 · System-1/2", "System-1 免配分函数；精确边际只作参照/教师"],
    ["J1 · 假设空间", "最终采用扁平头+DP；层级级联负结果不写成贡献"],
  ];
  maps.forEach((r, i) => {
    const y = 1.55 + i * 0.92;
    rect(s, 8.13, y, 4.68, 0.74, i === 4 ? T.paleRed : T.surface, T.rule);
    text(s, r[0], 8.32, y + 0.10, 4.25, 0.23, 10.2, i === 4 ? T.red : T.blue, { bold: true });
    text(s, r[1], 8.32, y + 0.39, 4.25, 0.23, 8.4, T.ink);
  });
  text(s, "竞争定位：不主张首次免 DP 或首次校准；贡献在去污染评测、结构先验 OOD、效率与迁移的组合证据。", MARGIN, 6.42, CONTENT_W, 0.30, 9.5, T.muted);
  s.render();
}

// 18 附录 D：当前实验与复现状态
{
  const s = pptx.addSlide();
  chrome(s, "附录 D / 当前队列", HAS_MATCHED ? "实验状态与统计门：所有结果先验证，再决定主文呈现" : "新实验只进入记录，不在主文伪装成收官结果", HAS_MATCHED ? "matched_30k 三种子统计已纳入；mixed_s0/s1 的负方向仍如实保留。" : "matched_30k 三种子尚未齐；mixed_s0 单点负结果如实保留。", 19, "T-A126–T-A132 / audits / result.json / run_meta.json");
  const cur = D.matched_30k_current;
  const rows = [["arm", "Test", "GenA", "GenC", "GenF", "OOD mean", "当前判定"]];
  ["clean_s0", "mixed_s0", "clean_s1", "mixed_s1"].forEach(name => {
    const v = cur[name];
    const ood = (v[1] + v[2] + v[3]) / 3;
    rows.push([name, ...v.map(x => x.toFixed(4)), ood.toFixed(4), name.startsWith("mixed") ? "单点未过门" : "已完成三通道复核"]);
  });
  const currentRowH = rows.length <= 5 ? 0.45 : 0.36;
  const stateY = rows.length <= 5 ? 4.05 : 4.35;
  const currentCaptionY = rows.length <= 5 ? 6.46 : 6.70;
  table(s, rows, { x: MARGIN, y: 1.55, w: CONTENT_W, colW: [1.65, 0.85, 0.85, 0.85, 0.85, 1.05, 2.35], rowH: currentRowH, fontSize: 8.7 });
  const states = [
    ["T-A127", "matched 六臂与统计", HAS_MATCHED ? "24 格、3-seed mean±std 与 Wilcoxon–Holm 已完成；按预注册门判读。" : "mixed_s1 已 early-stopping；s2 与 24 格验收后继续，未齐不写最终结论。"],
    ["T-A128", "五类消融 40/40", HAS_AB ? "40/40 与控制 8/8 已完成；按 ID/外部/PDB-OOD 分组读图。" : "既有权重直接补评；norlcd 为唯一重训臂，VL0 early-stopping。"],
    ["T-A129", "新增 RiNALMo 下游", HAS_MRL ? DOWNSTREAM_NOTE : "三组 MRL 已预注册；5-seed、early-stopping、paired t-test+Holm。"],
    ["复现门", "代码与证据", `GitHub round29-handoff 已推送；${HAS_MATCHED ? "24" : "16"} 格三通道 PASS；权重与数据 SHA256 留档。`],
  ];
  states.forEach((r, i) => {
    const x = MARGIN + (i % 2) * 6.18, y = stateY + Math.floor(i / 2) * 1.20;
    rect(s, x, y, 5.96, 1.00, i === 3 ? T.paleTeal : T.surface, T.rule);
    text(s, `${r[0]} · ${r[1]}`, x + 0.18, y + 0.12, 5.60, 0.25, 10.5, i === 3 ? T.teal : T.blue, { bold: true });
    text(s, r[2], x + 0.18, y + 0.46, 5.60, 0.40, 8.7, T.ink);
  });
  text(s, "splice、GenF 与其他负结果均保留在原始结果和台账；主文只选择有明确科学问题与完整证据的内容。", MARGIN, currentCaptionY, CONTENT_W, 0.28, 9.4, T.muted);
  s.render();
}

// 19 结束页
{
  const s = pptx.addSlide();
  s.background = { color: T.dark };
  text(s, "RNA-JEV / ROUND 32", 0.68, 0.62, 5.8, 0.30, 11, "73D4D0", { bold: true, charSpacing: 1.5 });
  text(s, "把每个数字放回它所属的\n模型、数据集和协议", 0.68, 1.55, 8.8, 1.62, 29, T.white, { bold: true });
  const checks = [
    ["01", "模型身份", "统一 xens2=6/8；历史混合集成不再冒充单一模型。"],
    ["02", "科学证据", "缺一格即不出图；负结果、OOD 与校准边界同页呈现。"],
    ["03", "可复现交付", "原始 result.json、权重/数据哈希、训练台账与 GitHub 提交互相对应。"],
  ];
  checks.forEach((r, i) => {
    const y = 3.68 + i * 0.92;
    token(s, r[0], { x: 0.72, y, w: 0.48, h: 0.36, fontSize: 17, bold: true, color: "73D4D0" });
    text(s, r[1], 1.46, y - 0.01, 2.25, 0.28, 12, T.white, { bold: true });
    text(s, r[2], 3.65, y + 0.01, 8.60, 0.34, 11, "C4CEDD");
  });
  text(s, "主文 10 图 · 附录 4 组 · 微软雅黑 · 1.5 倍行距", 0.68, 6.72, 8.2, 0.28, 10.5, "AAB6C8");
  token(s, "END", { x: 11.42, y: 6.65, w: 1.15, h: 0.35, fontSize: 13, bold: true, color: "73D4D0", align: "right" });
  s.render();
}

bounds();
await pptx.writeFile({ fileName: OUT });
console.log(`DECK_WRITTEN ${OUT} slides=${CURRENT_SLIDE} ablation=${HAS_AB ? "complete" : "draft"} mrl=${HAS_MRL} matched=${HAS_MATCHED}`);
