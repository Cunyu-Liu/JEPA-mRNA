/**
 * Read back the generated deck's own OOXML and report each text frame's geometry.
 *
 * Why this exists: the only renderer available on this machine (Quick Look) lays
 * text out without honouring the text-frame width, so a thumbnail shows copy
 * running past the right edge whether or not the box is actually too narrow.
 * Reading cx back from the file settles it against the source of truth rather
 * than against a preview artifact.
 */
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";

const deck = process.argv[2];
const slideCount = Number(process.argv[3] || 13);
const EMU = 914400;

let flagged = 0;
for (let i = 1; i <= slideCount; i += 1) {
  const xml = execSync(
    `unzip -p ${JSON.stringify(deck)} ppt/slides/slide${i}.xml`,
    { maxBuffer: 64 * 1024 * 1024 },
  ).toString();

  const shapes = xml.match(/<p:sp>[\s\S]*?<\/p:sp>/g) || [];
  const rows = [];
  for (const sp of shapes) {
    const ext = sp.match(/<a:ext cx="(\d+)" cy="(\d+)"/);
    const texts = [...sp.matchAll(/<a:t>([\s\S]*?)<\/a:t>/g)].map((m) => m[1]);
    if (!ext || !texts.length) continue;
    const off = sp.match(/<a:off x="(\d+)" y="(\d+)"/);
    const w = Number(ext[1]) / EMU;
    const h = Number(ext[2]) / EMU;
    const x = off ? Number(off[1]) / EMU : NaN;
    const y = off ? Number(off[2]) / EMU : NaN;
    const wrapNone = /wrap="none"/.test(sp);
    const body = texts.join("").replace(/\s+/g, " ").slice(0, 30);
    const right = x + w;
    const bottom = y + h;
    const bad = [];
    if (Number.isFinite(right) && right > 13.333 - 0.30) bad.push(`right=${right.toFixed(2)}`);
    if (Number.isFinite(bottom) && bottom > 7.5 - 0.10) bad.push(`bottom=${bottom.toFixed(2)}`);
    if (bad.length) flagged += 1;
    if (bad.length || i === 1) {
      rows.push(`    x=${x.toFixed(2)} y=${y.toFixed(2)} w=${w.toFixed(2)} h=${h.toFixed(2)} `
        + `wrapNone=${wrapNone ? "Y" : "n"} ${bad.join(" ")} | ${body}`);
    }
  }
  console.log(`slide ${String(i).padStart(2, "0")}: ${shapes.length} shapes, `
    + `${rows.length ? "flagged" : "in bounds"}`);
  rows.forEach((r) => console.log(r));
}
console.log(`\ntotal frames outside the safe area: ${flagged}`);