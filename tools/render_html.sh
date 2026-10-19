#!/bin/bash
# Render the preprint markdown to a self-contained HTML for review/submission
# prep (bioRxiv accepts PDF; HTML is the review-read format and converts
# losslessly via browser print). CJK not expected in body (English draft).
set -e
DST=/tmp/rna_jev_deliverables
SRC="$DST/paper/preprint_draft.md"
OUT="$DST/paper/preprint_draft_v3.26.html"

# CSS: manuscript-like typography
CSS=/tmp/ppt_work/manuscript.css
cat > $CSS <<'EOF'
@page { size: A4; margin: 22mm 20mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt;
       line-height: 1.45; color: #111; max-width: 100%; margin: 0 auto; }
h1 { font-size: 17pt; text-align: left; margin: 0 0 6pt; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2pt;
     margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; font-size: 9pt; margin: 8pt 0;
        page-break-inside: avoid; }
th, td { border: 0.5pt solid #888; padding: 2pt 5pt; text-align: left; }
th { background: #eee; }
blockquote { border-left: 2.5pt solid #0e7c86; margin: 6pt 0; padding: 2pt 10pt;
             color: #444; background: #f6f8f9; font-size: 9.5pt; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f2f2f2; }
img { max-width: 100%; }
EOF

pandoc "$SRC" -f markdown -t html5 -s --toc --toc-depth=2 \
  --metadata title="DP-Free Calibrated Base-Pair Probabilities for RNA Secondary Structure Prediction (v3.26 draft)" \
  -c $CSS -o "$OUT"
echo "HTML written: $OUT"
ls -la "$OUT"
