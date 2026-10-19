#!/usr/bin/env python3
"""Embed figures + inline CSS into the pandoc HTML, making it fully
self-contained. Inserts the two evidence figures right before the
'Appendix D. Figure captions' section (which describes them)."""
import base64, re

HTML = "/tmp/rna_jev_deliverables/paper/preprint_draft_v3.26.html"
CSS = "/tmp/ppt_work/manuscript.css"
FIG1 = "/tmp/rna_jev_deliverables/figs/rnajev_evidence_panel.png"
FIG2 = "/tmp/rna_jev_deliverables/figs/seed_family_pr.png"
OUT = "/tmp/rna_jev_deliverables/paper/preprint_draft_v3.26_standalone.html"

src = open(HTML, encoding="utf-8").read()
css = open(CSS, encoding="utf-8").read()

# inline CSS (replace the <link>)
assert '<link rel="stylesheet" href="/tmp/ppt_work/manuscript.css" />' in src
src = src.replace('<link rel="stylesheet" href="/tmp/ppt_work/manuscript.css" />',
                  f"<style>\n{css}\n</style>")

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()

fig_html = (
    '<div class="figures">'
    f'<figure><img src="data:image/png;base64,{b64(FIG1)}" '
    'alt="Figure 1: evidence panel" style="width:100%"/>'
    "<figcaption><b>Figure 1.</b> Six-panel evidence summary (see Appendix D "
    "for the caption).</figcaption></figure>"
    f'<figure><img src="data:image/png;base64,{b64(FIG2)}" '
    'alt="Figure 2: seed family P/R" style="width:72%"/>'
    "<figcaption><b>Figure 2.</b> Three-seed P/R family profiles (see "
    "Appendix D for the caption).</figcaption></figure>"
    "</div>")

anchor = '<h2 id="appendix-d.-figure-captions">Appendix D. Figure captions</h2>'
assert anchor in src, "appendix D anchor"
src = src.replace(anchor, fig_html + "\n" + anchor, 1)

# figure styling
src = src.replace("</style>", (
    "figure { margin: 10pt 0; text-align: center; page-break-inside: avoid; }\n"
    "figcaption { font-size: 9pt; color: #333; margin-top: 4pt; }\n"
    "</style>"), 1)

open(OUT, "w", encoding="utf-8").write(src)
print("standalone HTML:", OUT, len(src), "bytes")
