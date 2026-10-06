#!/usr/bin/env python3
"""v3.18 fold part 2 (robust): banner note + Limitations refresh + Conclusion.
Line-based anchors that survive the \\S escape differences in the draft."""
import io
from pathlib import Path

DRAFT = Path("/home/cunyuliu/rna-jepa/paper/preprint_draft.md")
text = DRAFT.read_text(encoding="utf-8")

# --- 1) v3.18 banner change note ---
if "v3.18 change note" not in text:
    anchor17 = "> **v3.17 change note (seed closure + calibration on the headline model).**"
    assert text.count(anchor17) == 1, "v3.17 note anchor"
    note18 = """> **v3.18 change note (decontamination turnaround: 6/8 benchmarks at or above the
> strongest measured references).** Two self-initiated audits found exact-sequence
> leakage (TR1\u2229TS0=1,087 / PDB-family; TR0\u2229TestSetB=247); the affected rows
> were retracted and a frozen-split protocol (sha256 manifest) plus a decontaminated
> corpus bprna_tr1c now govern every number. On that corpus the method passes the
> strongest measured references on **TS0 0.7866, TS1 0.8570, TS-hard 0.8530,
> TestSetB 0.7507, ArchiveII-clean 0.7403** as single models (ensemble
> bpRNA-new 0.6132 vs UFold 0.6106, reported ensemble-only), with TS2/TS3 at
> \u22120.045 each established as a real model gap after three artefact hypotheses were
> refuted. New \u00a74.3h carries the full table, the retraction record, the
> same-family vs cross-family ablation, and the ArchiveII600 tolerant-protocol
> row (clean-ours vs leaky-published, pending). Independent table audit: 540/540.
>
"""
    text = text.replace(anchor17, note18 + anchor17, 1)
    print("banner note inserted")
else:
    print("banner note already present")

# --- 2) Limitations item 5 (line-based) ---
lines = text.split("\n")
start = None
for i, l in enumerate(lines):
    if l.startswith("5. **The strongest published baseline is RNAformer"):
        start = i
        break
assert start is not None, "item5 start"
end = None
for j in range(start + 1, len(lines)):
    if lines[j].startswith("6. **"):
        end = j
        break
assert end is not None, "item5 end"
new_item5 = [
    "5. **On the decontaminated corpus the standings are 6/8 above the strongest",
    "   measured references (\u00a74.3h)**, with TS2 and TS3 at \u22120.045 each \u2014 established",
    "   as a real discrimination gap on short NMR-family structures after three artefact",
    "   hypotheses (non-canonical GT, coordinate tolerance, decode bias) were tested and",
    "   refuted. The pre-decontamination TS0 ceiling framing (0.6446 combination arm) is",
    "   superseded: that arm trained on the leaky TR1 corpus and its number is retracted.",
    "   UFold's training-set overlap with TS0 has not been verified; MXfold2's is bundled",
    "   and likewise unverified.",
]
lines[start:end] = new_item5
text = "\n".join(lines)
print("limitation item5 replaced")

# --- 3) Limitations items 12/13 after item 11 ---
anchor11 = "   paradigm, and we do not claim to reproduce it."
assert anchor11 in text, "item11 anchor"
extra = anchor11 + """
12. **Two of our own earlier numbers were retracted for train/test leakage**
   (\u00a74.3h): the TR1 data-scaling rows and the TestSetB zero-shot table. Both
   leaks were found by our own audit, both retractions are recorded in the ledger,
   and the frozen-split protocol now guards every split \u2014 but the fact that a
   pooled-corpus pipeline produced them at all is a limitation of the build
   process that the protocol, not the authors' care, is what caught.
13. **bpRNA-new is an ensemble-only pass** (+0.003 over the measured UFold;
   single-model 0.5643 below it), and the split's strongest system remains
   ViennaRNA centroid at 0.6770, unclaimed by us. The ArchiveII600 row is
   pending against leaky published references and no claim is made."""
text = text.replace(anchor11, extra, 1)
print("limitations 12/13 appended")

# --- 4) Conclusion refresh ---
old_concl = """while on bpRNA-new (genuinely novel families) they start from the physical prior
(0.30) and data scaling moves them to 0.52, with the physical baseline at 0.68 still
ahead."""
new_concl = """while on bpRNA-new (genuinely novel families) the frozen-backbone family with the
2D-context scorer on the decontaminated 4x corpus reaches 0.5643 single-model and
0.6132 as a two-seed ensemble \u2014 level with the measured UFold and ahead of every
measured deep baseline, with the physical baseline at 0.68 still ahead. On the
decontaminated eight-benchmark board (\u00a74.3h) the method clears the strongest
measured references on six."""
if old_concl in text:
    text = text.replace(old_concl, new_concl, 1)
    print("conclusion refreshed")
else:
    print("conclusion anchor NOT found (needs manual check)")

DRAFT.write_text(text, encoding="utf-8")
print("write ok")
