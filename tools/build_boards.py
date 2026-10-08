#!/usr/bin/env python3
"""T-A45: build the published-baseline board (8 splits x 9 baseline systems)
and the ours-history board (8 splits x our recipe family), all cells
re-derived from result.json / baselines_*.json on disk. Then extend
tables/sota_vs_ours.md with:
  (a) the baseline board (measured cells + '—' gaps, provenance column),
  (b) the ours-history board (one row per trained arm, best protocol cell),
  (c) a provenance note answering "are the reference values ours?".
"""
import json
import os
import glob
import re

E = "/mnt/cunyuliu/rna-jepa/eval_decision"
OUT = "/home/cunyuliu/rna-jepa/tables/sota_vs_ours.md"
SPLITS = ["TS0", "bpRNA-new", "TS1", "TS2", "TS3", "TS-hard", "ArchiveII-clean", "TestSetB"]
SPLIT_KEYS = {
    "TS0": "bprna_ts0", "bpRNA-new": "bprna_new", "TS1": "ref_pdb_ts1",
    "TS2": "ref_pdb_ts2", "TS3": "ref_pdb_ts3", "TS-hard": "ref_pdb_ts_hard",
    "ArchiveII-clean": "archiveii_embok_clean", "TestSetB": "testsetb",
}

def micro_of(d):
    try:
        return round(d["pair_level"]["micro"]["f1"], 4)
    except Exception:
        return None

# ---------- baseline board ----------
# Vienna trio + nussinov live in baselines_{split}.json -> baselines.<method>.micro_f1
# per-method: mxfold2_{split}, ufold via ufold_* result dirs / baselines files,
# rnaformer via baselines_rnaformer_* files, eternafold_testsetb.json
board = {}

def sp_label_of(base):
    # exact split name (multi-method file: baselines_{split}.json)
    for sp_label, sp in SPLIT_KEYS.items():
        if base == sp:
            return sp_label
    # suffixed (baselines_{variant}_{split}.json)
    for sp_label, sp in SPLIT_KEYS.items():
        if base.endswith("_" + sp):
            return sp_label
    return None

for f in sorted(glob.glob(f"{E}/baselines_*.json")):
    base = os.path.basename(f)[len("baselines_"):-5]
    d = json.load(open(f))
    inner = d.get("baselines", {})
    split_label = sp_label_of(base)
    if split_label is None:
        continue
    variant = None
    for sp in SPLIT_KEYS.values():
        if base.endswith("_" + sp):
            variant = base[: -(len(sp) + 1)]
            break
    for method, mres in inner.items():
        v = mres.get("micro_f1")
        if v is None:
            continue
        if variant in (None, "", "ref", "ufold_ref", "eternafold"):
            key = {"mxfold2": "MXfold2", "ufold": "UFold", "eternafold": "EternaFold"}.get(method, method)
            board.setdefault(key, {})[split_label] = round(v, 4)
        elif variant.startswith("rnaformer"):
            label = {"rnaformer_bprna": "RNAformer (bprna ckpt)",
                     "rnaformer_interfam": "RNAformer (inter-family ckpt)",
                     "rnaformer_bio": "RNAformer (biophysical ckpt)"}.get(variant, variant)
            board.setdefault(label, {})[split_label] = round(v, 4)
        elif variant in ("mxfold2", "ufold"):
            key = "MXfold2" if variant == "mxfold2" else "UFold"
            board.setdefault(key, {})[split_label] = round(v, 4)

# eternafold testsetb
p = f"{E}/eternafold_testsetb.json"
if os.path.exists(p):
    d = json.load(open(p))
    board.setdefault("EternaFold", {})["TestSetB"] = round(d["strict_micro_f1"], 4)

# RiNALMo-ft (Zenodo ckpt rinalmo_giga_ss_bprna_ft.pt): our_protocol.strict_micro_f1
# from rinalmo_ft_{split}.json; TS0 prefers the board split file (bprna_ts0)
# over the official-1305 run (rinalmo_ft_official_ts0.json).
for sp_label, sp in SPLIT_KEYS.items():
    p = f"{E}/rinalmo_ft_{sp}.json"
    if sp_label == "TS0" and not os.path.exists(p):
        p = f"{E}/rinalmo_ft_official_ts0.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    v = d.get("our_protocol", {}).get("strict_micro_f1")
    if v is not None:
        board.setdefault("RiNALMo-ft (Zenodo ckpt)", {})[sp_label] = round(v, 4)

# NucleicBERT (frozen official MLM encoder + official SSP head trained by us
# on bprna_tr1c): baselines_nucleicbert_{split}.json -> baselines.nucleicbert.micro_f1
for sp_label, sp in SPLIT_KEYS.items():
    p = f"{E}/baselines_nucleicbert_{sp}.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    v = d.get("baselines", {}).get("nucleicbert", {}).get("micro_f1")
    if v is not None:
        board.setdefault("NucleicBERT (frozen enc + our head)", {})[sp_label] = round(v, 4)

ORDER = ["vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner",
         "MXfold2", "EternaFold", "UFold",
         "RNAformer (bprna ckpt)", "RNAformer (inter-family ckpt)", "RNAformer (biophysical ckpt)",
         "RiNALMo-ft (Zenodo ckpt)", "NucleicBERT (frozen enc + our head)"]

lines = []
lines.append("")
lines.append("## Published baselines on the 8-split board (measured by this repository; — = not run)")
lines.append("")
lines.append("| baseline | " + " | ".join(SPLITS) + " |")
lines.append("|---|" + "---|" * len(SPLITS))
for m in ORDER:
    if m not in board:
        continue
    cells = []
    for sp in SPLITS:
        cells.append("%.4f" % board[m][sp] if sp in board[m] else "—")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")
lines.append("")
lines.append("Provenance: every cell is OUR measurement (project scorer, project GT,")
lines.append("same split files) — ViennaRNA 2.7.2, MXfold2 (repo weights), UFold")
lines.append("(released ufold_train_alldata.pt), RNAformer (3 released checkpoints),")
lines.append("EternaFold (make multi, EternaFoldParams.v1), RiNALMo-ft (Zenodo")
lines.append("giga_ss_bprna_ft.pt, strict pairs re-scored by our protocol), NucleicBERT")
lines.append("(official frozen MLM encoder; SSP head trained by us on bprna_tr1c with")
lines.append("their SecStruct2DPredictionHead — labelled, not a released SSP ckpt).")
lines.append("")

# ---------- ours-history board ----------
ours_rows = {}
for dd in sorted(glob.glob(f"{E}/*")):
    if not os.path.isdir(dd):
        continue
    name = os.path.basename(dd)
    p = os.path.join(dd, "result.json")
    if not os.path.exists(p):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    v = micro_of(d)
    if v is None:
        continue
    for label, sp in SPLIT_KEYS.items():
        if name.endswith("_" + sp):
            arm = name[: -(len(sp) + 1)]
            ours_rows.setdefault(arm, {})[label] = v
            break

# plana arms (plan_a_result.json multi-split)
for dd in sorted(glob.glob(f"{E}/plana_*")):
    p = os.path.join(dd, "plan_a_result.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    arm = os.path.basename(dd)
    mapping = {"ts0": "TS0", "new": "bpRNA-new", "ts1": "TS1", "ts2": "TS2",
               "ts3": "TS3", "hard": "TS-hard", "archiveii": "ArchiveII-clean", "testsetb": "TestSetB"}
    for k, v in d.get("splits", {}).items():
        lab = mapping.get(k)
        if lab:
            ours_rows.setdefault(arm, {})[lab] = round(v["micro"]["f1"], 4)

lines.append("## Our trained models on the 8-split board (history; — = not evaluated)")
lines.append("")
lines.append("| arm (recipe family) | " + " | ".join(SPLITS) + " |")
lines.append("|---|" + "---|" * len(SPLITS))
# canonical order: final recipes first, then families
def arm_sort(a):
    pri = []
    if a.startswith("ow_rinalmo_r2dtr1c_b4_s0_step20000"): pri.append(0)
    elif a.startswith("xens2_"): pri.append(1)
    elif a.startswith("xens3_"): pri.append(2)
    elif a.startswith("plana_giga_tr1c"): pri.append(3)
    elif a.startswith("plana2same"): pri.append(4)
    elif a.startswith("xens_"): pri.append(5)
    elif a.startswith("ow_rinalmo_r2dtr1c"): pri.append(6)
    elif a.startswith("ow_rinalmo_fftr1c"): pri.append(7)
    else: pri.append(8)
    return (pri[0], a)
for arm in sorted(ours_rows, key=arm_sort):
    cells = []
    for sp in SPLITS:
        cells.append("%.4f" % ours_rows[arm][sp] if sp in ours_rows[arm] else "—")
    lines.append(f"| {arm} | " + " | ".join(cells) + " |")
lines.append("")

# append to sota_vs_ours.md (idempotent: remove old block first)
text = open(OUT).read()
for marker in ["## Published baselines on the 8-split board", "## Our trained models on the 8-split board"]:
    i = text.find(marker)
    if i != -1:
        text = text[:i].rstrip() + "\n"
text = text.rstrip() + "\n" + "\n".join(lines)
open(OUT, "w").write(text)
print("sota_vs_ours.md extended")
print("baseline systems:", [m for m in ORDER if m in board])
print("ours arms:", len(ours_rows))
