"""Full metrics matrix: every measured (method, dataset) cell with ALL
available metrics (micro P/R/F1, macro F1, INF, n) — read from result.json
files only, no typing. Output: tables/metrics_matrix.md + .json (2026-10-02).
Quoted-only rows (NucleicBERT, RiNALMo S5 table) are listed separately at
the end with their source flag, since they have no result.json here.
"""
import json
from pathlib import Path

ART = Path("/mnt/cunyuliu/rna-jepa/eval_decision")
OUT_MD = Path("/mnt/cunyuliu/rna-jepa/tables/metrics_matrix.md")
OUT_JSON = Path("/mnt/cunyuliu/rna-jepa/tables/metrics_matrix.json")

# (label, kind, path-ish, dataset) — kinds: baseline_json (baselines_*.json
# with baselines/<name>/...), ow (ow_* result.json pair_level+per_seq INF),
# plana (eval_plan_a style), probe (calibration probe)
SOURCES = []

def baseline(label, fname, ds):
    SOURCES.append((label, "baseline", ART / fname, ds))

def ow(label, dirname, ds):
    SOURCES.append((label, "ow", ART / dirname / "result.json", ds))

def plana(label, dirname, ds):
    SOURCES.append((label, "plana", ART / dirname / "result.json", ds))

# --- Vienna family / Nussinov prior (project scorer) ---
for ds, fn in [("TS0", "baselines_bprna_ts0.json"), ("bpRNA-new", "baselines_bprna_new.json")]:
    baseline("ViennaRNA centroid", fn, ds)
    baseline("ViennaRNA mfe", fn, ds)
    baseline("ViennaRNA mea", fn, ds)
    baseline("Nussinov+Turner (our prior)", fn, ds)
# TestSetB
for nm in ["vienna_centroid", "vienna_mfe", "nussinov_turner", "eternafold"]:
    baseline(nm, "baselines_testsetb.json", "TestSetB")
# clean ArchiveII per-bucket Vienna
for nm, fn in [("vienna_centroid", f"baselines_archiveii_embok_clean_{b}.json")
               for b in ["le100", "gt100_le200", "gt200_le400", "gt400"]]:
    baseline(nm, fn, "ArchiveII-clean " + {
        "le100": "<=100", "gt100_le200": "100-200", "gt200_le400": "200-400",
        "gt400": ">400"}[fn.split("_clean_")[1].split(".")[0]])

# --- external models re-run through the project scorer ---
baseline("UFold", "baselines_ufold_bprna_ts0.json", "TS0")
baseline("UFold", "baselines_ufold_bprna_new.json", "bpRNA-new")
baseline("MXfold2", "baselines_mxfold2_bprna_ts0.json", "TS0")
baseline("MXfold2", "baselines_mxfold2_bprna_new.json", "bpRNA-new")
baseline("RNAformer (bprna ckpt)", "baselines_rnaformer_ref_bprna_ts0.json", "TS0")
baseline("RNAformer (bprna ckpt)", "baselines_rnaformer_bprna_new.json", "bpRNA-new")
for s in ["ref_pdb_ts2", "ref_pdb_ts3"]:
    baseline("RNAformer (bprna ckpt)", f"baselines_rnaformer_{s}.json", s.replace("ref_", ""))

# --- our arms (ow_* result.json: pair_level micro + macro, INF mean) ---
def ow_if(label, dirname, ds):
    p = ART / dirname / "result.json"
    if p.exists():
        ow(label, dirname, ds)

ow("Ours ff (8-seed family, s0 shown)", "ow_rinalmo_ff_b4_s0_step20000_bprna_ts0", "TS0")
ow_if("Ours ff", "ow_rinalmo_ff_b4_s0_step20000_bprna_new", "bpRNA-new")
ow_if("Ours big (4.8x capacity)", "ow_rinalmo_big_b4_s0_step20000_bprna_ts0", "TS0")
ow_if("Ours big", "ow_rinalmo_big_b4_s0_step20000_bprna_new", "bpRNA-new")
ow_if("Ours TR1 (4.29x data)", "ow_rinalmo_ff_tr1_b4_s0_step20000_bprna_ts0", "TS0")
ow_if("Ours TR1", "ow_rinalmo_ff_tr1_b4_s0_step20000_bprna_new", "bpRNA-new")
ow_if("Ours TR1 @40k", "ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_ts0", "TS0")
ow_if("Ours TR1 @40k", "ow_rinalmo_ff_tr1_b4_s0_step40000_bprna_new", "bpRNA-new")
ow_if("Ours bigtr1 (capacity x data)", "ow_rinalmo_bigtr1_b4_s0_step20000_bprna_ts0", "TS0")
ow_if("Ours bigtr1", "ow_rinalmo_bigtr1_b4_s0_step20000_bprna_new", "bpRNA-new")
ow("Ours r2d (Plan-B, frozen+2D, s0)", "ow_rinalmo_r2d_b4_s0_step20000_bprna_ts0", "TS0")
ow("Ours r2d (Plan-B, s0)", "ow_rinalmo_r2d_b4_s0_step20000_bprna_new", "bpRNA-new")
ow("Ours r2d (Plan-B, s1)", "ow_rinalmo_r2d_b4_s1_step20000_bprna_ts0", "TS0")
ow("Ours r2d (Plan-B, s1)", "ow_rinalmo_r2d_b4_s1_step20000_bprna_new", "bpRNA-new")

# --- plana arms ---
plana("Ours Plan-A (adapted+2D, s0)", "plana_giga_s0_step20000", "TS0+new")
plana("Ours Plan-A s1", "plana_giga_s1_step20000", "TS0+new")
plana("Ours Plan-A s2", "plana_giga_s2_step20000", "TS0+new")
plana("Ours Plan-A ext40k @30k", "plana_giga_s0_ext40k_step30000", "TS0+new")
plana("Ours Plan-A ext40k @40k", "plana_giga_s0_ext40k_step40000", "TS0+new")

def read_cell(label, kind, path, ds):
    if not path.exists():
        return None
    if kind == "baseline":
        d = json.load(open(path))
        for name, b in d.get("baselines", {}).items():
            # match label to the baseline name we declared
            want = label.split(" (")[0].replace("ViennaRNA ", "vienna_").replace(" ", "_").lower()
            got = name.lower()
            if want in got or got in want or label.startswith(name) or name.startswith(label.split(" (")[0]):
                return {"label": label, "dataset": ds,
                        "micro_P": round(b["micro_precision"], 4),
                        "micro_R": round(b["micro_recall"], 4),
                        "micro_F1": round(b["micro_f1"], 4),
                        "macro_F1": round(b["macro_f1"], 4),
                        "INF": round(b["inf"], 4) if "inf" in b else None,
                        "n": b.get("n_sequences", d.get("n_sequences")),
                        "source": str(path.name)}
        return None
    if kind == "ow":
        d = json.load(open(path))
        pl = d["pair_level"]
        per = d.get("per_sequence", [])
        inf = round(sum(r["inf"] for r in per) / len(per), 4) if per else None
        return {"label": label, "dataset": ds,
                "micro_P": round(pl["micro"]["precision"], 4),
                "micro_R": round(pl["micro"]["recall"], 4),
                "micro_F1": round(pl["micro"]["f1"], 4),
                "macro_F1": round(pl["macro"]["f1"], 4) if isinstance(pl.get("macro"), dict) else round(pl["macro_f1"], 4) if "macro_f1" in pl else None,
                "INF": inf, "n": d.get("n_sequences"),
                "source": str(path.parent.name)}
    if kind == "plana":
        d = json.load(open(path))
        rows = []
        for sp, m in d["splits"].items():
            rows.append({"label": label, "dataset": {"ts0": "TS0", "new": "bpRNA-new"}[sp],
                         "micro_P": round(m["micro"]["precision"], 4),
                         "micro_R": round(m["micro"]["recall"], 4),
                         "micro_F1": round(m["micro"]["f1"], 4),
                         "macro_F1": round(m["macro_f1"], 4), "INF": None,
                         "n": m["n_sequences"],
                         "source": str(path.parent.name)})
        return rows

cells = []
for label, kind, path, ds in SOURCES:
    r = read_cell(label, kind, path, ds)
    if r is None:
        continue
    if isinstance(r, list):
        cells.extend(r)
    else:
        cells.append(r)

QUOTED = [
    {"label": "NucleicBERT ft (quoted)", "dataset": "TS0 (Mathews macro)", "micro_P": 0.718, "micro_R": 0.610, "micro_F1": None, "macro_F1": 0.649, "INF": None, "n": 1305, "source": "NucleicBERT Table 1 (quoted)"},
    {"label": "RNAErnie+ (quoted)", "dataset": "TS0 (Mathews macro)", "micro_P": 0.575, "micro_R": 0.678, "micro_F1": None, "macro_F1": 0.622, "INF": None, "n": 1305, "source": "NucleicBERT Table 1 (quoted)"},
    {"label": "RNA-FM ft (quoted)", "dataset": "TS0 (Mathews macro)", "micro_P": 0.518, "micro_R": 0.620, "micro_F1": None, "macro_F1": 0.564, "INF": None, "n": 1305, "source": "NucleicBERT Table 1 (quoted)"},
    {"label": "RiNALMo-ft (quoted)", "dataset": "TestSetB", "micro_P": None, "micro_R": None, "micro_F1": None, "macro_F1": None, "INF": 0.67, "n": 430, "source": "RiNALMo Supp S5 (quoted)"},
    {"label": "CONTRAfold (quoted)", "dataset": "TestSetB", "micro_P": None, "micro_R": None, "micro_F1": None, "macro_F1": None, "INF": 0.64, "n": 430, "source": "RiNALMo Supp S5 (quoted)"},
    {"label": "MXfold2 (quoted)", "dataset": "TestSetB", "micro_P": None, "micro_R": None, "micro_F1": None, "macro_F1": None, "INF": 0.63, "n": 430, "source": "RiNALMo Supp S5 (quoted)"},
    {"label": "RNAstructure (quoted)", "dataset": "TestSetB", "micro_P": None, "micro_R": None, "micro_F1": None, "macro_F1": None, "INF": 0.56, "n": 430, "source": "RiNALMo Supp S5 (quoted)"},
    {"label": "RNA-FM (quoted)", "dataset": "TestSetB", "micro_P": None, "micro_R": None, "micro_F1": None, "macro_F1": None, "INF": 0.49, "n": 430, "source": "RiNALMo Supp S5 (quoted)"},
]

# ---- render markdown ----
def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}" if isinstance(v, float) else str(v)

lines = ["# Full metrics matrix — every measured (method, dataset) cell",
         "",
         "All values re-derived from result.json / baselines_*.json on disk",
         "(script tools/metrics_matrix.py; no hand-typed numbers). Quoted",
         "rows carry their paper source and are flagged. '—' = the metric is",
         "not produced by that run's protocol (e.g. INF needs per-seq rows).",
         "",
         "## Measured cells",
         "",
         "| Method | Dataset | micro P | micro R | micro F1 | macro F1 | INF | n | source |",
         "|---|---|---|---|---|---|---|---|---|"]
for c in sorted(cells, key=lambda x: (x["dataset"], -x["micro_F1"])):
    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        c["label"], c["dataset"], fmt(c["micro_P"]), fmt(c["micro_R"]),
        fmt(c["micro_F1"]), fmt(c["macro_F1"]), fmt(c["INF"]), c["n"], c["source"]))
lines += ["", "## Quoted (paper) rows — not measured here",
          "",
          "| Method | Dataset | P | R | F1 | macro F1 | INF | n | source |",
          "|---|---|---|---|---|---|---|---|---|"]
for c in QUOTED:
    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        c["label"], c["dataset"], fmt(c["micro_P"]), fmt(c["micro_R"]),
        fmt(c["micro_F1"]), fmt(c["macro_F1"]), fmt(c["INF"]), c["n"], c["source"]))

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
json.dump({"measured": cells, "quoted": QUOTED},
          open(OUT_JSON, "w"), indent=1)
print(f"written {OUT_MD} ({len(cells)} measured cells, {len(QUOTED)} quoted rows)")
