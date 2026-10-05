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
# 15.09: TS1 / TS-hard / TS2 / TS3 coverage (paper Table 4 comparisons)
PDB_DS = {"ref_pdb_ts1": "TS1", "ref_pdb_ts2": "TS2", "ref_pdb_ts3": "TS3",
          "ref_pdb_ts_hard": "TS-hard"}
for s in ["ref_pdb_ts2", "ref_pdb_ts3"]:
    baseline("RNAformer (bprna ckpt)", f"baselines_rnaformer_{s}.json", PDB_DS[s])
for s in ["ref_pdb_ts1", "ref_pdb_ts_hard"]:
    for tag, lbl in [("rnaformer_bprna", "RNAformer (bprna ckpt)"),
                     ("rnaformer_interfam", "RNAformer (inter-family ckpt, paper Tab.4 setting)")]:
        baseline(lbl, f"baselines_{tag}_{s}.json", PDB_DS[s])
# classical baselines already scored on TS1/TS-hard by the project scorer
for nm in ["vienna_centroid", "vienna_mfe", "vienna_mea", "nussinov_turner"]:
    baseline(nm, "baselines_ref_pdb_ts1.json", "TS1")
    baseline(nm, "baselines_ref_pdb_ts_hard.json", "TS-hard")
baseline("UFold", "baselines_ufold_ref_pdb_ts1.json", "TS1")

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
for _s in (10000, 20000):
    for _sp, _dsn in [("bprna_ts0", "TS0"), ("bprna_new", "bpRNA-new")]:
        ow_if("Ours r2d_tr1 (frozen+2D+TR1, s0 @%dk)" % _s,
              "ow_rinalmo_r2dtr1_b4_s0_step%d_%s" % (_s, _sp), _dsn)
# 15.09: r2d_tr1 on the PDB test family (RETRACTED 15.10: tr1 corpus carried
# exact test sequences; kept in the matrix only under the retraction notice,
# superseded by the r2dtr1c rows below)
for _sp, _dsn in [("ref_pdb_ts1", "TS1"), ("ref_pdb_ts_hard", "TS-hard"),
                  ("ref_pdb_ts2", "TS2"), ("ref_pdb_ts3", "TS3")]:
    ow_if("Ours r2d_tr1 (RETRACTED, leaky tr1)", "ts1hard_r2dtr1_s0_%s" % _sp, _dsn)
# ff arm has an existing TS1 eval
ow_if("Ours ff", "ff20000_ref_pdb_ts1", "TS1")

# 15.12: r2dtr1c — the same recipe on the DECONTAMINATED corpus. All six
# splits quotable. Rows for both the 20k auto-eval and the 8k curve probe.
for _sp, _dsn in [("bprna_ts0", "TS0"), ("bprna_new", "bpRNA-new"),
                  ("ref_pdb_ts1", "TS1"), ("ref_pdb_ts_hard", "TS-hard"),
                  ("ref_pdb_ts2", "TS2"), ("ref_pdb_ts3", "TS3")]:
    ow_if("Ours r2d_tr1c (frozen+2D+clean TR1, s0 @20k)",
          "ow_rinalmo_r2dtr1c_b4_s0_step20000_%s" % _sp, _dsn)
    ow_if("Ours r2d_tr1c (frozen+2D+clean TR1, s0 @8k)",
          "ow_rinalmo_r2dtr1c_b4_s0_step8000_%s" % _sp, _dsn)

# 15.13: TestSetB + ArchiveII-clean on the clean family (tr1c has zero
# overlap with both). The ff/big/bigtr1 TestSetB rows and the 15.11
# ensemble ArchiveII row are RETRACTED (TR0 ∩ TestSetB = 247; tr1 ∩
# ArchiveII-clean = 843) — only clean sources are wired in here.
for _st in (20000, 8000):
    ow_if("Ours r2d_tr1c (clean, s0 @%dk) TestSetB" % (_st // 1000),
          "ow_rinalmo_r2dtr1c_b4_s0_step%d_testsetb" % _st, "TestSetB")
    ow_if("Ours r2d_tr1c (clean, s0 @%dk) ArchiveII-clean" % (_st // 1000),
          "ow_rinalmo_r2dtr1c_b4_s0_step%d_archiveii_embok_clean" % _st,
          "ArchiveII-clean")

# 15.11: 2-seed ensemble (score-average, exact Nussinov) — tier-1 rows
ow_if("Ours r2d_tr1 2-seed ensemble @10k (leaky corpus; new split clean)",
      "ens_r2dtr1_2seed_bprna_ts0", "TS0")
ow_if("Ours r2d_tr1 2-seed ensemble @10k", "ens_r2dtr1_2seed_bprna_new", "bpRNA-new")
ow_if("Ours r2d_tr1 2-seed ensemble @10k", "ens_r2dtr1_2seed_archiveii_embok_clean", "ArchiveII-clean")
# UFold clean-ArchiveII reference rows (project scorer)
for _b, _n in [("le100", "ArchiveII-clean <=100"), ("gt100_le200", "ArchiveII-clean 100-200"),
               ("gt200_le400", "ArchiveII-clean 200-400"), ("gt400", "ArchiveII-clean >400")]:
    baseline("UFold", "ufold_archiveii_embok_clean_%s.json" % _b, _n)

# --- plana arms ---
plana("Ours Plan-A (adapted+2D, s0)", "plana_giga_s0_step20000", "TS0+new")
plana("Ours Plan-A s1", "plana_giga_s1_step20000", "TS0+new")
plana("Ours Plan-A s2", "plana_giga_s2_step20000", "TS0+new")
plana("Ours Plan-A ext40k @30k", "plana_giga_s0_ext40k_step30000", "TS0+new")
plana("Ours Plan-A ext40k @40k", "plana_giga_s0_ext40k_step40000", "TS0+new")
# 15.09: Plan-A seeds on the PDB test family (eval_plan_a --splits ts1,hard,ts2,ts3;
# closer_ts1hard.sh writes plan_a_result.json under <arm>_ts1hard_all/)
for _a in ["plana_giga_s0", "plana_giga_s1", "plana_giga_s2"]:
    _p = ART / ("%s_ts1hard_all" % _a) / "plan_a_result.json"
    if _p.exists():
        SOURCES.append(("Ours Plan-A %s (PDB family)" % _a[-2:], "plana", _p, "ts1+hard+ts2+ts3"))


# 15.18: xens cross-family ensemble (plana_tr1c x r2dtr1c). Two protocol
# generations on disk: w=0.5 default sweep (10-04 16:16-17:27) and the
# VL0-locked w=0.7 uniform protocol (single run per split). Both are kept;
# the sota table quotes only w0.7. VL0 w-sweep rows are selection-split
# runs, not test rows. 15.17: r2dtr1c s1 (seed=1) 6-split eval — worse
# basin, kept for the record.
XENS_DS = {"bprna_ts0": "TS0", "bprna_new": "bpRNA-new",
           "ref_pdb_ts1": "TS1", "ref_pdb_ts2": "TS2",
           "ref_pdb_ts3": "TS3", "ref_pdb_ts_hard": "TS-hard",
           "testsetb": "TestSetB", "archiveii_embok_clean": "ArchiveII-clean"}
for _sp, _dsn in XENS_DS.items():
    _p = ART / ("xens_%s" % _sp) / "result.json"
    if _p.exists():
        SOURCES.append(("Ours xens (plana_tr1c x r2dtr1c, w0.5 sweep)", "xens", _p, _dsn))
    _p7 = ART / ("xens_%s_w0.7" % _sp) / "result.json"
    if _p7.exists():
        SOURCES.append(("Ours xens (plana_tr1c x r2dtr1c, VL0-locked w0.7)", "xens", _p7, _dsn))
for _sp, _dsn in [("ref_pdb_ts1", "TS1"), ("ref_pdb_ts2", "TS2"),
                  ("ref_pdb_ts3", "TS3"), ("ref_pdb_ts_hard", "TS-hard")]:
    for _w in ("0.7", "0.9"):
        _p = ART / ("xens_%s_w%s" % (_sp, _w)) / "result.json"
        if _p.exists():
            SOURCES.append(("Ours xens %s w%s (lock-verify / exploratory)" % (_dsn, _w),
                            "xens", _p, _dsn))
for _sp, _dsn in [("bprna_vl0", "VL0")]:
    for _w in ("0.3", "0.5", "0.7", "0.8", "0.85", "0.9"):
        _p = ART / ("xens_%s_w%s" % (_sp, _w)) / "result.json"
        if _p.exists():
            SOURCES.append(("Ours xens VL0 w%s (weight-selection split)" % _w,
                            "xens", _p, _dsn))
# 15.17: r2dtr1c s1 (seed=1) — worse basin, kept for the record
for _sp, _dsn in XENS_DS.items():
    if _sp == "archiveii_embok_clean":
        continue
    ow_if("Ours r2d_tr1c (seed=1, worse basin)",
          "ow_rinalmo_r2dtr1c_b4_s1_step20000_%s" % _sp, _dsn)

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
        # ensemble_eval rows carry f1/precision/recall but not inf — tolerate
        inf = round(sum(r["inf"] for r in per if "inf" in r) /
                    max(1, sum(1 for r in per if "inf" in r)), 4) \
            if per and any("inf" in r for r in per) else None
        mic = pl["micro"]
        if "precision" not in mic and "tp" in mic:  # ensemble_eval layout
            tp, fp, fn = mic["tp"], mic["fp"], mic["fn"]
            mic = dict(mic)
            mic["precision"] = tp / (tp + fp) if tp + fp else 0.0
            mic["recall"] = tp / (tp + fn) if tp + fn else 0.0
        return {"label": label, "dataset": ds,
                "micro_P": round(mic["precision"], 4),
                "micro_R": round(mic["recall"], 4),
                "micro_F1": round(mic["f1"], 4),
                "macro_F1": round(pl["macro"]["f1"], 4) if isinstance(pl.get("macro"), dict) else round(pl["macro_f1"], 4) if "macro_f1" in pl else None,
                "INF": inf, "n": d.get("n_sequences"),
                "source": str(path.parent.name)}
    if kind == "xens":
        d = json.load(open(path))
        pl = d["pair_level"]
        mic = dict(pl["micro"])
        tp, fp, fn = mic.get("tp", 0), mic.get("fp", 0), mic.get("fn", 0)
        if "precision" not in mic:
            mic["precision"] = tp / (tp + fp) if tp + fp else 0.0
        if "recall" not in mic:
            mic["recall"] = tp / (tp + fn) if tp + fn else 0.0
        mac = None
        if isinstance(pl.get("macro"), dict):
            mac = pl["macro"].get("f1")
        return {"label": label, "dataset": ds,
                "micro_P": round(mic["precision"], 4),
                "micro_R": round(mic["recall"], 4),
                "micro_F1": round(mic["f1"], 4),
                "macro_F1": round(mac, 4) if mac is not None else None,
                "INF": None, "n": d.get("n_sequences"),
                "source": str(path.parent.name)}
    if kind == "plana":
        d = json.load(open(path))
        split_names = {"ts0": "TS0", "new": "bpRNA-new", "ts1": "TS1",
                       "hard": "TS-hard", "ts2": "TS2", "ts3": "TS3"}
        rows = []
        for sp, m in d["splits"].items():
            if sp not in split_names:
                continue
            rows.append({"label": label, "dataset": split_names[sp],
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
