#!/usr/bin/env python
"""Audit v2 (15.22): correct source-layout dispatch.
Layouts on disk:
  - baselines_*.json      : {"split": ..., "baselines": {<name>: {micro_precision, micro_recall, micro_f1, macro_f1, inf, n_sequences}}}
  - ow_/xens_/ens_/ts1hard_/ff20000_/xens2_ dirs : result.json {"pair_level": {"micro": {f1,precision,recall,tp,fp,fn}, "macro": {"f1"}}, "per_sequence": [...], "n_sequences"}
  - plana_* dirs          : plan_a_result.json {"splits": {ts0/new/ts1/hard/ts2/ts3: {"micro": {...}, "macro_f1", "n_sequences"}}}
  - spec/eval_splits_frozen.json : {"frozen_at", "note"?, "splits": {name: {sha256, n, ...}}} (verify actual top-level)
"""
import json
import hashlib
from pathlib import Path

ART = Path("/mnt/cunyuliu/rna-jepa/eval_decision")
TB = Path("/home/cunyuliu/rna-jepa/tables")

errors = []
checked = 0

def err(msg):
    errors.append(msg); print("MISMATCH:", msg)

def ok(msg):
    global checked; checked += 1

def close(a, b, tol=0.0006):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol

METHOD_TO_BASELINE = {
    "ViennaRNA centroid": "vienna_centroid",
    "vienna_centroid": "vienna_centroid",
    "ViennaRNA mfe": "vienna_mfe",
    "ViennaRNA mea": "vienna_mea",
    "Nussinov+Turner (our prior)": "nussinov_turner",
    "nussinov_turner": "nussinov_turner",
    "UFold": "ufold",
    "MXfold2": "mxfold2",
    "eternafold": "eternafold",
    "RNAformer (bprna ckpt)": "rnaformer",
    "RNAformer (inter-family ckpt, paper Tab.4 setting)": "rnaformer_interfam",
}

def read_baseline_cell(fp, method):
    d = json.load(open(fp))
    want = METHOD_TO_BASELINE.get(method)
    bl = d.get("baselines", {})
    if want and want in bl:
        return bl[want]
    # fuzzy fallback
    for name, b in bl.items():
        if name.lower() in method.lower() or method.lower() in name.lower():
            return b
    return None

def read_ow_cell(fp):
    d = json.load(open(fp))
    pl = d.get("pair_level", {})
    mic = dict(pl.get("micro", {}))
    if "precision" not in mic and "tp" in mic:
        tp, fp, fn = mic["tp"], mic["fp"], mic["fn"]
        mic["precision"] = tp / (tp + fp) if tp + fp else 0.0
        mic["recall"] = tp / (tp + fn) if tp + fn else 0.0
        mic["f1"] = (2 * mic["precision"] * mic["recall"] /
                     (mic["precision"] + mic["recall"])) if mic["precision"] + mic["recall"] else 0.0
    mac = pl.get("macro", {}).get("f1") if isinstance(pl.get("macro"), dict) else pl.get("macro_f1")
    return {"P": mic.get("precision"), "R": mic.get("recall"), "F1": mic.get("f1"),
            "macro": mac, "n": d.get("n_sequences")}

def read_plana_cell(fp, ds):
    sp_map = {"TS0": "ts0", "bpRNA-new": "new", "TS1": "ts1", "TS-hard": "hard",
              "TS2": "ts2", "TS3": "ts3"}
    sp = sp_map.get(ds.split(" ")[0])
    if sp is None:
        return None
    d = json.load(open(fp))
    if sp not in d.get("splits", {}):
        return None
    m = d["splits"][sp]
    return {"P": m["micro"].get("precision"), "R": m["micro"].get("recall"),
            "F1": m["micro"].get("f1"), "macro": m.get("macro_f1"),
            "n": m.get("n_sequences")}

# ---------- audit metrics_matrix.md ----------
print("=== metrics_matrix.md measured rows ===")
mm = (TB / "metrics_matrix.md").read_text()
nrows = 0
for ln in mm.split("\n"):
    if not ln.startswith("| "):
        continue
    parts = [p.strip() for p in ln.strip("|").split("|")]
    if len(parts) != 9:
        continue
    method, ds, P, R, F1, mac, INF, n, source = parts
    if parts[0] == "Method" or set(parts[0]) <= set("- "):
        continue
    if "quoted" in method.lower():
        continue
    nrows += 1
    # locate the source file
    fp = None
    kind = None
    if source.endswith(".json"):
        cand = ART / source
        if cand.exists():
            fp, kind = cand, "baseline"
    else:
        c1 = ART / source / "result.json"
        c2 = ART / source / "plan_a_result.json"
        if c2.exists() and not c1.exists():
            fp, kind = c2, "plana"
        elif c1.exists():
            d = json.load(open(c1))
            fp, kind = c1, ("plana_splits" if "splits" in d else "ow")
    if fp is None:
        err(f"[{method} | {ds}] source {source} not found")
        continue
    if kind == "baseline":
        b = read_baseline_cell(fp, method)
        if b is None:
            err(f"[{method} | {ds}] baseline entry not found in {source}")
            continue
        if F1 != "—" and not close(b.get("micro_f1"), float(F1)):
            err(f"[{method} | {ds}] F1 table={F1} disk={b.get('micro_f1')}")
        elif F1 != "—":
            ok(0)
        if P != "—" and not close(b.get("micro_precision"), float(P)):
            err(f"[{method} | {ds}] P table={P} disk={b.get('micro_precision')}")
        elif P != "—":
            ok(0)
        if R != "—" and not close(b.get("micro_recall"), float(R)):
            err(f"[{method} | {ds}] R table={R} disk={b.get('micro_recall')}")
        elif R != "—":
            ok(0)
        if mac != "—" and not close(b.get("macro_f1"), float(mac)):
            err(f"[{method} | {ds}] macro table={mac} disk={b.get('macro_f1')}")
        elif mac != "—":
            ok(0)
    elif kind in ("ow",):
        m = read_ow_cell(fp)
        if F1 != "—" and not close(m["F1"], float(F1)):
            err(f"[{method} | {ds}] F1 table={F1} disk={m['F1']:.4f} ({source})")
        elif F1 != "—":
            ok(0)
        if P != "—" and not close(m["P"], float(P)):
            err(f"[{method} | {ds}] P table={P} disk={m['P']:.4f} ({source})")
        elif P != "—":
            ok(0)
        if R != "—" and not close(m["R"], float(R)):
            err(f"[{method} | {ds}] R table={R} disk={m['R']:.4f} ({source})")
        elif R != "—":
            ok(0)
        if mac != "—" and not close(m["macro"], float(mac)):
            err(f"[{method} | {ds}] macro table={mac} disk={m['macro']} ({source})")
        elif mac != "—":
            ok(0)
    elif kind in ("plana", "plana_splits"):
        m = read_plana_cell(fp, ds)
        if m is None:
            err(f"[{method} | {ds}] split not found in {source}")
            continue
        if F1 != "—" and not close(m["F1"], float(F1)):
            err(f"[{method} | {ds}] F1 table={F1} disk={m['F1']:.4f} ({source})")
        elif F1 != "—":
            ok(0)
        if P != "—" and not close(m["P"], float(P)):
            err(f"[{method} | {ds}] P table={P} disk={m['P']:.4f} ({source})")
        elif P != "—":
            ok(0)
        if R != "—" and not close(m["R"], float(R)):
            err(f"[{method} | {ds}] R table={R} disk={m['R']:.4f} ({source})")
        elif R != "—":
            ok(0)

print(f"  parsed & checked {nrows} measured rows")

# ---------- sota reference values (re-check TS3 with correct filename) ----------
print("=== sota reference values v2 ===")
REF_V2 = [
    ("TS0 rnaformer", "baselines_rnaformer_ref_bprna_ts0.json", 0.7578),
    ("TS1 interfam", "baselines_rnaformer_interfam_ref_pdb_ts1.json", 0.8150),
    ("TS-hard rnaformer", "baselines_rnaformer_bprna_ref_pdb_ts_hard.json", 0.7845),
    ("TS2 interfam", "baselines_rnaformer_interfam_ref_pdb_ts2.json", 0.9043),
    ("TS3 rnaformer", "baselines_rnaformer_ref_pdb_ts3.json", 0.9410),
    ("bpRNA-new ufold", "baselines_ufold_bprna_new.json", 0.6106),
]
for label, src, qv in REF_V2:
    fp = ART / src
    if not fp.exists():
        err(f"{label}: file missing {src}"); continue
    d = json.load(open(fp))
    bl = d.get("baselines", {})
    val = None
    for name in ("rnaformer", "rnaformer_interfam", "ufold"):
        if name in bl:
            val = bl[name]["micro_f1"]; break
    if val is None or not close(val, qv):
        err(f"{label}: table={qv} disk={val}")
    else:
        ok(0); print(f"  ok: {label}={val:.4f}")

# ArchiveII vienna bucket-max
ref = 0.0
for f in ["baselines_archiveii_embok_clean_le100.json",
          "baselines_archiveii_embok_clean_gt100_le200.json",
          "baselines_archiveii_embok_clean_gt200_le400.json",
          "baselines_archiveii_embok_clean_gt400.json"]:
    d = json.load(open(ART / f))
    for name, b in d["baselines"].items():
        if "centroid" in name:
            ref = max(ref, b["micro_f1"])
if not close(ref, 0.7212):
    err(f"ArchiveII vienna ref: table=0.7212 disk={ref:.4f}")
else:
    ok(0); print(f"  ok: ArchiveII vienna bucket-max={ref:.4f}")

# ---------- frozen splits manifest ----------
print("=== eval_splits_frozen.json ===")
frozen = json.load(open("/home/cunyuliu/rna-jepa/spec/eval_splits_frozen.json"))
print("  top-level keys:", list(frozen.keys())[:6])
splits = frozen.get("splits", frozen)
for name, meta in splits.items():
    if not isinstance(meta, dict) or "sha256" not in meta:
        continue
    fp = Path(meta.get("path", "")) if meta.get("path") else Path("/mnt/cunyuliu/rna-jepa/ss_data/jsonl") / f"{name}.jsonl"
    if not fp.exists():
        cands = list(Path("/mnt/cunyuliu/rna-jepa/ss_data/jsonl").glob(f"{name}*.jsonl"))
        if not cands:
            err(f"frozen {name}: file not found"); continue
        fp = cands[0]
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["sha256"]:
        err(f"frozen {name}: sha256 MISMATCH (file changed since freeze!)")
    else:
        ok(0); print(f"  ok: frozen {name} sha256 verified")

# ---------- ArchiveII600 row ----------
print("=== ArchiveII600 tolerant macro ===")
cands = list(ART.glob("arch600*")) + list(ART.glob("*arch600*.json"))
found = False
for c in cands:
    if c.is_dir():
        for j in c.glob("*.json"):
            print("  found:", j)
            found = True
    elif c.suffix == ".json":
        print("  found:", c)
        found = True
if not found:
    print("  NOTE: no arch600 json found under eval_decision/ — row cites 'json, not result.json'; check location manually")

print()
print(f"=== AUDIT v2: {checked} cells ok, {len(errors)} mismatches, {nrows} rows parsed ===")
for e in errors:
    print(" -", e)
raise SystemExit(1 if errors else 0)
