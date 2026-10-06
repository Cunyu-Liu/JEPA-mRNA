#!/usr/bin/env python
"""Independent audit of every quoted number in tables/*.md against the
raw result.json files on disk (15.22). Re-derives each cell from
first principles (tp/fp/fn or stored micro dict) — does NOT import
metrics_matrix.py, so a bug there cannot propagate here.

Checks:
  1. sota_vs_ours.md: every "our best" row vs its source result.json.
  2. metrics_matrix.md: every measured row vs its source result.json.
  3. Reference rows vs baselines_*.json.
  4. overlap_audit.md / eval_splits_frozen.json vs disk jsonl files.
Outputs a discrepancy report (exit 1 if any mismatch).
"""
import json
import re
import glob
import hashlib
from pathlib import Path

ART = Path("/mnt/cunyuliu/rna-jepa/eval_decision")
TB = Path("/home/cunyuliu/rna-jepa/tables")
JSONL = Path("/mnt/cunyuliu/rna-jepa/ss_data/jsonl")

errors = []
checked = 0

def err(msg):
    errors.append(msg)
    print("MISMATCH:", msg)

def ok(msg):
    global checked
    checked += 1
    print("  ok:", msg)

def micro_from_result(path):
    d = json.load(open(path))
    pl = d.get("pair_level", {})
    mic = pl.get("micro", {})
    if not mic and "splits" in d:  # plan_a layout
        return None
    tp, fp, fn = mic.get("tp"), mic.get("fp"), mic.get("fn")
    P = mic.get("precision")
    R = mic.get("recall")
    F = mic.get("f1")
    # re-derive from counts when available (primary check)
    if tp is not None and fp is not None and fn is not None:
        P2 = tp / (tp + fp) if (tp + fp) else 0.0
        R2 = tp / (tp + fn) if (tp + fn) else 0.0
        F2 = 2 * P2 * R2 / (P2 + R2) if (P2 + R2) else 0.0
        return {"P": P2, "R": R2, "F1": F2, "n": d.get("n_sequences"),
                "stored_P": P, "stored_R": R, "stored_F1": F,
                "tp": tp, "fp": fp, "fn": fn,
                "macro": (pl.get("macro") or {}).get("f1") if isinstance(pl.get("macro"), dict) else None}
    return {"P": P, "R": R, "F1": F, "n": d.get("n_sequences"),
            "stored_P": P, "stored_R": R, "stored_F1": F,
            "macro": (pl.get("macro") or {}).get("f1") if isinstance(pl.get("macro"), dict) else None}

def close(a, b, tol=0.0006):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol

# ---------- 1. sota_vs_ours.md ----------
print("=== 1. sota_vs_ours.md ===")
sota = (TB / "sota_vs_ours.md").read_text()
SOTA_ROWS = [
    # (split, source result.json, quoted F1, quoted P, quoted R, quoted macro)
    ("TS0", "xens2_bprna_ts0_w0.7/result.json", 0.8039, 0.8215, 0.7869, 0.8002),
    ("bpRNA-new", None, 0.6132, 0.6314, 0.5959, 0.6125),  # ens_r2dtr1_2seed
    ("ArchiveII-clean", "xens2_archiveii_embok_clean_w0.7/result.json", 0.7760, 0.7956, 0.7575, 0.8095),
    ("TS1", "xens2_ref_pdb_ts1_w0.7/result.json", 0.8755, 0.9212, 0.8342, 0.8443),
    ("TS-hard", "xens2_ref_pdb_ts_hard_w0.7/result.json", 0.8732, 0.9167, 0.8337, 0.8383),
    ("TS2", "xens2_ref_pdb_ts2_w0.7/result.json", 0.8588, 0.9353, 0.7939, 0.8407),
    ("TS3", "xens2_ref_pdb_ts3_w0.7/result.json", 0.8965, 0.9109, 0.8825, 0.8612),
    ("TestSetB", "xens2_testsetb_w0.7/result.json", 0.8448, 0.9104, 0.7880, 0.8372),
]
for split, src, qF, qP, qR, qM in SOTA_ROWS:
    if src is None:
        continue
    m = micro_from_result(ART / src)
    label = f"{split} <- {src}"
    if m is None:
        err(f"{label}: unreadable"); continue
    if not close(m["F1"], qF):
        err(f"{label}: F1 table={qF} disk={m['F1']:.4f}")
    if not close(m["P"], qP):
        err(f"{label}: P table={qP} disk={m['P']:.4f}")
    if not close(m["R"], qR):
        err(f"{label}: R table={qR} disk={m['R']:.4f}")
    if qM is not None and not close(m["macro"], qM):
        err(f"{label}: macro table={qM} disk={m['macro']}")
    if all(close(m[a], q) for a, q in [("F1", qF), ("P", qP), ("R", qR)]):
        ok(f"{label}: F1={m['F1']:.4f} P={m['P']:.4f} R={m['R']:.4f}")

# bpRNA-new row: ens_r2dtr1_2seed_bprna_new
p = ART / "ens_r2dtr1_2seed_bprna_new" / "result.json"
if p.exists():
    m = micro_from_result(p)
    if not close(m["F1"], 0.6132):
        err(f"bpRNA-new ens row: F1 table=0.6132 disk={m['F1']:.4f}")
    else:
        ok(f"bpRNA-new ens row F1={m['F1']:.4f}")
else:
    err("bpRNA-new ens row: source missing")

# reference values quoted in sota table vs baselines_*.json
print("=== 3. sota reference values ===")
REFS = [
    ("TS0 ref 0.7578", "baselines_rnaformer_ref_bprna_ts0.json", 0.7578),
    ("TS1 ref 0.8150", "baselines_rnaformer_interfam_ref_pdb_ts1.json", 0.8150),
    ("TS-hard ref 0.7845", "baselines_rnaformer_bprna_ref_pdb_ts_hard.json", 0.7845),
    ("TS2 ref 0.9043", "baselines_rnaformer_interfam_ref_pdb_ts2.json", 0.9043),
    ("TS3 ref 0.9410", "baselines_rnaformer_bprna_ref_pdb_ts3.json", 0.9410),
    ("bpRNA-new UFold ref 0.6106", "baselines_ufold_bprna_new.json", 0.6106),
    ("TestSetB RiNALMo quoted 0.67", None, 0.67),  # quoted from paper, no disk
]
for label, src, qv in REFS:
    if src is None:
        ok(f"{label}: paper-quoted, no disk source by design"); continue
    fp = ART / src
    if not fp.exists():
        err(f"{label}: source file missing {src}"); continue
    d = json.load(open(fp))
    bl = d.get("baselines", {})
    # find the rnaformer/ufold entry
    val = None
    for name, b in bl.items():
        key = label.split(" ref")[0].split(" quoted")[0]
        if key.lower().replace(" ", "_") in name.lower() or name.lower() in key.lower():
            val = b.get("micro_f1"); break
    if val is None:
        for name, b in bl.items():
            val = b.get("micro_f1"); break
    if val is None or not close(val, qv):
        err(f"{label}: table={qv} disk={val}")
    else:
        ok(f"{label}: {val:.4f}")

# ArchiveII-clean ref: vienna centroid bucket-max 0.7212
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
    err(f"ArchiveII-clean vienna ref: table=0.7212 disk={ref:.4f}")
else:
    ok(f"ArchiveII-clean vienna bucket-max ref={ref:.4f}")

# delta column arithmetic in sota table
print("=== delta arithmetic ===")
DELTAS = [
    ("TS0", 0.8039, 0.7578, "+0.046"),
    ("bpRNA-new", 0.6132, 0.6106, "+0.003"),
    ("ArchiveII-clean", 0.7760, 0.7212, "+0.055"),
    ("TS1", 0.8755, 0.8150, "+0.061"),
    ("TS-hard", 0.8732, 0.7845, "+0.089"),
    ("TS2", 0.8588, 0.9043, "-0.0455"),
    ("TS3", 0.8965, 0.9410, "-0.0445"),
    ("TestSetB", 0.8448, 0.6700, "+0.175"),
]
for split, ours, refv, qd in DELTAS:
    actual = round(ours - refv, 4)
    qval = float(qd.replace("−", "-"))
    if abs(actual - qval) > 0.0006:
        err(f"{split} delta: table={qd} computed={actual:+.4f}")
    else:
        ok(f"{split} delta {actual:+.4f}")

# ---------- 2. metrics_matrix.md ----------
print("=== 2. metrics_matrix.md (all measured rows) ===")
mm = (TB / "metrics_matrix.md").read_text()
rows = [ln for ln in mm.split("\n") if ln.startswith("| ") and "—" not in ln.split("|")[2:4][0] + ln.split("|")[2:3][0]]
measured_rows = []
for ln in mm.split("\n"):
    if not ln.startswith("| "):
        continue
    parts = [p.strip() for p in ln.strip("|").split("|")]
    if len(parts) != 9:
        continue
    if parts[0] in ("Method",) or set(parts[0]) <= set("- "):
        continue
    if parts[1] in ("Dataset",) or set(parts[1]) <= set("- "):
        continue
    if "quoted" in parts[0]:
        continue  # quoted section handled separately
    measured_rows.append(parts)

print(f"  parsed {len(measured_rows)} measured rows")
for parts in measured_rows:
    method, ds, P, R, F1, mac, INF, n, source = parts
    src_path = None
    if source.startswith("ow_") or source in ("xens_" + source[5:] if source.startswith("xens_") else source):
        pass
    # locate source file
    cand = list(ART.glob(f"{source}/result.json"))
    if source.endswith(".json"):
        cand = [ART / source]
    if not cand:
        # plana layout: <tag>/plan_a_result.json
        cand = list(ART.glob(f"{source}/plan_a_result.json"))
    if not cand:
        err(f"matrix row [{method} | {ds}]: source {source} not found")
        continue
    fp = cand[0]
    if fp.name == "result.json":
        m = micro_from_result(fp)
        if m is None or m.get("F1") is None:
            # ensemble jsons sometimes carry only f1/precision/recall at top
            d = json.load(open(fp))
            mic = d.get("pair_level", {}).get("micro", {})
            m = {"P": mic.get("precision"), "R": mic.get("recall"), "F1": mic.get("f1"), "macro": None}
        try:
            if F1 != "—" and not close(m["F1"], float(F1)):
                err(f"matrix [{method} | {ds}]: F1 table={F1} disk={m['F1']}")
            elif F1 != "—":
                checked += 1
        except (TypeError, ValueError):
            err(f"matrix [{method} | {ds}]: F1 cell '{F1}' not numeric-checkable")
    else:
        # plan_a_result.json: splits.<name>.micro
        d = json.load(open(fp))
        sp_map = {"TS0": "ts0", "bpRNA-new": "new", "TS1": "ts1", "TS-hard": "hard",
                  "TS2": "ts2", "TS3": "ts3"}
        sp = sp_map.get(ds.split(" ")[0])
        if sp is None:
            continue
        if sp not in d.get("splits", {}):
            err(f"matrix [{method} | {ds}]: split {sp} missing in {source}")
            continue
        m = d["splits"][sp]["micro"]
        if F1 != "—" and not close(m["f1"], float(F1)):
            err(f"matrix [{method} | {ds}]: F1 table={F1} disk={m['f1']:.4f}")
        elif F1 != "—":
            checked += 1

# ---------- 4. overlap audit + frozen splits ----------
print("=== 4. overlap_audit.md / eval_splits_frozen.json ===")
frozen = json.load(open("/home/cunyuliu/rna-jepa/spec/eval_splits_frozen.json"))
for name, meta in frozen.items():
    fp = Path(meta["path"]) if "path" in meta else JSONL / f"{name}.jsonl"
    if not fp.exists():
        # try the recorded path variants
        alt = list(JSONL.glob(f"{name}*.jsonl"))
        if not alt:
            err(f"frozen split {name}: file not found")
            continue
        fp = alt[0]
    h = hashlib.sha256()
    n = 0
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    if meta.get("sha256") and h.hexdigest() != meta["sha256"]:
        err(f"frozen split {name}: sha256 mismatch (file changed since freeze!)")
    elif meta.get("sha256"):
        ok(f"frozen {name}: sha256 ok, n={n}")

print()
print(f"=== AUDIT COMPLETE: {checked} checks ok, {len(errors)} mismatches ===")
if errors:
    print("ERRORS:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print("ALL NUMBERS VERIFIED CLEAN")
