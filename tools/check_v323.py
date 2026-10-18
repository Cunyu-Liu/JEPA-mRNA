#!/usr/bin/env python3
"""v3.23 checker: re-derive the newly added numbers from disk."""
import json
from pathlib import Path

DRAFT = Path("/root/rna_jev/paper/preprint_draft.md")
text = DRAFT.read_text(encoding="utf-8")
E = "/root/rna_jepa_data/eval_decision"
checks = []
def ck(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

# speed v4 B=256
d = json.load(open(f"{E}/speed_v4_4090_fp16_b256.json"))
ck("124.8 B=256 seq/s", abs(d["seq_per_s"] - 124.78) < 0.1, str(d["seq_per_s"]))
ck("B=256 F1 0.667971", d["f1"] == 0.667971)
for B in [128, 192, 256]:
    dd = json.load(open(f"{E}/speed_v4_4090_fp16_b{B}.json"))
    ck(f"B={B} tp/fp/fn identical", (dd["tp"], dd["fp"], dd["fn"]) == (24406, 13778, 10485))
ck("'124.8' in draft", "124.8" in text)
ck("'124.78' NOT misquoted", "124.78 sequences/s" in text)

# confidence board
S = json.load(open(f"{E}/confidence_board/board_conf_summary.json"))
expected = {
    "bprna_ts0": (0.1604, 0.0315, 0.708, 1281),
    "bprna_new": (0.2011, 0.0677, 0.416, 5375),
    "ref_pdb_ts1": (0.121, 0.1111, 0.559, 62),
    "ref_pdb_ts2": (0.0821, 0.1851, 0.337, 39),
    "ref_pdb_ts3": (0.1088, 0.1841, 0.362, 19),
    "ref_pdb_ts_hard": (0.1243, 0.1219, 0.676, 27),
    "archiveii_embok_clean": (0.1176, 0.0605, 0.716, 2505),
    "testsetb": (0.1004, 0.0445, 0.694, 428),
}
for sp, (raw, cal, rho, n) in expected.items():
    s = S[sp]
    ck(f"{sp} raw {raw}", abs(s["ece_raw"] - raw) < 5e-4, str(s["ece_raw"]))
    ck(f"{sp} cal {cal}", abs(s["ece_cal"] - cal) < 5e-4, str(s["ece_cal"]))
    ck(f"{sp} n {n}", s["n"] == n, str(s["n"]))

fails = [c for c in checks if not c[1]]
for name, ok, det in checks:
    if not ok:
        print("FAIL", name, det)
print(f"{len(checks)-len(fails)}/{len(checks)} passed")
raise SystemExit(1 if fails else 0)
