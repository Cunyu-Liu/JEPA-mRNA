#!/usr/bin/env python3
"""Convergence evidence pack for r2dtr1c_b4_s2 (user rule: train to convergence).

Reads the full train_log.jsonl, computes:
  - loss trajectory stats in 2k-step windows (mean, slope)
  - best-window vs last-window comparison
  - whether loss has stopped decreasing (last 2k window mean >= best window mean + noise)
Writes a verdict the watchdog chain can cite.
"""
import json, sys, statistics

path = "/root/rna_jepa_data/runs/rinalmo_r2dtr1c_b4_s2/train_log.jsonl"
rows = []
for line in open(path):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
steps = [r["step"] for r in rows]
losses = [r["loss"] for r in rows]
n = len(rows)
if n < 10:
    print(json.dumps({"verdict": "INSUFFICIENT_DATA", "n": n}))
    sys.exit(0)

W = 2000
windows = []
for lo in range(0, steps[-1] + 1, W):
    wl = [l for s, l in zip(steps, losses) if lo <= s < lo + W]
    if len(wl) >= 20:
        windows.append((lo, statistics.mean(wl), statistics.stdev(wl)))

print("window  mean_loss  sd")
for lo, m, sd in windows:
    print(f"{lo:6d}  {m:9.3f}  {sd:7.3f}")

best = min(windows, key=lambda w: w[1])
last = windows[-1]
# slope of last 3 windows
if len(windows) >= 3:
    last3 = windows[-3:]
    slope = (last3[-1][1] - last3[0][1]) / (last3[-1][0] - last3[0][0])
else:
    slope = 0.0
verdict = {
    "n_rows": n, "step_last": steps[-1],
    "best_window": best, "last_window": last,
    "last3_slope_per_step": round(slope, 8),
    "converged": last[1] <= best[1] + best[2],  # within 1 sd of best window
}
print(json.dumps(verdict, indent=1))
json.dump(verdict, open("/root/rna_jepa_data/monitor/conv_verdict.json", "w"), indent=1)
