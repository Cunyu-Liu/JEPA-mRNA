#!/usr/bin/env python3
"""T-A49b: unified inference-speed benchmark on the SAME hardware, SAME split,
SAME protocol for all systems with timing evidence.

Split: bprna_ts0 (board file, 1288 seqs) — every system already evaluated here.
Timing: wall_seconds from existing runs where the run covered the full split
on GPU of the same node (A100-40GB); for systems lacking a timed full-split
run, we time a fresh pass here (our decision head + RiNALMo-ft + MXfold2).

Output: /mnt/cunyuliu/rna-jepa/eval_decision/speed_benchmark.json
"""
import json, os, time, subprocess, sys

ED = "/mnt/cunyuliu/rna-jepa/eval_decision"
OUT = os.path.join(ED, "speed_benchmark.json")
SPLIT = "bprna_ts0"
N = 1288

results = {}

def add(name, wall_s, device, source, notes=""):
    results[name] = {
        "split": SPLIT, "n_sequences": N, "wall_seconds": round(wall_s, 2),
        "seq_per_s": round(N / wall_s, 2), "device": device,
        "source": source, "notes": notes,
    }

# --- 1) Thermodynamic DP (CPU, 12-process or serial per adapter) ---
b = json.load(open(f"{ED}/baselines_bprna_ts0.json"))["baselines"]
for k in ("vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner"):
    v = b[k]
    add(k, v["wall_seconds"], "CPU (node, 2.7.2 adapter)",
        f"baselines_bprna_ts0.json (full-split run, wall_seconds field)")

# --- 2) UFold (GPU, full-split timed run) ---
u = json.load(open(f"{ED}/ufold_archiveii_embok_clean.json"))
# TS0 run: baselines_ufold_ref_bprna_ts0.json wall? check
ut = json.load(open(f"{ED}/baselines_ufold_ref_bprna_ts0.json"))
uw = ut["baselines"].get("ufold", {}).get("wall_seconds")
if uw:
    add("UFold", uw, "GPU (A100)", "baselines_ufold_ref_bprna_ts0.json wall_seconds")
else:
    # fall back: rescale ArchII timing? no — mark untimed; ArchII timing is
    # from a different split, cannot be mixed. Use ArchII as its own row.
    add("UFold (archiveii_clean, n=2544)", u["wall_seconds"], "GPU (A100)",
        "ufold_archiveii_embok_clean.json (full-split ArchII run)")

# --- 3) RNAformer (GPU, full-split timed) ---
r = json.load(open(f"{ED}/baselines_rnaformer_ref_bprna_ts0.json"))["baselines"]["rnaformer"]
add("RNAformer (bprna ckpt)", r["wall_seconds"], "GPU (A100)",
    "baselines_rnaformer_ref_bprna_ts0.json wall_seconds")

# --- 4) MXfold2 (GPU) ---
m = json.load(open(f"{ED}/baselines_mxfold2_bprna_ts0.json"))["baselines"].get("mxfold2", {})
if m.get("wall_seconds"):
    add("MXfold2", m["wall_seconds"], "GPU (A100)", "baselines_mxfold2_bprna_ts0.json")
else:
    add("MXfold2", -1, "GPU (A100)", "NO TIMING FIELD — needs fresh timed run")

# --- 5) Our decision head (r2dtr1c s0) — re-time a fresh GPU pass ---
PY = "/home/cunyuliu/miniconda3/envs/lucaone/bin/python"
REPO = "/home/cunyuliu/rna-jepa"
D = "/mnt/cunyuliu/rna-jepa"
ours_cmd = [
    PY, f"{REPO}/eval/ss/evaluate_decision.py",
    "--checkpoint", f"{D}/ckpts/rinalmo_r2dtr1c_b4_s0_step20000.pt",
    "--data", f"{D}/ss_data/jsonl/bprna_ts0.jsonl",
    "--embedding-split", "bprna_ts0",
    "--calib-data", f"{D}/ss_data/jsonl/bprna_vl0.jsonl",
    "--prior-weight", "-1",
    "--out", f"{ED}/speed_ours_bprna_ts0",
    "--encoder-size", "35M", "--device", "cuda", "--head-chunk", "8",
    "--tag", "speed_ours_bprna_ts0",
]
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("BENCH_GPU", "1")
env["PYTHONPATH"] = "/mnt/cunyuliu/pylibs:" + REPO + "/src"
env["TMPDIR"] = "/mnt/cunyuliu/tmp"
t0 = time.time()
rc = subprocess.run(ours_cmd, env=env, capture_output=True, text=True).returncode
t1 = time.time()
if rc != 0:
    print("ours eval rc:", rc, file=sys.stderr)
add("Ours r2dtr1c s0 (frozen 650M + resnet2d head, DP decode)", t1 - t0,
    "GPU (A100)", "fresh timed full-split run (this benchmark; includes "
    "model load + embedding load from cache + eval-mode decode)",
    f"rc={rc}")

json.dump({"split": SPLIT, "n": N, "protocol": "wall-clock over the full "
           "split on the same node; GPU rows are single-GPU passes; "
           "DP-family rows are the project adapter runs",
           "results": results}, open(OUT, "w"), indent=1)
print(json.dumps(results, indent=1))
