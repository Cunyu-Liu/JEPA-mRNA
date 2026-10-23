#!/usr/bin/env python3
"""CHANRG four-split (Test/GenA/GenC/GenF) OOD evaluation for our systems.

Board protocol (r25 pre-registered):
- Our single-model r2d (rinalmo_r2dtr1c_b4_s0 @200k frozen-embedding recipe) and
  the xens2 board ensemble (plana s0+s1 bucket @ s1w=0.5 x r2d @ w=0.7) are
  evaluated with evaluate_decision.py on each CHANRG split (50-200nt, pk-excluded,
  T->U, ACGU-only), --prior-weight -1, VL0 Platt calib, head-chunk 8.
- Baselines: ViennaRNA (centroid/mfe/mea), nussinov+Turner — physics-only,
  CPU-side; learning baselines (UFold, RNAformer, RiNALMo-ft) run via their
  adapters if weights are on-disk.

Outputs: /root/rna_jepa_data/eval_decision/chanrg_<split>/<system>/result.json
plus a summary board json.
"""
import argparse, json, os, subprocess, sys

PY = "/root/miniconda3/envs/lucaone/bin/python"
J = "/root/rna_jepa_data/ss_data/jsonl"
OUT = "/root/rna_jepa_data/eval_decision"
CK = "/root/rna_jepa_data/weights_20261013/ckpts"
SPLITS = ["chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]

def run(cmd, log):
    with open(log, "a") as f:
        f.write("+ " + " ".join(cmd) + "\n"); f.flush()
        return subprocess.call(cmd, stdout=f, stderr=f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="r2d,xens2")
    ap.add_argument("--out-root", default=OUT)
    args = ap.parse_args()
    sysl = args.systems.split(",")
    env = dict(os.environ)
    env["PYTHONPATH"] = "/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools"

    for sp in SPLITS:
        for system in sysl:
            outdir = f"{args.out_root}/chanrg_{sp[len('chanrg_'):]}/{system}"
            res = f"{outdir}/result.json"
            if os.path.exists(res):
                print(f"[skip] {res} exists"); continue
            os.makedirs(outdir, exist_ok=True)
            log = f"{outdir}/run.log"
            if system == "r2d":
                rc = run([PY, "/home/rna_jev/eval/ss/evaluate_decision.py",
                          "--checkpoint", f"{CK}/rinalmo_r2dtr1c_b4_s0_step20000.pt",
                          "--data", f"{J}/{sp}.jsonl",
                          "--out", outdir, "--tag", f"chanrg_{system}_{sp}",
                          "--prior-weight", "-1",
                          "--calib-data", f"{J}/bprna_vl0.jsonl",
                          "--embedding-split", sp,
                          "--head-chunk", "8", "--device", "cuda"], log)
            elif system == "xens2":
                rc = run([PY, "/home/rna_jev/tools/xens2_eval.py",
                          "--plana-ckpt", f"{CK}/plana_giga_tr1c_s0_step20000.pt",
                          "--plana-ckpt2", f"{CK}/plana_giga_tr1c_s1_step20000.pt",
                          "--r2d-ckpt", f"{CK}/rinalmo_r2dtr1c_b4_s0_step20000.pt",
                          "--data", f"{J}/{sp}.jsonl",
                          "--embedding-split", sp,
                          "--out", res,
                          "--w-plana", "0.7", "--plana-s1-weight", "0.5"], log)
            else:
                print(f"[unknown system] {system}"); continue
            print(f"[done rc={rc}] {system} {sp}")

if __name__ == "__main__":
    main()
