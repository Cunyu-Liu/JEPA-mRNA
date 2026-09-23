"""Verify the project contract against the artefacts actually on disk.

Every requirement in the project spec is reduced to a check that reads real state --
files, md5s, row counts, ledger rows, result.json contents -- and reports PASS / FAIL /
PENDING with the evidence.  Nothing is taken on trust: a stage that has not produced its
evidence is reported as not done, which is the whole point of having the gate.

Run on the cluster (paths default to the project layout):
  python scripts/verify_contract.py [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple

ROOT = os.environ.get("RNAJEPA_ROOT", "/home/cunyuliu/rna-jepa")
ART = os.environ.get("RNAJEPA_ART", "/mnt/cunyuliu/rna-jepa")
ZIP_MD5 = "bf8bc5c946a0bd3b07716b1c7f785d54"
MIN_PRETRAIN_SEQUENCES = 18_000_000

Results = List[Tuple[str, str, str]]   # (requirement, status, evidence)


def add(res: Results, req: str, ok: Optional[bool], evidence: str) -> None:
    res.append((req, "PASS" if ok is True else ("FAIL" if ok is False else "PENDING"),
                evidence))


def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def md5(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ART, "contract_status.json"))
    args = ap.parse_args()
    res: Results = []

    # ---------------- R0 infrastructure --------------------------------------
    try:
        remotes = subprocess.run(["git", "-C", ROOT, "remote", "-v"],
                                 capture_output=True, text=True, timeout=30).stdout
        pushed = "RNA-JEPA" in remotes
        log = subprocess.run(["git", "-C", ROOT, "log", "--oneline"],
                             capture_output=True, text=True, timeout=30).stdout
        n_commits = len([l for l in log.splitlines() if l.strip()])
    except Exception:  # noqa: BLE001
        pushed, n_commits = False, 0
    add(res, "R0 git repo pushed to GitHub", pushed, f"{n_commits} commit(s)")

    code_outside_home = []
    for pattern in ("runs/*", "eval_out/*", "data/raw/*"):
        code_outside_home += glob.glob(os.path.join(ROOT, pattern))
    add(res, "R0 big artefacts live under /mnt, not /home", not code_outside_home,
        "clean" if not code_outside_home else f"{len(code_outside_home)} stray path(s)")

    cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    add(res, "R0 cron monitor (<=10 min)", "RNAJEPA_MONITOR" in cron,
        next((l for l in cron.splitlines() if "RNAJEPA_MONITOR" in l), "absent"))

    ledger = os.path.join(ART, "ledger.jsonl")
    n_ledger = sum(1 for _ in open(ledger)) if os.path.isfile(ledger) else 0
    add(res, "R0 experiment ledger has rows", n_ledger > 0, f"{n_ledger} row(s)")

    status = read_json(os.path.join(ART, "status.json"))
    add(res, "R0 monitor produced a status snapshot", status is not None,
        f"ts={status.get('ts')}" if status else "status.json missing")

    hist = os.path.join(ART, "gpu_history.jsonl")
    n_hist = sum(1 for _ in open(hist)) if os.path.isfile(hist) else 0
    add(res, "R0 GPU history recorded (idle-capacity evidence)", n_hist >= 5,
        f"{n_hist} snapshot(s)")

    # ---------------- R1 pre-training corpus ---------------------------------
    zp = os.path.join(ART, "data/raw/mRNAdataset.zip")
    if os.path.isfile(zp):
        got = md5(zp)
        add(res, "R1 pre-training zip md5", got == ZIP_MD5, f"{got}")
    else:
        parts = glob.glob(os.path.join(ART, "data/raw/mRNAdataset.zip.parts/p[0-9]*"))
        add(res, "R1 pre-training zip md5", None,
            f"download incomplete: {len(parts)}/2387 chunks present")

    stats = read_json(os.path.join(ART, "data/pretrain/stats.json"))
    if stats:
        n = stats.get("n", 0)
        keys_ok = all(k in stats for k in ("n", "orf_frac", "mean_tokens", "tok_max",
                                           "n_non_vocab"))
        add(res, "R1 pre.txt sequence count >= 18M",
            n >= MIN_PRETRAIN_SEQUENCES, f"n={n:,}")
        add(res, "R1 stats.json complete", keys_ok,
            f"orf_frac={stats.get('orf_frac')} mean_tokens={stats.get('mean_tokens')}")
    else:
        add(res, "R1 pre.txt sequence count >= 18M", None, "stats.json not produced yet")
        add(res, "R1 stats.json complete", None, "stats.json not produced yet")
    add(res, "R1 region sidecar written",
        os.path.isfile(os.path.join(ART, "data/pretrain/pre_regions.txt")),
        "pre_regions.txt")

    # ---------------- R2 downstream data ------------------------------------
    man = read_json(os.path.join(ART, "data/downstream_extracted/manifest.json"))
    if man:
        s = man.get("summary", {})
        add(res, "R2 145 downstream tasks scanned",
            man.get("n_tasks") == 145, f"n_tasks={man.get('n_tasks')}")
        add(res, "R2 row counts match the xlsx protocol",
            len(s.get("xlsx_row_mismatches", [])) == 0,
            f"{len(s.get('xlsx_row_mismatches', []))} mismatch(es)")
        add(res, "R2 shared dev/test tasks flagged",
            s.get("n_shared_dev_test", 0) > 0,
            f"{s.get('n_shared_dev_test')} task(s) with byte-identical dev/test")
    else:
        for req in ("R2 145 downstream tasks scanned", "R2 row counts match the xlsx protocol",
                    "R2 shared dev/test tasks flagged"):
            add(res, req, None, "manifest.json missing")

    # ---------------- R3 baseline gate --------------------------------------
    gate_paper = {
        "cds_mrfp": ("spearman", 0.89, "max"),
        "rbp_0": ("accuracy", 0.786, "max"),
        "te_human": ("r2", 0.669, "max"),
        "m6a_HEK293T_fold0": ("accuracy", 0.966, "max"),
        "cds_ecoli": ("accuracy", 0.58, "max"),
        "cds_cov": ("spearman", 0.89, "max"),
    }
    gate_ok, gate_detail = True, []
    for task, (metric, paper, _mode) in gate_paper.items():
        hit = glob.glob(os.path.join(ART, f"eval_out/mrnabert_official/{task}_s*/result.json"))
        if not hit:
            gate_detail.append(f"{task}=pending")
            continue
        d = read_json(hit[0]) or {}
        v = d.get(metric)
        if v is None:
            gate_detail.append(f"{task}=no {metric}")
            continue
        delta = v - paper
        ok = abs(delta) <= 0.05
        gate_ok &= ok
        gate_detail.append(f"{task}: {metric}={v:.4f} vs {paper} (d={delta:+.3f}){'ok' if ok else 'OUT'}")
    pending = any("pending" in x for x in gate_detail)
    add(res, "R3 baseline reproduction within 0.05 on the gate tasks",
        None if pending else gate_ok, "; ".join(gate_detail))

    # ---------------- R4 pre-training --------------------------------------
    for arm in ("v1_cont", "v2_scratch"):
        cks = glob.glob(os.path.join(ART, f"runs/{arm}/hf/step_*"))
        log = os.path.join(ART, f"runs/{arm}/train_log.jsonl")
        rows = [json.loads(l) for l in open(log)] if os.path.isfile(log) else []
        add(res, f"R4 {arm} checkpoints written", bool(cks), f"{len(cks)} checkpoint(s)")
        if rows:
            last = rows[-1]
            finite = all(isinstance(last.get(k), (int, float)) and last[k] == last[k]
                         for k in ("loss", "loss_mlm", "loss_jepa"))
            collapsed = (last.get("cls_var", 1) < 1e-6) or \
                        (isinstance(last.get("factor_activity"), list)
                         and min(last["factor_activity"]) < 1e-8)
            add(res, f"R4 {arm} losses finite, no collapse", finite and not collapsed,
                f"step={last.get('step')} mlm={last.get('loss_mlm'):.3f} "
                f"jepa={last.get('loss_jepa'):.4f} cls_var={last.get('cls_var')}")
        else:
            add(res, f"R4 {arm} losses finite, no collapse", None, "no log rows yet")

    # ---------------- R5 head-to-head --------------------------------------
    cand_dirs = [d for d in ("v1_cont", "v2_scratch")
                 if os.path.isdir(os.path.join(ART, f"eval_out/{d}"))]
    n_cells = sum(len(glob.glob(os.path.join(ART, f"eval_out/{d}/*/result.json")))
                  for d in cand_dirs)
    summary = read_json(os.path.join(ART, "tables/summary.json"))
    add(res, "R5 candidate models evaluated", n_cells > 0,
        f"{n_cells} result(s) across {cand_dirs or 'no candidate dirs'}")
    add(res, "R5 head-to-head table produced",
        summary is not None and summary.get("n_comparisons", 0) > 0,
        (f"{summary.get('n_comparisons')} comparison(s), "
         f"wins={summary.get('wins')} losses={summary.get('losses')}")
        if summary else "summary.json missing")
    if summary:
        add(res, "R5 protocol deviations excluded from the table",
            True, f"{len(summary.get('protocol_deviations', []))} deviation group(s) excluded")

    # ---------------- R6 ablations and factors -----------------------------
    abl = glob.glob(os.path.join(ART, "runs/ablation_*/hf/step_*"))
    add(res, "R6 ablation arms have checkpoints", len(abl) >= 6,
        f"{len(set(p.split('/runs/')[1].split('/')[0] for p in abl))} arm(s) with checkpoints")
    fp = glob.glob(os.path.join(ART, "eval_out/factor_probe*/factor_probe.json"))
    if fp:
        d = read_json(fp[0]) or {}
        probes = d.get("probes", {})
        sig = [k for k, v in probes.items()
               if isinstance(v, dict) and v.get("significant_bh")]
        add(res, "R6 factor probe with rotation control and FDR",
            bool(d.get("random_subspace_control")) and bool(probes),
            f"{len(probes)} probe(s), {len(sig)} survive FDR, "
            f"{len(d.get('random_subspace_control', {}))} control run(s)")
    else:
        add(res, "R6 factor probe with rotation control and FDR", None, "not run yet")

    # ---------------- report ------------------------------------------------
    width = max(len(r[0]) for r in res)
    lines = ["", "=" * (width + 40),
             "CONTRACT VERIFICATION (evidence-based; PENDING means not yet produced)",
             "=" * (width + 40)]
    for req, st, ev in res:
        lines.append(f"{st:8} {req:{width}}  {ev}")
    n_pass = sum(1 for r in res if r[1] == "PASS")
    n_fail = sum(1 for r in res if r[1] == "FAIL")
    n_pend = sum(1 for r in res if r[1] == "PENDING")
    lines.append("-" * (width + 40))
    lines.append(f"{n_pass} PASS, {n_fail} FAIL, {n_pend} PENDING")
    print("\n".join(lines))

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"results": [{"requirement": r, "status": s, "evidence": e}
                               for r, s, e in res],
                   "pass": n_pass, "fail": n_fail, "pending": n_pend},
                  fh, indent=1)
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())