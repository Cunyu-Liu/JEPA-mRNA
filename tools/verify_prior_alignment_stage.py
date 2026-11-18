#!/usr/bin/env python3
"""Engineering and identity acceptance for one completed prior-alignment stage."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.prior_alignment import scheduled_unfreeze_blocks, sha256_file


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected_steps = {"smoke": 2, "pilot": 3200, "full": 20000}[args.mode]
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise ValueError(f"{name}: {detail}")

    metas, orders, logs, checkpoint_hashes = {}, {}, {}, {}
    for arm in ("on", "off"):
        trajectory = f"train_{arm}"
        root = args.stage / trajectory
        meta = json.loads((root / "run_meta.json").read_text())
        metas[arm] = meta
        prior = meta["prior_alignment"]
        check(f"{arm}_completed", meta["status"] == "completed", meta["status"])
        check(f"{arm}_identity", prior["training_trajectory"] == trajectory)
        check(f"{arm}_mode", prior["mode"] == args.mode)
        check(f"{arm}_seed", meta["seed"] == args.seed)
        check(f"{arm}_steps", meta["schedule"]["steps"] == expected_steps)
        check(f"{arm}_no_early_stop", not meta["early_stopped"])
        check(f"{arm}_length_nll", meta["objective"]["nll_normalization"] == "length")
        check(f"{arm}_pure_nll", all(meta["objective"][key] == 0 for key in
              ("lambda_distill", "lambda_rlcd", "lambda_cal", "lambda_jepa")))
        check(f"{arm}_storage", str(meta["storage"]["snapshot_directory"]).startswith(
              "/teams/HoulabData_1788502323/liucunyu/"))
        order = _jsonl(root / "batch_order.jsonl")
        check(f"{arm}_batch_rows", len(order) == expected_steps, len(order))
        check(f"{arm}_batch_steps", [row["step"] for row in order] == list(range(1, expected_steps + 1)))
        orders[arm] = order
        log = _jsonl(root / "train_log.jsonl")
        check(f"{arm}_finite_loss", all(math.isfinite(row["loss"]) for row in log))
        check(f"{arm}_finite_grad", all(row["gnorm"] is None or math.isfinite(row["gnorm"]) for row in log))
        for row in log:
            expected_unfrozen = scheduled_unfreeze_blocks(row["step"], 1600, 800, 2, 33)
            check(f"{arm}_unfreeze_{row['step']}", row["unfrozen"] == expected_unfrozen,
                  f"{row['unfrozen']} != {expected_unfrozen}")
        logs[arm] = log
        snapshot = args.stage / "snapshots" / f"{trajectory}_step{expected_steps}.pt"
        check(f"{arm}_snapshot_exists", snapshot.is_file(), snapshot)
        checkpoint_hashes[arm] = sha256_file(snapshot)
    check("same_shared_init", metas["on"]["prior_alignment"]["shared_init_sha256"] ==
          metas["off"]["prior_alignment"]["shared_init_sha256"])
    check("different_terminal_checkpoints", checkpoint_hashes["on"] != checkpoint_hashes["off"])
    for step, (on, off) in enumerate(zip(orders["on"], orders["off"]), 1):
        check(f"batch_identity_{step}", on["sequences"] == off["sequences"])
    if args.mode == "pilot":
        on_by_step = {row["step"]: row for row in logs["on"]}
        off_by_step = {row["step"]: row for row in logs["off"]}
        for step, expected in ((2400, 2), (3200, 4)):
            check(f"pilot_on_unfreeze_{step}", on_by_step[step]["unfrozen"] == expected)
            check(f"pilot_off_unfreeze_{step}", off_by_step[step]["unfrozen"] == expected)
    decode = json.loads((args.stage / "decode_vl0/vl0_selected_summary.json").read_text())
    decode_verification = json.loads((args.stage / "decode_vl0/decode_verification.json").read_text())
    check("decode_verification_pass", decode_verification["status"] == "PASS")
    check("decode_checkpoint_on", decode["train_on_checkpoint_sha256"] == checkpoint_hashes["on"])
    check("decode_checkpoint_off", decode["train_off_checkpoint_sha256"] == checkpoint_hashes["off"])
    check("decode_seed", decode["seed"] == args.seed)
    check("decode_terminal_step", decode["terminal_step"] == expected_steps)
    check("decode_sequence_count", decode["n_sequences"] == (2 if args.mode == "smoke" else 196))
    check("decode_no_external", not decode["old_external_panels_used"])
    check("decode_four_arms", set(decode["arms"]) == {"C0", "C1", "C2", "C3"})
    for arm in ("C0", "C1", "C2", "C3"):
        check(f"decode_stem_{arm}", "micro_stem_f1" in decode["arms"][arm] and
              "macro_stem_f1" in decode["arms"][arm])
    report = {
        "status": "PASS", "scope": "ENGINEERING_AND_IDENTITY_ACCEPTANCE_ONLY",
        "mode": args.mode, "seed": args.seed, "steps": expected_steps,
        "n_checks": len(checks), "checks": checks,
        "shared_init_sha256": metas["on"]["prior_alignment"]["shared_init_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
        "batch_order_identical": True, "old_external_panels_used": False,
        "scientific_route_selected": False,
        "next_gate": "pilot acceptance permits frozen full seeds 0/1/2; it does not permit production or SOTA claims",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n_checks": len(checks),
                      "batch_order_identical": True, "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
