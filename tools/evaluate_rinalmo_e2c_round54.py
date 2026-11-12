import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
sys.path.insert(0, str(ROOT / "bin"))
import physics_dp
from rnajepa.harness import valid_pair_mask
from structrfm_decode_round48 import decode_scores
from xens2_eval import load_plana, plana_scores


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(predicted, expected, span=None):
    if span is not None:
        predicted = {(i, j) for i, j in predicted if j - i >= span}
        expected = {(i, j) for i, j in expected if j - i >= span}
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407])
    args = parser.parse_args()
    seed = args.seed
    records = [json.loads(line) for line in Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl").read_text().splitlines()]
    checkpoint = ROOT / f"runs/round54_b0_s{seed}/best.pt"
    encoder, head = load_plana(checkpoint, "cuda")
    output = []
    for index, record in enumerate(records):
        sequence = record["seq"].upper().replace("T", "U")
        expected = {tuple(pair) for pair in record["pairs"]}
        full = plana_scores(encoder, head, sequence, "cuda").cpu().numpy()
        no_prior = plana_scores(encoder, head, sequence, "cuda", prior_weight_override=0).cpu().numpy()
        mask = valid_pair_mask(sequence)
        neural_residual = np.where(mask, no_prior - (full - no_prior), -1e4).astype(np.float32)
        original, _, _ = decode_scores(np.where(mask, full, -np.inf), mask, "nussinov")
        assert np.isfinite(neural_residual).all()
        energy, structure, table = physics_dp.decode(sequence, neural_residual.astype(np.float32))
        enhanced = {(i - 1, j - 1) for i, j in enumerate(table) if j > i}
        assert all(mask[i, j] for i, j in enhanced)
        output.append({"index": index, "length": len(sequence),
                       "original": {name: counts(set(original), expected, span) for name, span in (("all", None), ("span30", 30), ("span100", 100))},
                       "physics": {name: counts(enhanced, expected, span) for name, span in (("all", None), ("span30", 30), ("span100", 100))},
                       "physics_objective": float(energy), "dot_bracket": structure,
                       "n_original": len(original), "n_physics": len(enhanced)})
        if (index + 1) % 25 == 0:
            print(f"[E2c seed={seed}] {index + 1}/{len(records)}", flush=True)
    result = {"status": "PASS_E2C_VL0", "seed": seed, "n_sequences": len(records),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "records": output, "test_set_used": False,
              "input": "original calibrated score minus original Turner component, then explicit Turner loop states"}
    path = ROOT / f"audits/round54/e2c_s{seed}_vl0.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "seed": seed, "n_sequences": len(records)}, indent=2))


if __name__ == "__main__":
    main()
