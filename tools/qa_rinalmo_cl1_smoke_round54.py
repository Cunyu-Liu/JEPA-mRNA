import argparse
import hashlib
import json
from pathlib import Path

import torch

from qa_loss_optimizer_training_round51 import read_jsonl, sha256_file, stable_load
from rinalmo_preflight import VOCAB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["arm"] == "round54_cl1_smoke_s42" and meta["seed"] == 42
    assert meta["status"] == "completed" and meta["steps_completed"] == 2
    assert meta["stop_reason"] == "step_cap" and not meta["converged"]
    assert meta["environment"]["resolved_device"].startswith("cuda")
    assert meta["scorer"] == "resnet2d" and meta["ripe"]["mode"] is None
    assert meta["objective"]["cl1_token_mask"] == 0.15
    assert meta["objective"]["cl1_views"] == 2 and meta["objective"]["lambda_jepa"] == 0.0
    assert all(meta["objective"][key] == 1.0 for key in
               ("lambda_nll", "lambda_distill", "lambda_rlcd", "lambda_cal"))
    logs = read_jsonl(run / "train_log.jsonl")
    assert [row["step"] for row in logs] == [1]
    assert all(row["cl1_views"] == 2 for row in logs)
    first_hash = read_jsonl(run / "compute_ledger.jsonl")[0]["sequence_sha256"][0]
    sequence = next(row["seq"] for row in read_jsonl("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
                    if hashlib.sha256(row["seq"].encode()).hexdigest() == first_hash)
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in sequence] + [VOCAB["<eos>"]]
    expected = []
    masks = []
    for step in (1,):
        counts = []
        for view in (0, 1):
            generator = torch.Generator().manual_seed(42 + 2 * step + view)
            mask = torch.rand(len(ids), generator=generator) < 0.15
            mask[0] = mask[-1] = False
            counts.append(int(mask.sum()))
            masks.append(mask)
        expected.append(counts)
    assert [row["cl1_masked_tokens"] for row in logs] == expected
    assert not torch.equal(masks[0], masks[1])
    assert all(value > 0 for counts in expected for value in counts)
    assert all(set(row["terms"]) == {"nll", "distill", "rlcd", "cal"} for row in logs)
    resume = stable_load(run / "resume.pt")
    assert resume["step"] == 2 and resume["stream_state"]
    report = {"status": "PASS_CL1_TWO_VIEW_CUDA_ENGINEERING",
              "run": meta["arm"], "steps": 2, "masked_counts": expected,
              "train_log": logs, "run_meta_sha256": sha256_file(run / "run_meta.json"),
              "resume_sha256": sha256_file(run / "resume.pt"),
              "data_sha256": sha256_file("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl"),
              "deployment_change": False, "contrastive_loss": False,
              "scientific_conclusion": False}
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "masked_counts": expected}, indent=2))


if __name__ == "__main__":
    main()
