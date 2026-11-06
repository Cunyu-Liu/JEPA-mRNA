import hashlib
import json
import sys

import pytest

from tools.aggregate_structrfm_frozen_round48 import main


@pytest.mark.parametrize("full_model", [False, True])
def test_aggregate_three_seed_three_repeat(tmp_path, full_model):
    root = tmp_path / "root"
    clean = root / "data/structrfm_pretrain_clean"
    splits = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1",
              "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb",
              "rfam_fam", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
    for split in splits:
        clean.mkdir(parents=True, exist_ok=True)
        data = root / f"data/structrfm_pretrain_clean/{split}.jsonl"
        data.write_text("{}\n")
        (clean / f"{split}.audit.json").write_text(json.dumps({
            "clean_n": 1, "clean_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}))
        for seed in range(3):
            run = "structrfm_jev_s" if full_model else "structrfm_frozen_jev_s"
            experiment = "structrfm_full" if full_model else "structrfm_frozen"
            checkpoint = root / f"runs/{run}{seed}/best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            out = root / f"eval_decision/{experiment}/s{seed}/{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            span = {"tp": 1, "fp": 0, "fn": 0, "f1": 1.0}
            row = {"name": "toy", "length": 6, "f1": 1.0, "inf": 1.0,
                   "structure_conf": 1.0, "span_30": span, "span_100": span,
                   "tp": 1, "fp": 0, "fn": 0}
            prediction = [{"name": "toy", "pred_pairs": [[0, 4]]}]
            out.write_text(json.dumps({
                "status": "PASS", "device": "cuda",
                "encoder_load": {"strict_identity": True},
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "repeats": 3, "raw_n": 1, "n": 1, "n_long_unsupported": 0,
                "micro": {"f1": 1.0, "precision": 1.0, "recall": 1.0,
                          "tp": 1, "fp": 0, "fn": 0}, "macro_f1": 1.0,
                "micro_inf": 1.0, "macro_inf": 1.0,
                "calibration": {"n": 10, "ece": 0.0, "brier": 0.0, "nll": 0.0},
                "per_sequence": [row], "predictions": [prediction, prediction, prediction]}))
    out = root / f"eval_decision/{experiment}/summary.json"
    sys.argv = ["aggregate", "--root", str(root), "--out", str(out)]
    if full_model:
        sys.argv.append("--full-model")
    main()
    summary = json.loads(out.read_text())
    status = "FULL" if full_model else "FROZEN"
    assert summary["status"] == f"PASS_STRUCTRFM_{status}_ABC_13_SPLITS_3_SEEDS_3_REPEATS"
    assert len(summary["splits"]) == 13
    assert summary["splits"]["bprna_ts0"]["f1_mean"] == 1.0
    assert summary["splits"]["bprna_ts0"]["f1_sd"] == 0.0
    assert summary["splits"]["bprna_ts0"]["macro_f1_mean"] == 1.0
    assert summary["splits"]["bprna_ts0"]["cells"][0]["tp"] == 1
    assert summary["splits"]["bprna_ts0"]["cells"][0]["length_buckets"][0] == {
        "n": 1, "tp": 1, "fp": 0, "fn": 0, "micro_f1": 1.0, "macro_f1": 1.0}
    assert summary["splits"]["bprna_ts0"]["ece_mean"] == 0.0
