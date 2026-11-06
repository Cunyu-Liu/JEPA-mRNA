import hashlib
import json
import sys

from tools.aggregate_structrfm_frozen_round48 import main


def test_aggregate_three_seed_three_repeat(tmp_path):
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
            checkpoint = root / f"runs/structrfm_frozen_jev_s{seed}/best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            out = root / f"eval_decision/structrfm_frozen/s{seed}/{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            row = {"name": "toy", "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}
            prediction = [{"name": "toy", "pred_pairs": [[0, 4]]}]
            out.write_text(json.dumps({
                "status": "PASS", "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "repeats": 3, "raw_n": 1, "n": 1, "n_long_unsupported": 0,
                "micro": {"f1": 1.0, "precision": 1.0, "recall": 1.0,
                          "tp": 1, "fp": 0, "fn": 0},
                "per_sequence": [row], "predictions": [prediction, prediction, prediction]}))
    out = root / "eval_decision/structrfm_frozen/summary.json"
    sys.argv = ["aggregate", "--root", str(root), "--out", str(out)]
    main()
    summary = json.loads(out.read_text())
    assert summary["status"] == "PASS_STRUCTRFM_FROZEN_ABC_13_SPLITS_3_SEEDS_3_REPEATS"
    assert len(summary["splits"]) == 13
    assert summary["splits"]["bprna_ts0"]["f1_mean"] == 1.0
    assert summary["splits"]["bprna_ts0"]["f1_sd"] == 0.0
