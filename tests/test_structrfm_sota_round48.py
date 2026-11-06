import json
import sys

from tools.compare_structrfm_sota_round48 import CHANRG, MAIN, main


def test_sota_comparison_keeps_baseline_classes_separate(tmp_path):
    root = tmp_path / "root"
    output = root / "eval_decision/structrfm_frozen"
    output.mkdir(parents=True)
    splits = list(MAIN) + ["rfam_fam"] + CHANRG
    (output / "summary.json").write_text(json.dumps({
        "splits": {split: {"f1_mean": 0.8} for split in splits}}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "status": "PASS_COMPLETE_OWN_CLEAN_LEADERBOARD",
        "systems": {
            "xens2": {split: {"status": "PASS", "f1_mean": 0.9 if split in {"bprna_new", "ref_pdb_ts2"} else 0.7}
                      for split in MAIN.values()},
            "bpfold": {split: {"status": "PASS", "f1_mean": 0.75} for split in MAIN.values()},
        }}))
    rfam = tmp_path / "rfam.json"
    rfam.write_text(json.dumps({"repeats": [{"metrics": {"micro_f1": 0.787}} for _ in range(3)]}))
    chanrg = tmp_path / "chanrg.json"
    chanrg.write_text(json.dumps({"models": [
        {"name": "xens2", "values": [0.6] * 4, "evidence": "measured"},
        {"name": "quoted", "values": [0.9] * 4, "evidence": "quoted"},
    ]}))
    sys.argv = ["compare", "--root", str(root), "--experiment", "structrfm_frozen",
                "--baseline", str(baseline), "--rfam-baseline", str(rfam),
                "--chanrg-baseline", str(chanrg)]
    main()
    result = json.loads((output / "sota_comparison.json").read_text())
    assert result["main_sota_any_wins"] == 6
    assert not result["main_all_8_sota_any"]
    assert result["main_sota_external_single_wins"] == 8
    assert result["rfam"]["sota"]
    assert result["chanrg_measured_wins"] == 4
    assert all(not row["sota_quoted"] for row in result["chanrg"].values())
