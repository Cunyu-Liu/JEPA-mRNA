import argparse
import hashlib
import json
from pathlib import Path


MAIN = {"bprna_ts0": "bprna_ts0", "bprna_new": "bprna_new",
        "archiveii": "archiveii_embok_clean", "ref_pdb_ts1": "ref_pdb_ts1",
        "ref_pdb_ts2": "ref_pdb_ts2", "ref_pdb_ts3": "ref_pdb_ts3",
        "ref_pdb_ts_hard": "ref_pdb_ts_hard", "testsetb": "testsetb"}
CHANRG = ["chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best(rows):
    eligible = [(name, value) for name, value in rows if value is not None]
    return max(eligible, key=lambda row: row[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", choices=["structrfm_frozen", "structrfm_full", "structrfm_s3"], required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rfam-baseline", type=Path, required=True)
    parser.add_argument("--chanrg-baseline", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.root / f"eval_decision/{args.experiment}/summary.json"
    ours = json.loads(result_path.read_text())
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "PASS_COMPLETE_OWN_CLEAN_LEADERBOARD"
    rows = {}
    for split, baseline_name in MAIN.items():
        f1 = ours["splits"][split]["f1_mean"]
        candidates = {name: values.get(baseline_name, {}).get("f1_mean")
                      for name, values in baseline["systems"].items()
                      if values.get(baseline_name, {}).get("status") == "PASS"}
        any_name, any_f1 = best(candidates.items())
        single_name, single_f1 = best((name, value) for name, value in candidates.items()
                                      if name != "xens2")
        rows[split] = {"ours_f1_mean": f1, "best_any_baseline": any_f1,
                       "best_any_name": any_name, "delta_any": f1 - any_f1,
                       "sota_any": f1 > any_f1, "best_external_single": single_f1,
                       "best_external_single_name": single_name,
                       "delta_external_single": f1 - single_f1,
                       "sota_external_single": f1 > single_f1}
    rfam_source = json.loads(args.rfam_baseline.read_text())
    rfam_f1 = float(sum(repeat["metrics"]["micro_f1"] for repeat in rfam_source["repeats"]) /
                    len(rfam_source["repeats"]))
    rfam_ours = ours["splits"]["rfam_fam"]["f1_mean"]
    chanrg_source = json.loads(args.chanrg_baseline.read_text())
    chanrg = {}
    for index, split in enumerate(CHANRG):
        f1 = ours["splits"][split]["f1_mean"]
        measured_name, measured_f1 = best((model["name"], model["values"][index])
            for model in chanrg_source["models"] if model["evidence"] != "quoted")
        quoted_name, quoted_f1 = best((model["name"], model["values"][index])
            for model in chanrg_source["models"] if model["evidence"] == "quoted")
        chanrg[split] = {"ours_f1_mean": f1, "best_measured": measured_f1,
                         "best_measured_name": measured_name, "delta_measured": f1 - measured_f1,
                         "sota_measured": f1 > measured_f1, "best_quoted": quoted_f1,
                         "best_quoted_name": quoted_name, "delta_quoted": f1 - quoted_f1,
                         "sota_quoted": f1 > quoted_f1}
    output = {"status": "PASS_STRUCTRFM_INDEPENDENT_COMPARISON",
              "experiment": args.experiment, "result_sha256": sha256(result_path),
              "baseline_sha256": sha256(args.baseline),
              "rfam_baseline_sha256": sha256(args.rfam_baseline),
              "chanrg_baseline_sha256": sha256(args.chanrg_baseline),
              "main": rows,
              "main_sota_any_wins": sum(row["sota_any"] for row in rows.values()),
              "main_all_8_sota_any": all(row["sota_any"] for row in rows.values()),
              "main_sota_external_single_wins": sum(row["sota_external_single"] for row in rows.values()),
              "rfam": {"ours_f1_mean": rfam_ours, "bpfold_f1_mean": rfam_f1,
                       "delta": rfam_ours - rfam_f1, "sota": rfam_ours > rfam_f1},
              "chanrg": chanrg,
              "chanrg_measured_wins": sum(row["sota_measured"] for row in chanrg.values()),
              "protocol": "strict greater-than; own-clean native leaderboards; xens2 retained as any-baseline but excluded from external-single; quoted CHANRG separate"}
    path = args.root / f"eval_decision/{args.experiment}/sota_comparison.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
