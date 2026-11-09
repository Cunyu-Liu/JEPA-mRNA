import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = {"plana_a": 42, "plana_b": 2026, "r2d": 3407}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(predicted, truth):
    pred = {tuple(pair) for pair in predicted}
    gt = {tuple(pair) for pair in truth}
    return len(pred & gt), len(pred - gt), len(gt - pred)


def micro_score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def close(left, right):
    assert math.isclose(left, right, rel_tol=0, abs_tol=1e-12), (left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data/rna_complex_round49/xens2_complex"
    recipe = json.loads((data / "recipe.json").read_text())
    assert recipe["seeds"] == SEEDS and recipe["w_plana"] == 0.7
    out = args.root / "eval_decision/xens2_complex"
    result_paths = [out / f"rfam_repeat{index}.json" for index in range(3)]
    assert all(path.read_bytes() == result_paths[0].read_bytes() for path in result_paths)
    result = json.loads(result_paths[0].read_text())
    raw = [json.loads(line) for line in Path(recipe["rfam_data"]).read_text().splitlines() if line.strip()]
    assert result["n_sequences"] == len(raw) == 10790 and result["n_skipped"] == 0
    assert len(result["per_sequence"]) == len(raw)
    tp = fp = fn = 0
    macro = []
    for source, row in zip(raw, result["per_sequence"]):
        assert row["name"] == source.get("name", "")
        a, b, c = counts(row["pred_pairs"], source["pairs"])
        tp += a
        fp += b
        fn += c
        expected = 2 * a / (2 * a + b + c) if a + b + c else 1.0
        close(row["f1"], expected)
        macro.append(expected)
    precision, recall, f1 = micro_score(tp, fp, fn)
    micro = result["pair_level"]["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (tp, fp, fn)
    close(micro["f1"], f1)
    close(micro["precision"], precision)
    close(micro["recall"], recall)
    close(result["pair_level"]["macro"]["f1"], statistics.fmean(macro))
    sources = {str(path.relative_to(args.root)): sha256(path) for path in result_paths}
    for role, seed in SEEDS.items():
        run = Path(recipe["members"][role])
        meta = json.loads((run / "run_meta.json").read_text())
        assert meta.get("seed", meta.get("config", {}).get("seed")) == seed
        assert meta["converged"] and meta["early_stopped"]
        assert meta["status"] == "completed" and meta["environment"]["resolved_device"].startswith("cuda")
        checkpoint = run / "best.pt"
        sources[str(checkpoint.relative_to(args.root))] = sha256(checkpoint)
    assert Path(result["plana_ckpt"]) == Path(recipe["members"]["plana_a"]) / "best.pt"
    assert Path(result["plana_ckpt2"]) == Path(recipe["members"]["plana_b"]) / "best.pt"
    assert Path(result["r2d_ckpt"]) == Path(recipe["members"]["r2d"]) / "best.pt"
    sources[str(Path(recipe["rfam_data"]))] = sha256(recipe["rfam_data"])
    sources[str((data / "recipe.json").relative_to(args.root))] = sha256(data / "recipe.json")
    report = {"status": "PASS_XENS2_COMPLEX_RFAM_INDEPENDENT_QA", "complete": True,
              "ensemble": "Plan-A 42/2026 equal average + R2D 3407; w=0.7",
              "n_sequences": len(raw), "repeats": 3, "byte_identical_repeats": True,
              "micro_f1": f1, "precision": precision, "recall": recall,
              "macro_f1": statistics.fmean(macro), "population_sd": 0.0,
              "tp": tp, "fp": fp, "fn": fn,
              "reference_from_prior_accepted_results": {"prism_xens2_f1": 0.6514,
                                                         "bpfold_f1": 0.7871860033},
              "single_model_claim": False, "file_sha256": sources}
    path = out / "rfam_independent_qa.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "micro_f1": f1,
                      "macro_f1": report["macro_f1"], "tp": tp, "fp": fp, "fn": fn,
                      "sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
