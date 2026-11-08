#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path


REQUIRED = ["plana", "r2d", "xens2", "rinalmo_ft", "mxfold2", "rnaformer",
            "bpfold", "efold", "nucleicbert_ft", "kirigami_fast", "spotrna"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def throughput(document):
    if "throughput_median_seq_per_s" in document:
        return document["throughput_median_seq_per_s"]
    return statistics.median(row["sequences_per_second"] for row in document["repeats"])


def validate_gpu(document, data_sha256):
    assert document["status"] in {"PASS", "PASS_STRUCTRFM_RAW_SPEED", "PASS_NUCLEICBERT_RAW_SPEED"}
    assert document.get("data_sha256") or document.get("protocol", {}).get("data_sha256")
    if "protocol" in document and "n_raw" in document["protocol"]:
        protocol = document["protocol"]
        assert protocol["data_sha256"] == data_sha256 and protocol["n_raw"] == 79
        assert protocol["device"] == "cuda" and protocol["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert protocol["batch_size"] == 1 and protocol["warmup_sequences_per_repeat"] == 1
        assert protocol["repeats"] == 3 and protocol["cached_embeddings"] is False
        assert all(row["n"] == 79 and row["n_timed"] == 78 for row in document["runs"])
    elif document.get("system") == "nucleicbert_ft" and "throughput_median_seq_per_s" in document:
        assert document["data_sha256"] == data_sha256 and document["n_sequences"] == 79
        assert document["device"] == "cuda" and document["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert document["batch_size"] == 1 and document["warmup_sequences_per_repeat"] == 1
        assert document["cached_embeddings"] is False and document["cached_tokenization"] is False
        assert len(document["runs"]) == 3 and all(row["n"] == 79 and row["n_timed"] == 78 for row in document["runs"])
    elif "protocol" in document and isinstance(document["protocol"], dict):
        protocol = document["protocol"]
        assert document["status"] == "PASS" and protocol["device_type"] == "cuda"
        assert protocol["device"] == "NVIDIA GeForce RTX 4090" and protocol["data_sha256"] == data_sha256
        assert protocol["n_sequences"] == 79 and protocol["batch_size"] == 1
        assert protocol["repeats"] == 3 and protocol["warmup_batches_per_repeat"] == 1
        assert protocol["cached_embeddings"] is False and protocol["n_length_filtered"] == 0
        assert protocol["sample_scope"] == "full_source_file"
        assert all(row["n_sequences"] == 79 and row["n_timed"] == 78 for row in document["runs"])
    else:
        assert document["clean_speed_protocol"] is True and document["data_sha256"] == data_sha256
        assert document["device"] == "cuda" and document["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert document["n_sequences"] == 79 and document["batch_size"] == 1
        assert document["warmup_sequences_per_repeat"] == 1 and document["cached_embeddings"] is False
        assert len(document["repeats"]) == 3 and all(row["n_timed"] == 78 for row in document["repeats"])
    return throughput(document)


def ranking(rows):
    return sorted(rows, key=lambda row: (-row["throughput_seq_per_s"], row["model"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_path = args.root / "data/structrfm_pretrain_clean/bprna_ts0.jsonl"
    data_sha256 = sha256(data_path)
    decision_path = args.root / "audits/round48/complex_d1_decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["complete"] and decision["allow_lineardesign"]
    prefix = decision["next_base"]["run_prefix"]
    candidates = []
    for seed in range(3):
        meta = json.loads((args.root / f"runs/{prefix}{seed}/run_meta.json").read_text())
        assert meta["status"] == "completed" and meta["converged"] and meta["early_stopped"]
        candidates.append((meta["best_validation_loss"], seed))
    primary_loss, primary_seed = min(candidates)
    base = args.root / "eval_decision/speed_round49"
    rows = []
    sources = {str(data_path.relative_to(args.root)): data_sha256,
               str(decision_path.relative_to(args.root)): sha256(decision_path)}
    for name in ["ours"] + REQUIRED:
        filename = f"selected_s{primary_seed}_clean_ts0.json" if name == "ours" else f"{name}_clean_ts0.json"
        path = base / filename
        document = json.loads(path.read_text())
        value = validate_gpu(document, data_sha256)
        sources[str(path.relative_to(args.root))] = sha256(path)
        rows.append({"model": "single_structrfm_jev" if name == "ours" else name,
                     "throughput_seq_per_s": value, "source": str(path.relative_to(args.root))})
    ordered = ranking(rows)
    primary = next(row for row in rows if row["model"] == "single_structrfm_jev")
    fastest_baseline = max(row["throughput_seq_per_s"] for row in rows if row["model"] != "single_structrfm_jev")
    seed_rows = []
    for seed in range(3):
        path = base / f"selected_s{seed}_clean_ts0.json"
        document = json.loads(path.read_text())
        seed_rows.append({"seed": seed, "throughput_seq_per_s": validate_gpu(document, data_sha256),
                          "source_sha256": sha256(path)})
    cpu_path = base / "eternafold_clean_ts0.json"
    cpu = json.loads(cpu_path.read_text())
    assert cpu["status"] == "PASS" and cpu["clean_speed_protocol"] is True
    assert cpu["data_sha256"] == data_sha256 and cpu["n_sequences"] == 79 and len(cpu["repeats"]) == 3
    assert cpu["device"].startswith("CPU") and cpu["cross_device_gpu_rankable"] is False
    sources[str(cpu_path.relative_to(args.root))] = sha256(cpu_path)
    payload = {"status": "PASS_MATCHED_CLEAN_TS0_SPEED", "n_raw": 79, "data_sha256": data_sha256,
               "primary_seed": primary_seed, "primary_best_vl0": primary_loss,
               "selection_rule": "primary checkpoint is the minimum best VL0 seed in the D0/D1-selected run prefix; never the fastest seed",
               "coverage": REQUIRED + ["single_structrfm_jev"], "coverage_complete": True,
               "gpu_ranking": ordered, "primary": primary, "fastest_baseline_seq_per_s": fastest_baseline,
               "speed_sota_claim": primary["throughput_seq_per_s"] > fastest_baseline,
               "seed_sensitivity": seed_rows,
               "native_cpu_descriptive": {"model": "eternafold", "throughput_seq_per_s": throughput(cpu),
                                          "cross_device_gpu_rankable": False},
               "protocol_notes": ["all GPU rows: same 79 raw clean TS0 sequences, B=1, one warmup per repeat, three repeats, exclusive RTX4090",
                                  "EternaFold is native single-thread CPU and is not included in the GPU ranking",
                                  "RNAformer and NucleicBERT alternate weights share their respective architectures; listed checkpoints are the speed representatives"],
               "source_hashes": sources}
    atomic_json(args.out, payload)
    assert json.loads(args.out.read_text()) == payload
    print(json.dumps({"status": payload["status"], "primary": primary,
                      "fastest_baseline": fastest_baseline, "speed_sota_claim": payload["speed_sota_claim"]}, indent=2))


if __name__ == "__main__":
    main()
