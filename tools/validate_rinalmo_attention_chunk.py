import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "eval"))
from eval.ss.metrics import PairLevelMetrics
from rinalmo_preflight import VOCAB, load_encoder
from rnajepa.harness import nussinov_map, valid_pair_mask
from rnajepa.rinalmo_attention import enable_query_chunked_attention
from xens2_eval import load_plana, plana_scores

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward(model, sequence):
    ids = [VOCAB["<cls>"]] + [VOCAB[base] for base in sequence] + [VOCAB["<eos>"]]
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    return output.last_hidden_state.float()


def forward_batch(model, sequences):
    ids = [[VOCAB["<cls>"]] + [VOCAB[base] for base in sequence] + [VOCAB["<eos>"]] for sequence in sequences]
    width = max(map(len, ids))
    tokens = torch.tensor([row + [VOCAB["<pad>"]] * (width - len(row)) for row in ids], device="cuda")
    mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=tokens, attention_mask=mask)
    return output.last_hidden_state.float()


def similarity(expected, actual, length, chunk=512):
    difference = (expected - actual).abs()
    relative = float(difference.norm() / expected.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(expected.flatten(), actual.flatten(), dim=0))
    return {"length": length, "chunk": chunk, "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()), "relative_l2": relative, "cosine": cosine,
            "bitwise_equal": bool(torch.equal(expected, actual)), "finite": bool(torch.isfinite(actual).all()),
            "relative_l2_pass": relative < 0.01, "cosine_pass": cosine > 0.9999}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rng = np.random.default_rng(3407)
    encoder, report = load_encoder("cuda", dtype=torch.float32)
    encoder.eval()
    sequences = ["".join(rng.choice(list("ACGU"), size=length)) for length in (12, 129, 257, 1025, 4097)]
    expected_outputs = [forward(encoder, sequence) for sequence in sequences]
    original_repeat = forward(encoder, sequences[3])
    assert torch.equal(expected_outputs[3], original_repeat)
    padding_expected = forward_batch(encoder, [sequences[0], sequences[1]])
    enable_query_chunked_attention(encoder, 512)
    checks = []
    for sequence, expected in zip(sequences, expected_outputs):
        actual = forward(encoder, sequence)
        check = similarity(expected, actual, len(sequence))
        checks.append(check)
        assert check["finite"] and check["relative_l2_pass"] and check["cosine_pass"], check
    chunked_repeat = forward(encoder, sequences[3])
    repeat_actual = forward(encoder, sequences[3])
    assert torch.equal(chunked_repeat, repeat_actual)
    padding_actual = forward_batch(encoder, [sequences[0], sequences[1]])
    padding_check = similarity(padding_expected, padding_actual, -1)
    padding_check["case"] = "batch_with_padding"
    checks.append(padding_check)
    assert padding_check["relative_l2_pass"] and padding_check["cosine_pass"], padding_check
    del encoder, expected_outputs, original_repeat, padding_expected, padding_actual, chunked_repeat, repeat_actual
    torch.cuda.empty_cache()

    encoder, head = load_plana("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/p2/pilot_v1/H0/best.pt", "cuda")
    records = [json.loads(line) for line in Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl").open()]
    originals = []
    for record in records:
        sequence = record["seq"]
        scores = plana_scores(encoder, head, sequence, "cuda").cpu().numpy()
        mask = valid_pair_mask(sequence)
        predicted = {tuple(pair) for pair in nussinov_map(scores, mask)}
        originals.append((sequence, record["pairs"], predicted, PairLevelMetrics.from_pairs(predicted, record["pairs"], len(sequence), mask).f1))
    enable_query_chunked_attention(encoder, 512)
    pair_checks = []
    for sequence, truth, original, original_f1 in originals:
        scores = plana_scores(encoder, head, sequence, "cuda").cpu().numpy()
        mask = valid_pair_mask(sequence)
        predicted = {tuple(pair) for pair in nussinov_map(scores, mask)}
        intersection = len(original & predicted)
        agreement = 1.0 if not original and not predicted else 2 * intersection / (len(original) + len(predicted))
        f1 = PairLevelMetrics.from_pairs(predicted, truth, len(sequence), mask).f1
        pair_checks.append({"length": len(sequence), "pair_agreement": agreement,
                            "exact_pair_set": original == predicted,
                            "original_f1": original_f1, "chunked_f1": f1})
    pair_summary = {"n_sequences": len(pair_checks),
                    "mean_pair_agreement": float(np.mean([row["pair_agreement"] for row in pair_checks])),
                    "exact_pair_set_fraction": float(np.mean([row["exact_pair_set"] for row in pair_checks])),
                    "original_macro_f1": float(np.mean([row["original_f1"] for row in pair_checks])),
                    "chunked_macro_f1": float(np.mean([row["chunked_f1"] for row in pair_checks]))}
    pair_summary["absolute_macro_f1_difference"] = abs(pair_summary["original_macro_f1"] - pair_summary["chunked_macro_f1"])
    assert pair_summary["mean_pair_agreement"] >= 0.99, pair_summary
    assert pair_summary["absolute_macro_f1_difference"] <= 0.001, pair_summary
    del encoder, head, originals
    torch.cuda.empty_cache()

    encoder, report = load_encoder("cuda", dtype=torch.float32)
    encoder.eval()
    enable_query_chunked_attention(encoder, 512)
    long_checks = []
    for split in ("train", "holdout"):
        records = [json.loads(line) for line in (ROOT / f"{split}_contacts.jsonl").open()]
        record = max(records, key=lambda row: len(row["sequence"]))
        torch.cuda.reset_peak_memory_stats()
        actual = forward(encoder, record["sequence"])
        check = {"split": split, "name": record["name"], "length": len(record["sequence"]),
                 "shape": list(actual.shape), "finite": bool(torch.isfinite(actual).all()),
                 "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
        assert check["finite"] and actual.shape[1] == len(record["sequence"]) + 2
        long_checks.append(check)
        print(json.dumps(check), flush=True)
    result = {"status": "PASS_RINALMO_QUERY_CHUNK_ENGINEERING",
              "scientific_scope": "mathematically equivalent full-sequence attention with bounded numerical kernel differences and pair-output validation; not bitwise equivalence, long-sequence accuracy, or a 19000nt beam experiment",
              "query_chunk": 512, "full_sequence_preserved": True,
              "acceptance": {"relative_l2_max": 0.01, "cosine_min": 0.9999,
                             "mean_pair_agreement_min": 0.99, "absolute_macro_f1_difference_max": 0.001},
              "checks": checks, "pair_summary": pair_summary, "pair_checks": pair_checks, "long_checks": long_checks,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "weights_sha256": sha256(Path(report["weights"]) / "model.safetensors"),
              "source_sha256": {"tools/validate_rinalmo_attention_chunk.py": sha256(__file__),
                                "src/rnajepa/rinalmo_attention.py": sha256(REPO / "src/rnajepa/rinalmo_attention.py")}}
    (ROOT / "attention_chunk_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "pair_summary": pair_summary,
                      "long_checks": long_checks}, indent=2))


if __name__ == "__main__":
    main()
