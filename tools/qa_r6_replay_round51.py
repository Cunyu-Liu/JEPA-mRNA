import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(left, right, tolerance=1e-9):
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def slope(xs, ys):
    if len(xs) < 2 or len(set(xs)) == 1:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)


def verify(result_path, data_path, expected_unfrozen=12, backbone="structrfm",
           expected_head_lr=1e-4, expected_backbone_lr=1e-5):
    result_path = Path(result_path)
    raw = json.loads(result_path.read_text())
    assert raw["status"] == "PASS_R_REPLAY_CUDA"
    assert raw.get("backbone", "structrfm") == backbone
    assert raw["scope"] == "best_weights_with_terminal_optimizer_scheduler_and_stream_not_best_time_optimizer"
    assert raw["scientific_conclusion"] is False
    assert raw["device"].startswith("cuda")
    assert raw["resume_weights_equal_best"] is True
    assert raw["weights_unchanged"] is True
    assert raw["optimizer_step_called"] is False
    assert raw["scheduler_unchanged"] is True
    assert raw["stream_unchanged"] is True
    assert raw["n_unfrozen"] == expected_unfrozen
    assert raw["data_sha256"] == sha256_file(data_path)
    assert raw["best_sha256"] == sha256_file(Path(raw["run_dir"]) / "best.pt")
    assert raw["resume_sha256"] == sha256_file(Path(raw["run_dir"]) / "resume.pt")
    records = {}
    with open(data_path) as handle:
        for index, line in enumerate(handle):
            if line.strip():
                records[index] = json.loads(line)["seq"].upper().replace("T", "U")
    all_rows = []
    for batch in raw["batches"]:
        sum_terms = {key: 0.0 for key in ("total", "nll", "distill", "rlcd", "cal")}
        length_terms = dict(sum_terms)
        for row in batch["per_sequence"]:
            sequence = records[row["record_index"]]
            assert len(sequence) == row["length"]
            assert hashlib.sha256(sequence.encode()).hexdigest() == row["sequence_sha256"]
            for name in sum_terms:
                expected = row["sum"][name]
                if name == "nll":
                    expected /= row["length"]
                elif name == "total":
                    expected += row["sum"]["nll"] / row["length"] - row["sum"]["nll"]
                assert close(row["length_normalized"][name], expected)
                sum_terms[name] += row["sum"][name] / len(batch["per_sequence"])
                length_terms[name] += row["length_normalized"][name] / len(batch["per_sequence"])
            assert row["nll_grad_norm_relative_error"] <= 0.005
            all_rows.append(row)
        for name in sum_terms:
            assert close(batch["losses"]["sum"][name], sum_terms[name])
            assert close(batch["losses"]["length"][name], length_terms[name])
        assert batch["gradient_checks"]["max_nll_length_scaled_error"] <= 0.01
        assert batch["gradient_checks"]["max_aux_gradient_scaled_difference"] <= 0.01
        for variant, groups in batch["shadow_adam"].items():
            for group in groups:
                assert group["update_rms"] >= 0
                assert group["m_rms"] >= 0
                assert group["v_rms"] >= 0
                if variant.endswith("shadow_backbone_1e-6") and group["role"] == "backbone":
                    assert group["lr"] == 1e-6
                elif group["role"] == "backbone":
                    assert group["lr"] == expected_backbone_lr
                else:
                    assert group["lr"] == expected_head_lr
    expected_slopes = {
        "nll_loss_sum_vs_length": slope([r["length"] for r in all_rows], [r["sum"]["nll"] for r in all_rows]),
        "nll_loss_length_normalized_vs_length": slope([r["length"] for r in all_rows], [r["length_normalized"]["nll"] for r in all_rows]),
        "nll_grad_norm_sum_vs_length": slope([r["length"] for r in all_rows], [r["nll_grad_norm_sum"] for r in all_rows]),
        "nll_grad_norm_length_normalized_vs_length": slope([r["length"] for r in all_rows], [r["nll_grad_norm_length"] for r in all_rows]),
    }
    for key, value in expected_slopes.items():
        if value is None:
            assert raw["length_slopes"][key] is None
        else:
            assert close(raw["length_slopes"][key], value, 1e-8)
    return {
        "status": "PASS_INDEPENDENT_R_REPLAY_QA",
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "seed": raw["seed"],
        "backbone": backbone,
        "n_unfrozen": raw["n_unfrozen"],
        "best_step": raw["best_step"],
        "resume_step": raw["resume_step"],
        "batches": len(raw["batches"]),
        "sequences": len(all_rows),
        "code_commit": raw["code_commit"],
        "code_sha256": raw["code_sha256"],
        "max_nll_length_gradient_error": raw["max_nll_length_gradient_error"],
        "max_aux_gradient_difference": raw["max_aux_gradient_difference"],
        "length_slopes": expected_slopes,
        "scientific_conclusion": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--data", default="/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-unfrozen", type=int, default=12)
    parser.add_argument("--backbone", choices=("structrfm", "rinalmo"), default="structrfm")
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    args = parser.parse_args()
    report = verify(args.result, args.data, args.expected_unfrozen, args.backbone,
                    args.head_lr, args.backbone_lr)
    temporary = Path(args.output).with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
