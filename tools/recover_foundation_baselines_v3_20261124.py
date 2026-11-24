#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
OUTPUT_DEFAULT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v3_foundation")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")

SOURCES = {
    "rinalmo_author_bprna_ft": {
        "family": "RiNALMo",
        "path": ART / "rinalmo_ft/raw/bprna_new.json",
        "sha256": "6bf0080d656a3859e36492c7497e64f50fbcc5d7abf091e3546c8ce3c0e6baee",
        "kind": "predictions",
        "origin": "author_checkpoint",
        "model_form": "single_model_not_ensemble",
        "training_protocol": "author-released RiNALMo-giga secondary-structure fine-tune on bpRNA; two-block ResNet head; checkpoint threshold; raw full-length B=1 replay",
        "comparability": "Official author checkpoint evaluated on project raw labels; three recorded deterministic repeats are inference repeats, not training seeds or an ensemble.",
    },
    "nucleicbert_author_secstr_full_ft": {
        "family": "NucleicBERT",
        "path": ART / "nucleicbert_frozen/raw/bprna_new.json",
        "sha256": "e1a6d1f4f355648926505a3fa1e5bf5d7204b2eb037c47bfc13479bc46077a1c",
        "kind": "predictions",
        "origin": "author_checkpoint",
        "model_form": "single_model_not_ensemble",
        "training_protocol": "author secstr.ckpt epoch=99 global_step=258000; official secstr.yml has frozen=False (full fine-tune), RNAStrAlign/TR0-style secondary-structure training, B=2, threshold=0.5, first-598-nt protocol",
        "comparability": "The historical filename/arm label says frozen_official, but the official config says frozen=False; this is not the project frozen-encoder arm. Three deterministic inference repeats are not an ensemble.",
    },
    "nucleicbert_project_frozen_encoder_tr1c": {
        "family": "NucleicBERT",
        "path": ART / "nucleicbert_ft/raw/bprna_new.json",
        "sha256": "eb6dbea04a07db9140408d355e686629e7279c5005aed84a4fe5abcf3395cd34",
        "kind": "predictions",
        "origin": "project_downstream_checkpoint",
        "model_form": "single_model_not_ensemble",
        "training_protocol": "official NucleicBERT pretrained encoder frozen; official SecStruct2DPredictionHead trained by project on bprna_tr1c with BCE+pos_weight=100; VL0-selected threshold=0.55; original B=4 first-598-nt protocol",
        "comparability": "Project downstream adaptation, not an author-released secondary-structure checkpoint. Distinct from the author full-fine-tuned secstr.ckpt arm.",
    },
    "structrfm_author_bprna1m_ssp": {
        "family": "structRFM",
        "path": Path("/root/rna_jepa_data/eval_decision_ref/structrfm_bprna_new.json"),
        "sha256": "3243b141eed79a8c419bc737dece47943fd7708f2db5cd881523dcc9df1d6519",
        "kind": "structrfm_connects",
        "origin": "author_checkpoint",
        "model_form": "single_model_not_ensemble",
        "training_protocol": "author structRFM_SSP_bpRNA1m.pt; bpRNA1m downstream fine-tune; frozen LM embeddings to MixedFold CNN/LSTM pair scorer with Turner parameters and constrained DP decode; max_len=512",
        "comparability": "One author checkpoint, not a multi-source pretraining ensemble or a multi-checkpoint prediction ensemble. Project raw labels and exact pairs are used for this audit.",
    },
    "rnafm_project_frozen_encoder_crf": {
        "family": "RNA-FM",
        "path": Path("/root/rna_jepa_data/eval_decision_ref/ow_rnafm_ff_b4_s0_step20000_bprna_new/result.json"),
        "sha256": "758b0087ac4337420fbc6120913476fda98db9c63cf204ab69d6cb6720d60a8f",
        "kind": "per_sequence",
        "origin": "project_downstream_checkpoint",
        "model_form": "single_model_not_ensemble",
        "training_protocol": "frozen RNA-FM 640-d embeddings; project CRF/decision head trained on TR0 for 20000 steps with VL0 protocol; learned Turner prior weight; exact max-product/Nussinov decode",
        "comparability": "Project backbone-swap baseline, not the official RNA-FM-ResNet_bpRNA.pth downstream model. Official RNA-FM bpRNA config instead specifies a PC-ResNet head and threshold=0.47729617.",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def pair_set(values, length=None):
    pairs = set()
    partners = {}
    for value in values:
        assert len(value) == 2, value
        i, j = map(int, value)
        assert 0 <= i < j, value
        if length is not None:
            assert j < length, (value, length)
        assert i not in partners and j not in partners, value
        partners[i], partners[j] = j, i
        pairs.add((i, j))
    return pairs


def normalize_sequence(sequence):
    return sequence.upper().replace("T", "U")


def strict_metrics(rows, prediction_by_name):
    names = [row["name"] for row in rows]
    missing = [name for name in names if name not in prediction_by_name]
    extra = sorted(set(prediction_by_name) - set(names))
    assert not missing, missing[:10]
    assert not extra, extra[:10]
    tp = fp = fn = 0
    per_sequence = []
    for row in rows:
        name = row["name"]
        truth = pair_set(row["pairs"], len(row["seq"]))
        predicted = pair_set(prediction_by_name[name]["pairs"], len(row["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        denominator = 2 * a + b + c
        f1 = 2 * a / denominator if denominator else 1.0
        tp, fp, fn = tp + a, fp + b, fn + c
        per_sequence.append({
            "name": name,
            "length": len(row["seq"]),
            "n_truth_pairs": len(truth),
            "n_predicted_pairs": len(predicted),
            "tp": a,
            "fp": b,
            "fn": c,
            "precision": a / (a + b) if a + b else None,
            "recall": a / (a + c) if a + c else None,
            "f1": f1,
        })
    n = len(rows)
    return {
        "n_dataset_sequences": n,
        "n_with_prediction": n,
        "coverage": 1.0 if n else None,
        "missing_names": [],
        "extra_names": [],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(row["f1"] for row in per_sequence) / n if n else None,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }, per_sequence


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def prediction_digest(predictions):
    value = [{"name": row["name"], "pairs": row["pairs"]} for row in predictions]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def parse_structrfm(document, rows):
    assert document["n"] == len(document["seqs"]) == len(document["per_seq"]) == len(rows)
    predictions = []
    for index, (source, row) in enumerate(zip(document["seqs"], rows)):
        assert source["id"] == index, (source["id"], index)
        assert normalize_sequence(source["seq"]) == normalize_sequence(row["seq"]), row["name"]
        connects = source["connects"]
        assert len(connects) == len(row["seq"]), row["name"]
        directed = []
        for i, partner in enumerate(connects):
            if partner:
                j = int(partner) - 1
                assert 0 <= j < len(connects), (index, i, partner)
                directed.append((i, j))
        pairs = pair_set({(min(i, j), max(i, j)) for i, j in directed}, len(connects))
        predictions.append({"name": row["name"], "pairs": sorted(map(list, pairs))})
    return predictions


def load_predictions(spec, rows):
    document = json.loads(spec["path"].read_text())
    if spec["kind"] == "structrfm_connects":
        predictions = parse_structrfm(document, rows)
    else:
        key = "predictions" if spec["kind"] == "predictions" else "per_sequence"
        source = document[key]
        pair_key = "pairs" if spec["kind"] == "predictions" else "pred_pairs"
        predictions = [{"name": row["name"], "pairs": row[pair_key]} for row in source]
    assert len(predictions) == len(rows)
    assert [row["name"] for row in predictions] == [row["name"] for row in rows]
    assert all(count == 1 for count in Counter(row["name"] for row in predictions).values())
    for prediction, row in zip(predictions, rows):
        prediction["pairs"] = sorted(map(list, pair_set(prediction["pairs"], len(row["seq"]))))
    return document, predictions


def audit_recorded(document, spec, metrics, predictions):
    checks = {}
    digest = prediction_digest(predictions)
    if spec["family"] in {"RiNALMo", "NucleicBERT"}:
        runs = document["runs"] if spec["family"] == "RiNALMo" else document["repeats"]
        assert len(runs) == 3
        for run in runs:
            assert (run["tp"], run["fp"], run["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
            assert_close(run["strict_micro_f1"], metrics["micro_f1"])
            assert_close(run["strict_macro_f1"], metrics["macro_f1"])
            assert run["predictions_sha256"] == digest
        checks["recorded_deterministic_inference_repeats_verified"] = 3
    elif spec["family"] == "structRFM":
        recorded = document["our_protocol"]
        assert_close(recorded["strict_micro_f1"], metrics["micro_f1"])
        assert_close(recorded["precision"], metrics["micro_precision"])
        assert_close(recorded["recall"], metrics["micro_recall"])
        recorded_counts = [(row["tp"], row["fp"], row["fn"]) for row in document["per_seq"]]
        checks["recorded_per_sequence_counts_verified"] = True
        assert recorded_counts == [(row["tp"], row["fp"], row["fn"]) for row in metrics.pop("_per_sequence")]
        assert document["n_over_maxlen"] == 0
    else:
        recorded = document["pair_level"]
        assert (recorded["micro"]["tp"], recorded["micro"]["fp"], recorded["micro"]["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
        assert_close(recorded["micro"]["f1"], metrics["micro_f1"])
        assert_close(recorded["macro"]["f1"], metrics["macro_f1"])
        checks["recorded_metrics_verified"] = True
    checks["prediction_identity_sha256"] = digest
    return checks


def weight_audits(output_root):
    official = output_root / "official_assets"
    rinalmo_meta = json.loads((official / "rinalmo/zenodo_15043668.json").read_text())
    nucleic_meta = json.loads((official / "nucleicbert/zenodo_16989562.json").read_text())
    struct_meta = json.loads((official / "structrfm/release_v0.0.8.json").read_text())
    rinalmo_weight = Path("/root/rna_baselines_src/rinalmo_giga_ss_bprna_ft.pt")
    nucleic_weight = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/weights/round39/NucleicBERT/secstr.ckpt")
    nucleic_pretrain = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/pretrained.pt")
    nucleic_custom = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_data/nb_ft/nb_finetuned.pt")

    def zenodo_file(document, name):
        return next(row for row in document["files"] if row["key"] == name)

    rinalmo_official = zenodo_file(rinalmo_meta, "rinalmo_giga_ss_bprna_ft.pt")
    nucleic_official = zenodo_file(nucleic_meta, "secstr.ckpt")
    nucleic_pretrain_official = zenodo_file(nucleic_meta, "pretrained.pt")
    struct_official = next(row for row in struct_meta["assets"] if row["name"] == "structRFM_SSP_bpRNA1m.pt")
    assert rinalmo_weight.stat().st_size == rinalmo_official["size"]
    assert nucleic_weight.stat().st_size == nucleic_official["size"]
    assert nucleic_pretrain.stat().st_size == nucleic_pretrain_official["size"]
    rinalmo_md5 = md5(rinalmo_weight)
    nucleic_md5 = md5(nucleic_weight)
    nucleic_pretrain_md5 = md5(nucleic_pretrain)
    assert rinalmo_md5 == rinalmo_official["checksum"].removeprefix("md5:")
    assert nucleic_md5 == nucleic_official["checksum"].removeprefix("md5:")
    assert nucleic_pretrain_md5 == nucleic_pretrain_official["checksum"].removeprefix("md5:")
    return {
        "rinalmo_author_bprna_ft": {
            "status": "PASS_OFFICIAL_BYTES_SIZE_MD5_SHA256",
            "path": str(rinalmo_weight), "bytes": rinalmo_weight.stat().st_size,
            "md5": rinalmo_md5, "sha256": sha256(rinalmo_weight),
            "official_url": rinalmo_official["links"]["self"],
        },
        "nucleicbert_author_secstr_full_ft": {
            "status": "PASS_OFFICIAL_BYTES_SIZE_MD5_SHA256",
            "path": str(nucleic_weight), "bytes": nucleic_weight.stat().st_size,
            "md5": nucleic_md5, "sha256": sha256(nucleic_weight),
            "official_url": nucleic_official["links"]["self"],
        },
        "nucleicbert_project_frozen_encoder_tr1c": {
            "status": "PASS_OFFICIAL_PRETRAIN_BYTES_AND_PROJECT_CHECKPOINT_SHA256",
            "official_pretrained": {
                "path": str(nucleic_pretrain), "bytes": nucleic_pretrain.stat().st_size,
                "md5": nucleic_pretrain_md5, "sha256": sha256(nucleic_pretrain),
                "official_url": nucleic_pretrain_official["links"]["self"],
            },
            "project_downstream": {"path": str(nucleic_custom), "bytes": nucleic_custom.stat().st_size, "sha256": sha256(nucleic_custom)},
        },
        "structrfm_author_bprna1m_ssp": {
            "status": "OFFICIAL_RELEASE_IDENTITY_RECOVERED_LOCAL_BYTES_UNVERIFIED",
            "official_url": struct_official["browser_download_url"], "official_bytes": struct_official["size"],
            "official_sha256": struct_official["digest"].removeprefix("sha256:"),
            "blocker": "Historical checkpoint path is not present on px2. Three direct download attempts timed out after only partial transfers; partial bytes were deleted and no checksum claim was made.",
        },
        "rnafm_project_frozen_encoder_crf": {
            "status": "PROJECT_PREDICTIONS_VERIFIED_WEIGHT_BYTES_UNAVAILABLE_ON_PX2",
            "official_backbone_name": "RNA-FM_pretrained.pth",
            "official_backbone_source": "https://huggingface.co/cuhkaih/rnafm and official CUHK download URL in fm/pretrained.py",
            "blocker": "The historical pretrained backbone and project downstream checkpoint paths are absent on px2; no new backbone or head checksum is claimed.",
        },
        "ernie_rna_author_bprna_new_protocol": {
            "status": "OFFICIAL_DOWNLOAD_LINKS_RECOVERED_NO_LOCAL_WEIGHT_VERIFICATION",
            "protocol_checkpoint": "ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt",
            "checkpoint_link_provenance": "official checkpoint/ERNIE-RNA_ss_prediction_checkpoint/ERNIE-RNA_ss_prediction_checkpoint.txt",
        },
    }


def exposure_audit():
    root = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")
    exact_path = root / "exact_union_summary.json"
    homology_path = root / "homology_round56/homology_union_summary.json"
    exact = json.loads(exact_path.read_text())
    homology = json.loads(homology_path.read_text())
    assert sha256(exact_path) == "d441239311605a2572b3f561357531689b0ea24a19068e1ac417dca75148baa4"
    assert sha256(homology_path) == "69a0b13b2ef0cb061807e10f50efca3a6da7889ee006cd032e2e72bca2197131"
    return {
        "rinalmo_known_source_audit": {
            "exact_path": str(exact_path), "exact_sha256": sha256(exact_path),
            "exact_overlap_bprna_new": exact["queries"]["bprna_new"],
            "homology_path": str(homology_path), "homology_sha256": sha256(homology_path),
            "homology_80pct_bidirectional_coverage_bprna_new": homology["queries"]["bprna_new"],
            "limitation": "Known-source RiNALMo audit only: 5381/5388 exact hits and 5388/5388 detected homology hits. Seven exact-undetected sequences are UNKNOWN, not clean. No exact-undetected or homology-undetected performance stratum is imputed. The historical nt snapshot and complete pretraining corpus/family independence remain unresolved.",
        },
        "all_other_models": "Raw non-decontaminated results. RNA-FM (RNAcentral100), NucleicBERT (MARS), structRFM (RNAcentral/BPfold-derived 21M sequence-structure pairs plus bpRNA1m downstream), and ERNIE-RNA may have pretraining or downstream exposure. No model-specific bpRNA-new exact/homology clearance is claimed. Downstream split labels or TR0/TR1c decontamination do not establish pretraining decontamination.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--skip-weight-hashes", action="store_true")
    args = parser.parse_args()
    assert sha256(args.data) == DATA_SHA256
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    assert len(rows) == 5388
    assert len({row["name"] for row in rows}) == len(rows)
    assert sum(len(pair_set(row["pairs"], len(row["seq"]))) for row in rows) == 135184

    result = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FIVE_FULL_PREDICTION_IDENTITIES_ERNIE_BLOCKED",
        "scope": "CPU-only independent parsing and exact-pair scoring of existing frozen predictions; no model inference, training, calibration, model selection, or imputation",
        "dataset": {
            "path": str(args.data), "sha256": DATA_SHA256, "n_sequences": len(rows),
            "n_truth_pairs": 135184,
            "pair_convention": "zero-based inclusive exact pairs; no crossing or short-span filtering",
            "label_scope": "PROJECT_RAW_GT_ONLY_NON_DECONTAMINATED",
        },
        "systems": {},
        "blocked": {
            "ernie_rna_author_bprna_new_protocol": {
                "family": "ERNIE-RNA", "status": "BLOCKED_NO_FULL_PROJECT_PREDICTIONS",
                "metrics": None, "coverage": 0.0, "n_with_prediction": 0, "n_dataset_sequences": len(rows),
                "official_source": "https://github.com/Bruce-ywj/ERNIE-RNA",
                "official_protocol": "dataset_name=bpRNA-new selects ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt; bpRNA-new is not the downstream training set; post-processing uses constrained attention-map optimization and CT output",
                "reason": "Official source, checkpoint links, and protocol were recovered, but no full 5,388-sequence prediction artifact exists on px2. The repository's bpRNA-new FASTA and CT outputs are a one-sequence demonstration only. Paper/summary values were not substituted and no inference was run.",
            }
        },
        "official_protocol_context": {
            "rnafm_author_bprna_config": {
                "source": "redevelop/pretrained/experiments/ss_prediction_bpRNA.yml",
                "pairwise_predictor": "pc-resnet_1_sym_first:r-ss", "threshold": 0.47729617,
                "weight_path": "pretrained/Models/SS/RNA-FM-ResNet_bpRNA.pth",
                "prediction_status": "NO_FROZEN_AUTHOR_PREDICTIONS_ON_PX2",
            }
        },
        "exposure_and_homology": exposure_audit(),
    }
    for system_id, spec in SOURCES.items():
        path = spec["path"]
        assert sha256(path) == spec["sha256"]
        document, predictions = load_predictions(spec, rows)
        prediction_by_name = {row["name"]: row for row in predictions}
        metrics, per_sequence = strict_metrics(rows, prediction_by_name)
        metrics_for_check = dict(metrics)
        metrics_for_check["_per_sequence"] = per_sequence
        checks = audit_recorded(document, spec, metrics_for_check, predictions)
        write_jsonl(args.output_root / f"predictions/{system_id}/bprna_new_v3.jsonl", predictions)
        write_jsonl(args.output_root / f"reports/{system_id}_bprna_new_per_sequence_v3.jsonl", per_sequence)
        result["systems"][system_id] = {
            **{key: value for key, value in spec.items() if key not in {"path", "sha256", "kind"}},
            "status": "PASS_INDEPENDENT_RECOMPUTATION",
            "artifact": str(path), "artifact_sha256": spec["sha256"],
            "metrics": metrics, "checks": checks,
        }
    result["weights"] = weight_audits(args.output_root) if not args.skip_weight_hashes else {"status": "SKIPPED_BY_REQUEST"}
    out = args.output_root / "reports/foundation_bprna_new_metrics_verification_v3.json"
    write_json(out, result)
    print(json.dumps({
        "status": result["status"],
        "systems": {key: value["metrics"] for key, value in result["systems"].items()},
        "blocked": result["blocked"],
        "weights": {key: value["status"] for key, value in result["weights"].items()},
        "report": str(out), "report_sha256": sha256(out),
    }, indent=2))


if __name__ == "__main__":
    main()
