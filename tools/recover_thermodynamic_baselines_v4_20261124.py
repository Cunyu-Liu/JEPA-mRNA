#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v4_thermodynamic")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
DATA_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
CLEAN_DATA = ART / "clean_splits/tr1c/bprna_new.jsonl"
CLEAN_DATA_SHA256 = "231ae0079c183d481e4f5d4518f77a41fa52f6a6544b2feb0a7d60f8f9b23595"
RAW_SOURCES = {
    "vienna": (ART / "vienna/raw_strict/bprna_new.json", "615c47042b927f06d5870537ecc2d71ac3cbdabac03c14957d9e5c19ca42d86b"),
    "rnastructure": (ART / "rnastructure/raw/bprna_new.json", "ac78069c6fc49765c29e3c0cd376bb746b7bdbf0a64ab2b116c2c1bcec09ae42"),
    "contrafold": (ART / "contrafold/raw/bprna_new.json", "6964a40fa3174f1b43bc13955b8e2a41b7f83014dabf91a1036c46b625884234"),
}
CLEAN_CONTRAFOLD = (ART / "contrafold/clean_tr1c/bprna_new.json", "642059d4a0f4c48f4a8672a24a0daed2177a248133df238c9da281dc079d0597")
SYSTEMS = ("vienna_mfe", "vienna_centroid", "vienna_mea", "rnastructure_fold", "rnastructure_maxexpect", "contrafold_mea")
REPLAY_INDICES = (0, 1, 2, 17, 100, 501, 1000, 2000, 3000, 4000, 5000, 5387)


def sha256(path):
    digest = hashlib.sha256()
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


def strict_metrics(rows, prediction_by_name):
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))
    assert set(names) == set(prediction_by_name), "Prediction and label scopes differ"
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
            "name": name, "length": len(row["seq"]), "n_truth_pairs": len(truth),
            "n_predicted_pairs": len(predicted), "tp": a, "fp": b, "fn": c,
            "precision": a / (a + b) if a + b else None,
            "recall": a / (a + c) if a + c else None, "f1": f1,
        })
    n = len(rows)
    return {
        "n_dataset_sequences": n, "n_with_prediction": n,
        "coverage": 1.0 if n else None, "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "macro_f1": math.fsum(row["f1"] for row in per_sequence) / n if n else None,
        "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
    }, per_sequence


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def load_rows(path, expected_sha, expected_n):
    assert sha256(path) == expected_sha
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == expected_n
    assert all(count == 1 for count in Counter(row["name"] for row in rows).values())
    return rows


def unavailable_clean(reason):
    return {
        "status": "NOT_AVAILABLE_DISTINCT_CLEAN_SCOPE", "scope": None,
        "source_artifact": None, "source_artifact_sha256": None,
        "prediction_artifact": None, "prediction_artifact_sha256": None,
        "metrics": None, "coverage": None, "reason": reason,
    }


def audit_frozen(system, decoder, scope, source_path, expected_source_sha, rows, per_sequence, repeats, output_root):
    assert sha256(source_path) == expected_source_sha
    assert len(per_sequence) == len(rows)
    assert [row["name"] for row in per_sequence] == [row["name"] for row in rows]
    predictions = {row["name"]: {"pairs": row["pred_pairs"]} for row in per_sequence}
    assert len(predictions) == len(rows)
    metrics, independent_per_sequence = strict_metrics(rows, predictions)
    for actual, recorded in zip(independent_per_sequence, per_sequence):
        assert actual["name"] == recorded["name"]
        assert (actual["tp"], actual["fp"], actual["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"])
        assert_close(actual["f1"], recorded["f1"])
    identity = [[row["name"], predictions[row["name"]]["pairs"]] for row in rows]
    prediction_identity_sha256 = hashlib.sha256(canonical(identity).encode()).hexdigest()
    assert len(repeats) == 3
    for repeat in repeats:
        recorded = repeat["metrics"]
        assert repeat["prediction_sha256"] == prediction_identity_sha256
        assert recorded["n_sequences"] == len(rows)
        assert (recorded["tp"], recorded["fp"], recorded["fn"]) == (metrics["tp"], metrics["fp"], metrics["fn"])
        assert_close(recorded["micro_f1"], metrics["micro_f1"])
        assert_close(recorded["macro_f1"], metrics["macro_f1"])
    prediction_path = output_root / "predictions" / system / f"bprna_new_{scope}_v4.jsonl"
    metric_path = output_root / "reports" / f"{system}_bprna_new_{scope}_per_sequence_v4.jsonl"
    normalized = [{"name": row["name"], "pairs": predictions[row["name"]]["pairs"]} for row in rows]
    write_jsonl(prediction_path, normalized)
    write_jsonl(metric_path, independent_per_sequence)
    return {
        "status": "PASS_INDEPENDENT_RECOMPUTATION", "system": system, "decoder": decoder,
        "scope": scope, "source_artifact": str(source_path),
        "source_artifact_sha256": expected_source_sha,
        "prediction_artifact": str(prediction_path), "prediction_artifact_sha256": sha256(prediction_path),
        "prediction_identity_sha256": prediction_identity_sha256,
        "per_sequence_metrics_artifact": str(metric_path),
        "per_sequence_metrics_sha256": sha256(metric_path),
        "unique_names": len(predictions), "order_matches_labels": True,
        "recorded_deterministic_repeats_verified": 3, "metrics": metrics,
        "coverage": metrics["coverage"],
    }


def parse_dot_pairs(structure):
    stack = []
    pairs = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            assert stack
            pairs.append((stack.pop(), index))
        else:
            assert symbol == "."
    assert not stack
    return sorted(pairs)


def parse_ct_pairs(path, sequence):
    lines = path.read_text().splitlines()
    block_size = len(sequence) + 1
    assert lines and len(lines) % block_size == 0
    blocks = [lines[start:start + block_size] for start in range(0, len(lines), block_size)]
    first = []
    for index, line in enumerate(blocks[0][1:], 1):
        fields = line.split()
        assert len(fields) == 6 and int(fields[0]) == index
        assert fields[1].upper().replace("T", "U") == sequence[index - 1].upper().replace("T", "U")
        partner = int(fields[4]) - 1
        if partner > index - 1:
            first.append((index - 1, partner))
    return sorted(first), len(blocks)


def replay_cpu(rows, raw_documents, output_root):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", DATAPATH="/home/rna_jev/packages/rnastructure-6.6/RNAstructure/data_tables")
    selected = [rows[index] for index in REPLAY_INDICES]
    raw_vienna = raw_documents["vienna"]["results"]
    raw_rna = raw_documents["rnastructure"]["results"]
    raw_contra = {row["name"]: row["pred_pairs"] for row in raw_documents["contrafold"]["per_sequence"]}
    observations = {system: [] for system in SYSTEMS}
    import RNA
    assert RNA.__version__ == raw_documents["vienna"]["vienna_version"] == "2.7.2"
    rna_exe = Path("/home/rna_jev/packages/rnastructure-6.6/RNAstructure/exe")
    contra = Path("/home/rna_jev/third_party/EternaFold/src/contrafold")
    with tempfile.TemporaryDirectory(prefix="thermodynamic_recovery_v4_cpu_") as temporary:
        directory = Path(temporary)
        for index, row in enumerate(selected):
            compound = RNA.fold_compound(row["seq"].upper().replace("T", "U"))
            mfe, _ = compound.mfe()
            compound.pf()
            centroid, _ = compound.centroid()
            mea, _ = compound.MEA()
            decoded = {"vienna_mfe": mfe, "vienna_centroid": centroid, "vienna_mea": mea}
            for system, structure in decoded.items():
                expected = raw_vienna[system]["per_sequence"][REPLAY_INDICES[index]]["pred_pairs"]
                actual = parse_dot_pairs(structure)
                assert actual == [tuple(pair) for pair in expected], (system, row["name"])
                observations[system].append({"name": row["name"], "pairs_match": True})
            fasta = directory / "input.fa"
            fasta.write_text(f">{row['name']}\n{row['seq']}\n")
            for system, executable, flag in (("rnastructure_fold", "Fold", "--MFE"), ("rnastructure_maxexpect", "MaxExpect", "--sequence")):
                ct = directory / f"{system}.ct"
                ct.unlink(missing_ok=True)
                subprocess.run([str(rna_exe / executable), str(fasta), str(ct), flag], check=True, capture_output=True, text=True, env=env, timeout=180)
                actual, n_ct = parse_ct_pairs(ct, row["seq"])
                expected = raw_rna[system.removeprefix("rnastructure_")]["per_sequence"][REPLAY_INDICES[index]]["pred_pairs"]
                assert actual == [tuple(pair) for pair in expected], (system, row["name"])
                observations[system].append({"name": row["name"], "pairs_match": True, "ct_structures": n_ct, "decode_block": "first"})
            seq_path = directory / "input.seq"
            bpseq = directory / "structure.bpseq"
            seq_path.write_text(row["seq"] + "\n")
            bpseq.unlink(missing_ok=True)
            subprocess.run([str(contra), "predict", str(seq_path), "--bpseq", str(bpseq)], check=True, capture_output=True, text=True, env=env, timeout=180)
            directed = []
            for line_index, line in enumerate(bpseq.read_text().splitlines()):
                fields = line.split()
                assert len(fields) == 3 and int(fields[0]) == line_index + 1
                partner = int(fields[2]) - 1
                if partner >= 0:
                    directed.append((line_index, partner))
            actual = sorted(pair_set({(min(i, j), max(i, j)) for i, j in directed}, len(row["seq"])))
            assert actual == [tuple(pair) for pair in raw_contra[row["name"]]], row["name"]
            observations["contrafold_mea"].append({"name": row["name"], "pairs_match": True})
    result = {
        "status": "PASS_CPU_REPLAY_ALL_SELECTED_PAIRS_EXACT", "device": "CPU_ONLY",
        "gpu_used": False, "cuda_visible_devices": "", "n_indices": len(REPLAY_INDICES),
        "indices": list(REPLAY_INDICES), "systems": observations,
        "boundary": "Bounded deterministic replay verifies current runtime identity; all reported full-scope metrics are independently recomputed from frozen predictions.",
    }
    write_json(output_root / "reports/cpu_replay_verification_v4.json", result)
    return result


def vienna_provenance(document):
    import RNA
    from importlib import metadata
    distribution = metadata.distribution("ViennaRNA")
    extension = next(Path(RNA.__file__).parent.glob("_RNA*.so"))
    record_rows = distribution.read_text("RECORD").splitlines()
    record = next(row for row in record_rows if Path(row.split(",")[0]).name == extension.name)
    encoded = record.split(",")[1].removeprefix("sha256=")
    recorded_digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
    assert recorded_digest == sha256(extension)
    return {
        "family": "ViennaRNA", "version": document["vienna_version"],
        "interface": "Python RNA.fold_compound: mfe(); pf(); centroid(); MEA()",
        "binary": None, "binary_sha256": None,
        "runtime_extension": str(extension), "runtime_extension_sha256": sha256(extension),
        "distribution_record_sha256_verified": True, "distribution": "ViennaRNA 2.7.2 Python distribution",
        "parameters": {"status": "LIBRARY_DEFAULT_NO_EXPLICIT_FROZEN_PARAMETER_ASSET", "name": None, "path": None, "sha256": None},
        "decoders": {"vienna_mfe": "MFE", "vienna_centroid": "centroid after partition function", "vienna_mea": "MEA after partition function"},
    }


def rnastructure_provenance(document, output_root):
    exe = Path("/home/rna_jev/packages/rnastructure-6.6/RNAstructure/exe")
    archive = exe.parents[1] / "RNAstructureLinuxTextInterfaces64bit.tgz"
    sums = exe.parents[1] / "SHA256SUMS"
    assert str(sha256(archive)) in sums.read_text()
    table_root = exe.parent / "data_tables"
    table_hashes = {str(path): sha256(path) for path in sorted(table_root.rglob("*")) if path.is_file()}
    table_manifest = output_root / "provenance/rnastructure_6_6_data_tables_sha256_v4.json"
    write_json(table_manifest, {"root": str(table_root), "files": table_hashes})
    versions = {}
    for name in ("Fold", "MaxExpect"):
        completed = subprocess.run([str(exe / name), "--version"], text=True, capture_output=True)
        versions[name] = (completed.stdout + completed.stderr).strip().splitlines()[0]
        assert "Version 6.6" in versions[name]
        assert sha256(exe / name) == document["results"][name.lower()]["binary_sha256"]
    return {
        "family": "RNAstructure", "version": "6.6", "release_date": "2026-04-02",
        "version_commands": versions, "official_archive": str(archive), "official_archive_sha256": sha256(archive),
        "official_archive_sha256sums_verified": True,
        "binaries": {name: {"path": str(exe / name), "sha256": sha256(exe / name)} for name in ("Fold", "MaxExpect")},
        "parameters": {"DATAPATH": str(table_root), "alphabet": "rna", "manifest": str(table_manifest), "manifest_sha256": sha256(table_manifest)},
        "decoders": {
            "rnastructure_fold": "Fold --MFE; one CT structure",
            "rnastructure_maxexpect": "MaxExpect --sequence; defaults gamma=1, percent=50, structures=1000, window=5; first CT block retained by frozen protocol",
        },
    }


def contrafold_provenance(document):
    repository = Path(document["repository"])
    binary = Path(document["binary"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    assert commit == document["commit"]
    assert sha256(binary) == document["binary_sha256"]
    parameter_hashes = {}
    for name, expected in document["parameters_sha256"].items():
        path = binary.parent / name
        assert sha256(path) == expected
        parameter_hashes[str(path)] = expected
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--short"], text=True).splitlines()
    origin = subprocess.check_output(["git", "-C", str(repository), "remote", "get-url", "origin"], text=True).strip()
    return {
        "family": "CONTRAfold", "version": None, "source_repository": origin,
        "source_commit": commit, "source_tree_status": dirty,
        "source_note": "The commit and prediction binary/parameter bytes match; an unrelated existing src/score_prediction modification means the whole working tree is not claimed pristine.",
        "binary": str(binary), "binary_sha256": sha256(binary),
        "parameters": {"status": "NATIVE_CONTRAFOLD_PARAMS_VERIFIED", "files_sha256": parameter_hashes},
        "decoders": {"contrafold_mea": "contrafold predict <sequence> --bpseq <output>; native posterior MEA; single serial CPU process"},
    }


def report_markdown(document):
    lines = [
        "# Thermodynamic baseline recovery v4 (2026-11-24)", "",
        "## Scope and metric contract", "",
        "The primary comparison uses the frozen project bpRNA-new DATABASE labels and exactly 5,388 unique named sequences. Pair coordinates are zero-based inclusive `i < j`; scoring is exact pair-set equality with no crossing, span, or minimum-loop filtering. Micro-F1 is computed from pooled TP/FP/FN; macro-F1 is the unweighted mean of per-sequence F1. If truth and prediction both have zero pairs, per-sequence F1 is 1. Recovery is CPU-only.", "",
        "| Predictor / decoding identity | Scope | N | Coverage | TP | FP | FN | Micro-F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = document["systems"][system]["raw"]
        m = row["metrics"]
        lines.append(f"| {system} | raw | {m['n_dataset_sequences']} | {m['coverage']:.12f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['micro_f1']:.15f} | {m['macro_f1']:.15f} |")
    clean = document["systems"]["contrafold_mea"]["clean"]
    m = clean["metrics"]
    lines.extend([
        "", "## Separate clean scope", "",
        "ViennaRNA and RNAstructure are not sequence-fitted and have no distinct clean prediction asset; their clean artifact, metrics, and coverage fields are `null`. Their applicable raw 5,388 result is not duplicated under a clean label. CONTRAfold has a historical `clean_tr1c` artifact, shown separately below. It removes two sequences and is a project TR1c exclusion scope, **not** a verified CONTRAfold own-training-clean scope and not eligible for the same-5,388 comparison.", "",
        "| Predictor | Clean scope | N | Coverage within clean scope | TP | FP | FN | Micro-F1 | Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| CONTRAfold MEA | clean_tr1c | {m['n_dataset_sequences']} | {m['coverage']:.12f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['micro_f1']:.15f} | {m['macro_f1']:.15f} |",
        "", f"Excluded from raw by `clean_tr1c`: `{', '.join(document['clean_scope_exclusions']['names'])}`. Raw-scope coverage of this clean asset is {document['clean_scope_exclusions']['n_retained']}/5388; no scores are imputed for the excluded sequences.", "",
        "## Provenance verification", "",
        "- All four frozen source JSON hashes and the frozen raw/clean label hashes match their expected values.",
        "- ViennaRNA is version 2.7.2. The Python extension SHA-256 matches its installed distribution `RECORD`; no CLI binary or explicit frozen parameter file was recorded, so those fields remain `null`.",
        "- RNAstructure Fold and MaxExpect both report version 6.6 (April 2, 2026). Both executable hashes match the frozen artifacts, and the official Linux archive matches `SHA256SUMS`. Fold uses `--MFE`; MaxExpect uses `--sequence` and the frozen protocol retains the first CT block.",
        "- CONTRAfold uses source commit `87b9aac55cee14fd562049d08f7b92d3131f10ce`; its binary and all three native parameter hashes match. A pre-existing unrelated `src/score_prediction` working-tree modification is recorded rather than hidden.",
        f"- Bounded CPU replay checked {document['replay']['n_indices']} selected sequences per decoder; every regenerated pair set exactly matched the frozen prediction. Full-table metrics are from frozen predictions, not partial replay extrapolation.",
        "- Every system has three recorded deterministic inference repeats; all recorded counts, micro/macro F1 values, and prediction identity hashes were independently checked. These repeats are not training seeds or ensembles.", "",
        "## Verification", "",
        "The standalone verifier does not import this recovery script or the shared evaluator. It rescores normalized prediction JSONL with exact rational per-sequence arithmetic and verifies all artifact hashes, raw/clean scope boundaries, and unavailable-asset `null` fields.", "",
        "```bash",
        "/root/miniconda3/envs/codonflow/bin/python3 -m pytest -q tests/test_recover_thermodynamic_baselines_v4_20261124.py",
        "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/codonflow/bin/python3 tools/recover_thermodynamic_baselines_v4_20261124.py",
        "CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/codonflow/bin/python3 tools/verify_thermodynamic_recovery_v4_20261124.py",
        "```", "",
        "No shared registry, evaluator, running configuration, or PPT file is modified by this recovery.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    assert args.root == ROOT, "Use the isolated v4 thermodynamic artifact root"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rows = load_rows(DATA, DATA_SHA256, 5388)
    clean_rows = load_rows(CLEAN_DATA, CLEAN_DATA_SHA256, 5386)
    raw_documents = {}
    for family, (path, expected) in RAW_SOURCES.items():
        assert sha256(path) == expected
        document = json.loads(path.read_text())
        assert document["status"] == "PASS" and document["data_sha256"] == DATA_SHA256
        assert document["n_sequences"] == len(rows)
        raw_documents[family] = document
    systems = {system: {"raw": None, "clean": None, "provenance": None} for system in SYSTEMS}
    for system in SYSTEMS[:3]:
        result = raw_documents["vienna"]["results"][system]
        systems[system]["raw"] = audit_frozen(system, system, "raw", RAW_SOURCES["vienna"][0], RAW_SOURCES["vienna"][1], rows, result["per_sequence"], result["repeats"], ROOT)
        systems[system]["clean"] = unavailable_clean("ViennaRNA is not sequence-fitted; no distinct clean prediction artifact exists and the raw 5,388 result is not relabeled as clean")
        systems[system]["provenance"] = vienna_provenance(raw_documents["vienna"])
    for system, key in (("rnastructure_fold", "fold"), ("rnastructure_maxexpect", "maxexpect")):
        result = raw_documents["rnastructure"]["results"][key]
        systems[system]["raw"] = audit_frozen(system, key, "raw", RAW_SOURCES["rnastructure"][0], RAW_SOURCES["rnastructure"][1], rows, result["per_sequence"], result["repeats"], ROOT)
        systems[system]["clean"] = unavailable_clean("RNAstructure is not sequence-fitted; no distinct clean prediction artifact exists and the raw 5,388 result is not relabeled as clean")
        systems[system]["provenance"] = rnastructure_provenance(raw_documents["rnastructure"], ROOT)
    result = raw_documents["contrafold"]
    systems["contrafold_mea"]["raw"] = audit_frozen("contrafold_mea", "native_mea", "raw", RAW_SOURCES["contrafold"][0], RAW_SOURCES["contrafold"][1], rows, result["per_sequence"], result["repeats"], ROOT)
    clean_document = json.loads(CLEAN_CONTRAFOLD[0].read_text())
    assert clean_document["status"] == "PASS" and clean_document["data_sha256"] == CLEAN_DATA_SHA256
    systems["contrafold_mea"]["clean"] = audit_frozen("contrafold_mea", "native_mea", "clean_tr1c", CLEAN_CONTRAFOLD[0], CLEAN_CONTRAFOLD[1], clean_rows, clean_document["per_sequence"], clean_document["repeats"], ROOT)
    systems["contrafold_mea"]["clean"]["scope_interpretation"] = "PROJECT_TR1C_EXCLUSION_NOT_VERIFIED_CONTRAFOLD_OWN_TRAIN_CLEAN"
    systems["contrafold_mea"]["clean"]["coverage_of_raw_5388"] = len(clean_rows) / len(rows)
    systems["contrafold_mea"]["provenance"] = contrafold_provenance(result)
    raw_names = {row["name"] for row in rows}
    clean_names = {row["name"] for row in clean_rows}
    assert clean_names < raw_names
    excluded = sorted(raw_names - clean_names)
    replay = None if args.skip_replay else replay_cpu(rows, raw_documents, ROOT)
    document = {
        "schema_version": 4, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_RAW_5388_ALL_6_DECODERS_CLEAN_SCOPES_SEPARATED",
        "dataset": {"name": "bprna_new", "path": str(DATA), "sha256": DATA_SHA256, "n_sequences": len(rows), "unique_names": len(raw_names), "label_scope": "FROZEN_PROJECT_DATABASE_LABELS", "pair_convention": "zero-based inclusive i<j; exact set equality; no pair filtering"},
        "systems": systems,
        "clean_scope_exclusions": {"scope": "clean_tr1c", "n_retained": len(clean_names), "n_excluded": len(excluded), "names": excluded, "label_path": str(CLEAN_DATA), "label_sha256": CLEAN_DATA_SHA256},
        "replay": replay,
        "execution": {"device": "CPU_ONLY", "gpu_used": False, "cuda_visible_devices": ""},
        "boundary": "Primary metrics cover the same raw 5,388 sequences. The separate clean_tr1c CONTRAfold result covers 5,386 and is not substituted into the primary comparison.",
    }
    metrics_path = ROOT / "reports/thermodynamic_bprna_new_metrics_v4.json"
    write_json(metrics_path, document)
    report = report_markdown(document)
    repo_report = REPO / "docs/thermodynamic_baseline_recovery_v4_20261124.md"
    repo_report.write_text(report)
    artifact_report = ROOT / "reports/thermodynamic_baseline_recovery_v4_20261124.md"
    artifact_report.write_text(report)
    print(json.dumps({"status": document["status"], "metrics": str(metrics_path), "metrics_sha256": sha256(metrics_path), "report": str(repo_report), "replay": None if replay is None else replay["status"]}, indent=2))


if __name__ == "__main__":
    main()
