#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_REPOSITORY = "https://github.com/cobisLab/DEBFold"
OFFICIAL_COMMIT = "ff693cdf4cb177e50d8a6a15c7a6292af802b75e"
OFFICIAL_WEB = "https://cobis.bme.ncku.edu.tw/DEBFold/"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11094721/"
PAPER_DOI = "10.1021/acs.jcim.4c00458"
DATASET_SHA256 = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
ARCHIVE_NAMES = (
    "TrainingValidationSet.zip",
    "TestSet_alpha.zip",
    "TestSet_beta.zip",
    "TestSet_gamma.zip",
)
EXPECTED_ARCHIVE_ROWS = {
    "TrainingValidationSet.zip": (8782, 8384),
    "TestSet_alpha.zip": (44, 44),
    "TestSet_beta.zip": (40, 40),
    "TestSet_gamma.zip": (15, 15),
}
EXPECTED_TREE = {"README", *ARCHIVE_NAMES}
DEFAULT_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/debfold_official_recovery_20261126")
DEFAULT_DATASET = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
UNIFIED_TRAINING = {
    "status": "OFFICIAL_TRAIN_VALIDATION_ARCHIVE_RECOVERED; 25-fold assignments and checkpoint-specific training union not published",
    "project_training_dataset_ids": [],
    "external_training_corpora": ["DEBFold official TrainingValidationSet.zip: 8782 structures/8384 unique sequences; SHA256=20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2"],
    "source_commit": OFFICIAL_COMMIT,
    "archive_sha256": "20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run([
        "curl", "-fLsS", "--retry", "3", "--connect-timeout", "30", "--max-time", "900",
        "--continue-at", "-", url, "-o", str(temporary),
    ], check=True)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Empty download: {url}")
    os.replace(temporary, destination)
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def fetch_api(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "FoldNAR-DEBFold-recovery/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return {"url": url, "path": str(destination), "http_status": status, "bytes": len(body), "sha256": sha256(destination)}


def fetch(root):
    root = Path(root)
    source = root / "source/DEBFold"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", OFFICIAL_REPOSITORY + ".git", str(source)], check=True)
    actual_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Official commit changed: expected {OFFICIAL_COMMIT}, got {actual_commit}")
    evidence = []
    for endpoint, name in (("releases", "github_releases.json"), ("tags", "github_tags.json"), ("license", "github_license_response.json")):
        evidence.append(fetch_api(f"https://api.github.com/repos/cobisLab/DEBFold/{endpoint}", root / "evidence" / name))
    evidence.append(download(OFFICIAL_WEB, root / "evidence/website_main.html"))
    evidence.append(download(OFFICIAL_WEB + "download", root / "evidence/website_download.html"))
    evidence.append(download("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11094721/fullTextXML?format=xml", root / "paper/PMC11094721.fulltext.xml"))
    website_archives = []
    for name in ARCHIVE_NAMES:
        row = download(OFFICIAL_WEB + "data_download/" + name, root / "official_download" / name)
        row["git_archive_sha256"] = sha256(source / name)
        row["matches_git_archive"] = row["sha256"] == row["git_archive_sha256"]
        if not row["matches_git_archive"]:
            raise RuntimeError(f"Website/GitHub archive mismatch: {name}")
        website_archives.append(row)
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": actual_commit,
        "official_web": OFFICIAL_WEB,
        "paper_url": PAPER_URL,
        "paper_doi": PAPER_DOI,
        "evidence": evidence,
        "website_archives": website_archives,
        "download_scope": "Official public source/data/paper only; no third-party model or weights",
    }
    write_json(root / "evidence/download_manifest.json", manifest)
    return manifest


def parse_ct(text, name="archive.ct"):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Empty CT structure: {name}")
    entries = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "Energy":
            continue
        if len(fields) < 6:
            raise ValueError(f"Invalid CT row in {name}: {line}")
        entries.append((int(fields[0]), fields[1], int(fields[4])))
    if [row[0] for row in entries] != list(range(1, len(entries) + 1)):
        raise ValueError(f"Non-contiguous CT indices: {name}")
    sequence = "".join(row[1] for row in entries).upper().replace("T", "U")
    if not sequence or set(sequence) - set("ACGUN"):
        raise ValueError(f"Unsupported sequence alphabet: {name}")
    pairs = set()
    for index, _, partner in entries:
        if partner == 0:
            continue
        if not 1 <= partner <= len(entries) or partner == index:
            raise ValueError(f"Invalid CT partner: {name}:{index}")
        if entries[partner - 1][2] != index:
            raise ValueError(f"Non-reciprocal CT partner: {name}:{index}")
        pairs.add(tuple(sorted((index - 1, partner - 1))))
    return {"name": Path(name).stem, "seq": sequence, "pairs": sorted(pairs)}


def audit_archive(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure: {path}:{bad_member}")
        for name in sorted(item for item in archive.namelist() if item.endswith(".ct")):
            rows.append(parse_ct(archive.read(name).decode("utf-8"), name))
    if not rows:
        raise ValueError(f"No CT structures: {path}")
    return rows


def load_dataset(path=DEFAULT_DATASET, require_full=True):
    path = Path(path)
    if require_full and sha256(path) != DATASET_SHA256:
        raise ValueError("bpRNA-new identity hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate bpRNA-new names")
    if require_full and len(rows) != 5388:
        raise ValueError(f"Expected 5388 bpRNA-new rows, got {len(rows)}")
    for row in rows:
        if not row["name"] or not row["seq"]:
            raise ValueError("Empty name or sequence")
        if set(row["seq"].upper().replace("T", "U")) - set("ACGUN"):
            raise ValueError(f"Unsupported project sequence alphabet: {row['name']}")
    return rows


def validate_prediction_rows(dataset_rows, prediction_rows):
    if len(prediction_rows) != len(dataset_rows):
        raise ValueError("Prediction row count does not match bpRNA-new")
    expected = {row["name"]: row for row in dataset_rows}
    actual = {}
    for row in prediction_rows:
        name = row["name"]
        if name in actual or name not in expected:
            raise ValueError(f"Unknown or duplicate prediction name: {name}")
        source = expected[name]
        if row.get("seq") != source["seq"]:
            raise ValueError(f"Prediction sequence mismatch: {name}")
        pairs = set()
        for pair in row["pred_pairs"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid pair encoding: {name}")
            i, j = pair
            if type(i) is not int or type(j) is not int or not 0 <= i < j < len(source["seq"]):
                raise ValueError(f"Invalid pair indices: {name}:{pair}")
            if (i, j) in pairs:
                raise ValueError(f"Duplicate pair: {name}:{pair}")
            pairs.add((i, j))
        actual[name] = row
    missing = set(expected) - set(actual)
    if missing:
        raise ValueError(f"Missing predictions: {len(missing)}")
    return actual


def update_unified_registry(repo):
    path = Path(repo) / "manifests/foldnar_v1/baseline_registry.json"
    document = json.loads(path.read_text())
    rows = [row for row in document["rows"] if row["baseline_id"] == "debfold"]
    if len(rows) != 1:
        raise ValueError("Unified registry must contain exactly one DEBFold row")
    rows[0]["training_data"] = dict(UNIFIED_TRAINING)
    write_json(path, document)


def audit(root, dataset_path=DEFAULT_DATASET):
    root = Path(root)
    source = root / "source/DEBFold"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError("Official source commit mismatch")
    tree = set(subprocess.check_output(["git", "-C", str(source), "ls-tree", "--name-only", "HEAD"], text=True).splitlines())
    if tree != EXPECTED_TREE:
        raise ValueError(f"Unexpected official tree: {sorted(tree)}")
    releases = json.loads((root / "evidence/github_releases.json").read_text())
    tags = json.loads((root / "evidence/github_tags.json").read_text())
    license_status = json.loads((root / "evidence/download_manifest.json").read_text())["evidence"][2]["http_status"]
    if releases or tags or license_status != 404:
        raise ValueError("Official release/tag/license absence evidence changed")
    archives = {}
    for name, (expected_rows, expected_unique) in EXPECTED_ARCHIVE_ROWS.items():
        rows = audit_archive(source / name)
        unique = len({row["seq"] for row in rows})
        if (len(rows), unique) != (expected_rows, expected_unique):
            raise ValueError(f"Archive count mismatch: {name}: {(len(rows), unique)}")
        website_path = root / "official_download" / name
        if sha256(source / name) != sha256(website_path):
            raise ValueError(f"Website/GitHub archive mismatch: {name}")
        archives[name] = {
            "official_github_url": f"{OFFICIAL_REPOSITORY}/blob/{commit}/{name}",
            "official_download_url": OFFICIAL_WEB + "data_download/" + name,
            "git_path": str(source / name),
            "website_path": str(website_path),
            "sha256": sha256(source / name),
            "n_structures": len(rows),
            "n_unique_sequences": unique,
            "scope": "training/validation provenance" if name == "TrainingValidationSet.zip" else "official test archive; audited only, never used for tuning",
        }
    dataset = load_dataset(dataset_path)
    paper_xml = root / "paper/PMC11094721.fulltext.xml"
    paper_text = paper_xml.read_text()
    required_protocol_tokens = ("training-validation", "TestSet", "cosine", "Adam", "512", "250", "1.8", "RNAfold", "IPknot", "MaxExpect", "ProbKnot", "RNAprob")
    if not all(token in paper_text for token in required_protocol_tokens):
        raise ValueError("Paper full text does not contain the expected protocol")
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "baseline_id": "debfold",
        "status": "BLOCKED_OFFICIAL_IMPLEMENTATION_AND_WEIGHTS",
        "full_ready": False,
        "ready_scope": "Official source snapshot, data archives, paper protocol, bpRNA-new I/O contract and fail-closed GPU launcher",
        "blocked_scope": "No official local implementation, checkpoint/weights, checkpoint SHA, model/weight license, official batch CLI, or author-released bpRNA-new 5388 predictions",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": commit,
        "official_tree": sorted(tree),
        "official_releases": releases,
        "official_tags": tags,
        "official_web": OFFICIAL_WEB,
        "official_archives": archives,
        "weights": {"status": "NOT_PUBLISHED_IN_OFFICIAL_REPOSITORY_OR_WEB_DOWNLOAD", "url": None, "path": None, "sha256": None},
        "implementation": {"status": "NOT_PUBLISHED_IN_OFFICIAL_REPOSITORY_OR_WEB_DOWNLOAD", "source_commit": commit, "local_inference_entrypoint": None},
        "license": {"status": "NOT_SPECIFIED", "spdx": None, "github_license_api_http_status": license_status, "paper_license_cc_by_applies_to_paper_only": True},
        "paper": {"title": "DEBFold: Computational Identification of RNA Secondary Structures for Sequences across Structural Families Using Deep Learning", "doi": PAPER_DOI, "pmcid": "PMC11094721", "url": PAPER_URL, "fulltext_xml": str(paper_xml), "fulltext_xml_sha256": sha256(paper_xml)},
        "training_protocol": {
            "source": "PMC11094721 Methods and Data Sets",
            "input": "l x 10: l x 4 RNA one-hot plus six l x 1 paired-location tracks from RNAfold, IPknot, MaxExpect, ProbKnot, RNAprob and Fold",
            "stage1_output": "l folding probabilities Y; SHAPE-like score S=1-Y",
            "stage2": "RNAstructure Fold MFE with SHAPE-like soft constraints; default slope/intercept",
            "training_padding": 512,
            "prediction_padding": None,
            "length_protocol": "Methods: no prediction padding and <512 nt suggested; Limitations: current version suitable only for sequences up to 512 bps",
            "cross_validation_folds": 25,
            "epochs": 250,
            "batch_size": 512,
            "optimizer": "Adam",
            "loss": "sum of per-base binary cross-entropy; positive weight 1.8",
            "learning_rate": "cosine decay to 1e-13 for epochs 1-227, then exponential decay with multiplicative factor 1e-10 for 23 epochs",
            "dropout": {"residual": 0.1, "convolution_and_transpose_convolution": 0.4},
            "attention": {"blocks": 2, "heads": 16, "query_key_value_dimension": 512, "feed_forward_dimension": 1024},
            "training_gpu_reported_by_paper": "NVIDIA RTX 4080",
        },
        "dataset_protocol": {
            "training_validation": "bpRNA-1m/Rfam12.2; 43269 valid sequences/2125 families; reserve 44 families; 2081 training-validation families; at most 5 sequences/family; 8782 structures",
            "test_alpha": "44 sequences, one per reserved family; CD-HIT 0.85 separation",
            "test_beta": "bpRNA-new/Rfam14.4 excluding Rfam12.2; 481 families/5265 sequences eligible, then 4 families from each of 10 length bins = 40 sequences; CD-HIT 0.85 separation",
            "test_gamma": "PDB-derived; 147 eligible before length balancing; 15 sequences; excludes >85% similarity to DEBFold training-validation and SPOT-RNA TR1",
            "external_test_used_for_tuning": False,
        },
        "bprna_new_5388_contract": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256(dataset_path),
            "n_sequences": len(dataset),
            "unique_names": len({row["name"] for row in dataset}),
            "length_min": min(len(row["seq"]) for row in dataset),
            "length_max": max(len(row["seq"]) for row in dataset),
            "n_over_512": sum(len(row["seq"]) > 512 for row in dataset),
            "over_512_requires_verified_official_protocol": True,
            "input": "JSONL name/seq; project reference pairs retained only for later evaluation, never official-model tuning",
            "output": "One JSONL row per input: {name,seq,pred_pairs}; zero-based inclusive i<j; complete 5388-name/sequence identity required",
            "predictions": None,
            "prediction_sha256": None,
            "coverage": None,
            "metrics": None,
            "official_test_beta_40_is_not_project_5388": True,
            "summary_only_substitution_forbidden": True,
        },
        "gpu_execution": {"performed": False, "reason": "Official implementation/weights absent and occupied GPU must not be disturbed", "launcher": "tools/run_debfold_bprna_new_gpu_20261126.py"},
        "provenance_policy": "No unrelated same-name repository, paper-derived reimplementation, pseudo-weights, summary-only score, reference labels, or zero scores substituted",
    }
    write_json(root / "reports/debfold_official_provenance_20261126.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "audit"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_debfold_recovery"))
    args = parser.parse_args()
    result = fetch(args.root) if args.command == "fetch" else audit(args.root, args.dataset)
    if args.command == "audit":
        update_unified_registry(args.repo)
        write_json(args.repo / "manifests/foldnar_v2/debfold_official_recovery_20261126.json", result)
    print(json.dumps({"status": result.get("status", "PASS_DOWNLOAD"), "root": str(args.root)}, indent=2))


if __name__ == "__main__":
    main()
