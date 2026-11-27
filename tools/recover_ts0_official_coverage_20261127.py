#!/usr/bin/env python3
"""Recover TS0 official-full frozen baselines without rerunning the projected 1288."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
PANEL = PANEL_ROOT / "bprna_ts0_official_1305.jsonl.gz"
PROJECT = PANEL_ROOT / "bprna_ts0_projected_1288.jsonl.gz"
OUT = ROOT / "source_data/ts0_official_coverage_recovery_20261127"
ART = ROOT / "artifacts/ts0_official_coverage_recovery_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-ts0-official-coverage-recovery.md"
SUPPLEMENT = OUT / "missing17_cpu_predictions.json"
MANIFEST = OUT / "recovery_manifest.json"
RELEASE = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz")
RELEASE_SHA = "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073"
PANEL_SHA = "c8e82e25063ec5c2c09b1a38e28ea499b8c0f94b4d35fef9d93b3ac7cd44d9ae"
PROJECT_SHA = "c8a48651e6e64f96c049609aafe531455e36f65f1cc5354d371b68430b98466e"
PROJECT_DATA_SHA = "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882"
BENCH = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
RAW = {
    "vienna": (BENCH / "vienna/raw_strict/bprna_ts0.json", "d76865f8c8cfb630bc0335094fbe5f45a7cbf31e963857d0f87487fd5f073e32"),
    "contrafold": (BENCH / "contrafold/raw/bprna_ts0.json", "0a402bceba38d5fede0835cb6e986a7d556c9ca46c9699ceee59555cb2331ec4"),
    "eternafold": (BENCH / "eternafold/raw/bprna_ts0.json", "6e25763a001fecef5158a9e2669ac4bdaad63519b6d8d7604450e7817dc44c79"),
    "rnastructure": (BENCH / "rnastructure/raw/bprna_ts0.json", "27e4bafdabf85fea47da235e0caf435511e32d422536f0357485d6b49496b69f"),
}
SYSTEMS = (
    "vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner",
    "contrafold", "eternafold", "rnastructure_fold", "rnastructure_maxexpect",
)
REPLAY_INDICES = (0, 17, 100, 501, 1000, 1287)
DETERMINISTIC = set(SYSTEMS)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                text.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gzip_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def pair_set(values, length):
    result, occupied = set(), set()
    for value in values:
        i, j = map(int, value)
        if not 0 <= i < j < length or i in occupied or j in occupied:
            raise ValueError(f"invalid/conflicting pair {(i, j)} for length {length}")
        occupied.update((i, j))
        result.add((i, j))
    return result


def parse_bpseq(payload):
    bases, partners = [], []
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or int(fields[0]) != len(bases) + 1:
            raise ValueError("invalid BPSEQ")
        bases.append(fields[1].upper().replace("T", "U"))
        partners.append(int(fields[2]))
    pairs = set()
    for i, partner in enumerate(partners):
        if partner:
            if not 1 <= partner <= len(bases) or partners[partner - 1] != i + 1:
                raise ValueError("BPSEQ partner mismatch")
            if partner != i + 1:
                pairs.add(tuple(sorted((i, partner - 1))))
    return "".join(bases), pair_set(pairs, len(bases))


def raw_predictions(family, document):
    if family == "vienna":
        return {system: {row["name"]: row["pred_pairs"] for row in document["results"][system]["per_sequence"]}
                for system in SYSTEMS[:4]}
    if family == "rnastructure":
        return {f"rnastructure_{key}": {row["name"]: row["pred_pairs"] for row in document["results"][key]["per_sequence"]}
                for key in ("fold", "maxexpect")}
    return {family: {row["name"]: row["pred_pairs"] for row in document["per_sequence"]}}


def parse_dot(structure, length):
    if len(structure) != length:
        raise ValueError("dot-bracket length mismatch")
    stack, pairs = [], []
    for index, token in enumerate(structure):
        if token == "(":
            stack.append(index)
        elif token == ")":
            if not stack:
                raise ValueError("dot-bracket underflow")
            pairs.append((stack.pop(), index))
        elif token != ".":
            raise ValueError("unexpected dot-bracket token")
    if stack:
        raise ValueError("dot-bracket overflow")
    return sorted(pairs)


def predict_contra(binary, params, row, directory, env):
    sequence_path = directory / "input.seq"
    sequence_path.write_text(row["seq"] + "\n")
    command = [str(binary), "predict", str(sequence_path)]
    if params is not None:
        command.extend(["--params", str(params)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=300)
    lines = completed.stdout.splitlines()
    structure = lines[lines.index(">structure") + 1].strip()
    return parse_dot(structure, len(row["seq"]))


def predict_rnastructure(binary, row, directory, env):
    fasta = directory / "input.fa"
    ct = directory / "output.ct"
    fasta.write_text(f">{row['name']}\n{row['seq']}\n")
    ct.unlink(missing_ok=True)
    flag = "--MFE" if binary.name == "Fold" else "--sequence"
    subprocess.run([str(binary), str(fasta), str(ct), flag], check=True, capture_output=True,
                   text=True, env=env, timeout=300)
    lines = ct.read_text().splitlines()
    length = len(row["seq"])
    block_size = length + 1
    if not lines or len(lines) % block_size:
        raise ValueError(f"invalid CT blocks for {row['name']}")
    first = lines[:block_size]
    pairs = []
    for index, line in enumerate(first[1:], 1):
        fields = line.split()
        if len(fields) != 6 or int(fields[0]) != index:
            raise ValueError("invalid CT record")
        if fields[1].upper().replace("T", "U") != row["seq"][index - 1].upper().replace("T", "U"):
            raise ValueError("CT sequence mismatch")
        partner = int(fields[4]) - 1
        if partner > index - 1:
            pairs.append((index - 1, partner))
    return sorted(pair_set(pairs, length))


def cpu_predictors(directory, env):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import RNA
    from eval.ss.run_baselines import nussinov_turner, vienna_structure

    if RNA.__version__ != "2.7.2":
        raise ValueError("ViennaRNA version changed")
    contra = Path("/home/rna_jev/third_party/EternaFold/src/contrafold")
    eterna_params = Path("/home/rna_jev/third_party/EternaFold/parameters/EternaFoldParams.v1")
    rna_exe = Path("/home/rna_jev/packages/rnastructure-6.6/RNAstructure/exe")
    return {
        "vienna_mfe": lambda row: vienna_structure(row["seq"], "vienna_mfe", RNA),
        "vienna_centroid": lambda row: vienna_structure(row["seq"], "vienna_centroid", RNA),
        "vienna_mea": lambda row: vienna_structure(row["seq"], "vienna_mea", RNA),
        "nussinov_turner": lambda row: nussinov_turner(row["seq"]),
        "contrafold": lambda row: predict_contra(contra, None, row, directory, env),
        "eternafold": lambda row: predict_contra(contra, eterna_params, row, directory, env),
        "rnastructure_fold": lambda row: predict_rnastructure(rna_exe / "Fold", row, directory, env),
        "rnastructure_maxexpect": lambda row: predict_rnastructure(rna_exe / "MaxExpect", row, directory, env),
    }


def generate_supplement(full, projected, raw_maps):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["DATAPATH"] = "/home/rna_jev/packages/rnastructure-6.6/RNAstructure/data_tables"
    missing = [row for row in full if row["name"] not in {item["name"] for item in projected}]
    observations = {system: [] for system in SYSTEMS}
    results = {system: {"per_sequence": [], "repeat_prediction_sha256": []} for system in SYSTEMS}
    with tempfile.TemporaryDirectory(prefix="ts0_official17_cpu_") as temporary:
        predictors = cpu_predictors(Path(temporary), env)
        for index in REPLAY_INDICES:
            row = projected[index]
            for system, predictor in predictors.items():
                actual = sorted(pair_set(predictor(row), len(row["seq"])))
                expected = sorted(pair_set(raw_maps[system][row["name"]], len(row["seq"])))
                if actual != expected:
                    raise ValueError(f"frozen replay mismatch {system}/{row['name']}")
                observations[system].append({"name": row["name"], "pairs_exact": True})
        for system, predictor in predictors.items():
            reference = None
            blocked = {}
            for repeat in range(3):
                current = []
                for row in missing:
                    if row["name"] in blocked:
                        continue
                    try:
                        pairs = sorted(pair_set(predictor(row), len(row["seq"])))
                    except (subprocess.CalledProcessError, FileNotFoundError) as error:
                        message = (error.stderr or "").strip() if isinstance(error, subprocess.CalledProcessError) else "official tool produced no CT for R/Y sequence"
                        supported_blocker = "Invalid nucleobase" in message or (isinstance(error, FileNotFoundError) and set(row["seq"]) & set("RY"))
                        if not system.startswith("rnastructure_") or not supported_blocker or repeat:
                            raise
                        blocked[row["name"]] = {"name": row["name"], "status": "BLOCKED_OFFICIAL_TOOL_INVALID_NUCLEOBASE",
                                                "tool_error": message.splitlines()[0]}
                        continue
                    current.append({"name": row["name"], "seq": row["seq"],
                                    "sequence_sha256": row["sequence_sha256"], "pred_pairs": pairs})
                identity = [[row["name"], row["pred_pairs"]] for row in current]
                digest = value_sha(identity)
                results[system]["repeat_prediction_sha256"].append(digest)
                if reference is None:
                    reference = current
                elif current != reference:
                    raise ValueError(f"nondeterministic CPU prediction {system} repeat {repeat}")
            results[system]["per_sequence"] = reference
            results[system]["blocked_records"] = list(blocked.values())
    return {
        "schema_version": "ts0.official_coverage_recovery.missing17_cpu.v1",
        "status": "PASS_17_AUDITED_CPU_WITH_EXPLICIT_UNSUPPORTED",
        "device": "CPU_ONLY", "gpu_used": False, "cuda_visible_devices": "",
        "cpu_threads": 1, "missing_records": len(missing), "deterministic_repeats": 3,
        "replay_indices": list(REPLAY_INDICES), "frozen_replay": observations,
        "systems": results,
        "protocols": {
            "vienna": "ViennaRNA 2.7.2 RNA.fold / fold_compound.pf.centroid / MEA",
            "nussinov_turner": "project Turner score matrix MLP_T=0 with max-product decode",
            "contrafold": "official serial contrafold predict native parameters, posterior MEA",
            "eternafold": "official serial contrafold predict --params EternaFoldParams.v1, posterior MEA",
            "rnastructure_fold": "RNAstructure 6.6 Fold --MFE; single CT block",
            "rnastructure_maxexpect": "RNAstructure 6.6 MaxExpect --sequence; first CT block",
        },
    }


def registered_coverage(prior_manifest):
    answer = {}
    for entry in prior_manifest["datasets"]["bprna_ts0"]["predictions"]:
        if not entry["available"] or "ts0_official_coverage_recovery_20261127" in entry["path"]:
            continue
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"registered raw SHA changed {entry['prediction_id']}")
        document = load_json(path)
        if entry["prediction_id"].startswith("rnastructure_fold/"):
            source = document["results"]["fold"]["per_sequence"]
        elif entry["prediction_id"].startswith("rnastructure_maxexpect/"):
            source = document["results"]["maxexpect"]["per_sequence"]
        else:
            source = document.get("per_sequence", document.get("predictions"))
        answer[entry["prediction_id"]] = {row["name"] for row in source}
    return answer


def normalized_entry(prediction_id, baseline_id, source_group, origin, path):
    return {
        "prediction_id": prediction_id, "baseline_id": baseline_id, "source_group": source_group,
        "registry_status": "FROZEN", "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "data_path": str(PANEL.relative_to(ROOT)), "available": True,
        "model_form": "six_checkpoint_ensemble" if prediction_id.startswith("bpfold_official_release/")
        else "deterministic_single_predictor", "prediction_origin": origin,
    }


def render_report(audit):
    lines = [
        "# TS0 官方 1305 覆盖恢复", "",
        "仅恢复最终 frozen baseline 覆盖，不训练、不选模。BPfold 使用作者官方 release 的 1305 条 BPSEQ；Vienna/Nussinov、CONTRAfold、EternaFold 和 RNAstructure 保留原冻结 1288 条，只对官方多出的 17 条做 CPU 补充。其他神经模型没有 full1305 原始资产，保持部分覆盖或阻塞，不使用 GPU。", "",
        "所有输出按官方未过滤 BPSEQ 面板统一 ID、序列和序列 SHA256。部分覆盖不排名、不补 0；仅 truth 与 prediction 同时为空时 P/R/F1=1。唯一 truth-empty 为 `bpRNA_SRP_192.bpseq`。RNAstructure 官方工具拒绝 3 条含 R/Y 的序列，Fold/MaxExpect 因此保持 1302/1305 部分覆盖，不改写碱基。", "",
        "| Full1305 prediction identity | 来源 | 记录数 |",
        "|---|---|---:|",
    ]
    for entry in audit["prediction_entries"]:
        lines.append(f"| `{entry['prediction_id']}` | {entry['prediction_origin']} | {entry['n_prediction_records']} |")
    lines += ["", "## 逐 ID 资产核查", "",
              "每条均先核查既有冻结预测与 BPfold 官方 release；`existing_frozen_coverage` 为空表示既有原始预测没有该 ID，不以空结构代替缺失。", "",
              "| Official-only ID | 非 ACGU | truth-empty | 既有冻结覆盖 | BPfold release | 最终全覆盖身份数 |",
              "|---|---|---:|---|---|---:|"]
    for row in audit["per_id_audit"]:
        lines.append(f"| `{row['name']}` | {''.join(row['non_acgu_characters'])} | {str(row['truth_empty']).lower()} | {len(row['existing_frozen_coverage'])} | {row['bpfold_official_release']['member']} | {len(row['final_full_coverage_identities'])} |")
    lines += ["", "CPU 补充前的有界回放索引为 `0, 17, 100, 501, 1000, 1287`，8 个 decoder 均逐对精确匹配原冻结预测；17 条补充均做 3 次确定性重复。面板重算本身不推理。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-missing17", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sha256(PANEL) != PANEL_SHA or sha256(PROJECT) != PROJECT_SHA or sha256(RELEASE) != RELEASE_SHA:
        raise ValueError("panel or release identity changed")
    full, projected = gzip_rows(PANEL), gzip_rows(PROJECT)
    if len(full) != 1305 or len(projected) != 1288:
        raise ValueError("panel population changed")
    full_by_name = {row["name"]: row for row in full}
    project_names = {row["name"] for row in projected}
    missing = [row for row in full if row["name"] not in project_names]
    if len(missing) != 17 or sum(not row["pairs"] for row in missing) != 1:
        raise ValueError("official-only population changed")
    documents = {}
    raw_maps = {}
    for family, (path, expected) in RAW.items():
        if sha256(path) != expected:
            raise ValueError(f"raw source changed {family}")
        document = load_json(path)
        if document["data_sha256"] != PROJECT_DATA_SHA or document["n_sequences"] != 1288:
            raise ValueError(f"raw data contract changed {family}")
        documents[family] = document
        raw_maps.update(raw_predictions(family, document))
    for system in SYSTEMS:
        if set(raw_maps[system]) != project_names:
            raise ValueError(f"raw coverage changed {system}")
    if args.generate_missing17:
        write_json(SUPPLEMENT, generate_supplement(full, projected, raw_maps))
    if not SUPPLEMENT.is_file():
        raise FileNotFoundError("missing17 supplement is blocked; explicit --generate-missing17 is required")
    supplement = load_json(SUPPLEMENT)
    if supplement["status"] != "PASS_17_AUDITED_CPU_WITH_EXPLICIT_UNSUPPORTED" or supplement["gpu_used"]:
        raise ValueError("invalid CPU supplement")
    missing_names = {row["name"] for row in missing}
    for system in SYSTEMS:
        rows = supplement["systems"][system]["per_sequence"]
        blocked = supplement["systems"][system].get("blocked_records", [])
        row_names = {row["name"] for row in rows}
        blocked_names = {row["name"] for row in blocked}
        if row_names.isdisjoint(blocked_names) and row_names | blocked_names == missing_names:
            pass
        else:
            raise ValueError(f"supplement coverage changed {system}")
        if any(row["status"] != "BLOCKED_OFFICIAL_TOOL_INVALID_NUCLEOBASE" for row in blocked):
            raise ValueError(f"unexpected supplement blocker {system}")
        if len(set(supplement["systems"][system]["repeat_prediction_sha256"])) != 1:
            raise ValueError(f"supplement repeats differ {system}")
        for row in rows:
            ref = full_by_name[row["name"]]
            if row["seq"] != ref["seq"] or row["sequence_sha256"] != ref["sequence_sha256"]:
                raise ValueError(f"supplement sequence mismatch {system}/{row['name']}")
            pair_set(row["pred_pairs"], len(row["seq"]))
    entries = []
    release_members = {}
    with tarfile.open(RELEASE, "r:gz") as archive:
        members = [member for member in archive if member.isfile() and member.name.startswith("BPfold_test_results/bpRNA/") and member.name.endswith(".bpseq")]
        if len(members) != 1305:
            raise ValueError("BPfold release population changed")
        release_predictions = {}
        for member in members:
            name = Path(member.name).name
            payload = archive.extractfile(member).read()
            sequence, pairs = parse_bpseq(payload)
            ref = full_by_name.get(name)
            if ref is None or sequence != ref["seq"]:
                raise ValueError(f"BPfold release sequence/ID mismatch {name}")
            release_predictions[name] = sorted(pairs)
            if name not in project_names:
                release_members[name] = {"covered": True, "member": member.name,
                                         "member_sha256": hashlib.sha256(payload).hexdigest()}
    if set(release_predictions) != set(full_by_name):
        raise ValueError("BPfold release ID set mismatch")
    prediction_sets = {"bpfold_official_release/bprna_ts0": release_predictions}
    baseline_ids = {"bpfold_official_release/bprna_ts0": "bpfold"}
    origins = {"bpfold_official_release/bprna_ts0": "AUTHOR_BPFOLD_TEST_RESULTS_V0_2_FULL1305"}
    for system in SYSTEMS:
        prediction_id = system if system in {"contrafold", "eternafold", "rnastructure_fold", "rnastructure_maxexpect"} else f"{system}/bprna_ts0/recovered_official17"
        if "/" not in prediction_id:
            prediction_id += "/bprna_ts0"
        merged = dict(raw_maps[system])
        merged.update({row["name"]: row["pred_pairs"] for row in supplement["systems"][system]["per_sequence"]})
        prediction_sets[prediction_id] = merged
        baseline_ids[prediction_id] = system
        origins[prediction_id] = "FROZEN_RAW1288_PLUS_OFFICIAL17_CPU_COMPLETION"
    for prediction_id, predictions in prediction_sets.items():
        rows = []
        for ref in full:
            if ref["name"] not in predictions:
                continue
            pairs = sorted(pair_set(predictions[ref["name"]], len(ref["seq"])))
            rows.append({"name": ref["name"], "seq": ref["seq"],
                         "sequence_sha256": ref["sequence_sha256"], "pred_pairs": pairs})
        filename = prediction_id.replace("/", "__") + f"__{'full1305' if len(rows) == 1305 else f'partial{len(rows)}'}.json.gz"
        path = OUT / filename
        document = {
            "schema_version": "ts0.official_coverage_recovery.prediction.v1",
            "prediction_id": prediction_id, "dataset_id": "bprna_ts0",
            "data": str(PANEL.relative_to(ROOT)), "data_sha256": PANEL_SHA,
            "n_sequences": len(rows), "prediction_origin": origins[prediction_id], "per_sequence": rows,
        }
        write_gzip_json(path, document)
        group = "AUTHOR_OFFICIAL_RELEASE" if prediction_id.startswith("bpfold_official_release/") else "FROZEN_WITH_OFFICIAL17_CPU_COMPLETION"
        entry = normalized_entry(prediction_id, baseline_ids[prediction_id], group, origins[prediction_id], path)
        entry["n_prediction_records"] = len(rows)
        entries.append(entry)
    prior_manifest = json.loads((PANEL_ROOT / "source_manifest.json").read_text())
    existing = registered_coverage(prior_manifest)
    per_id = []
    final_ids = [entry["prediction_id"] for entry in entries if entry["n_prediction_records"] == 1305]
    partial_ids = [entry["prediction_id"] for entry in entries if entry["n_prediction_records"] < 1305]
    for ref in missing:
        name = ref["name"]
        per_id.append({
            "name": name, "sequence_sha256": ref["sequence_sha256"], "pair_set_sha256": ref["pair_set_sha256"],
            "non_acgu_characters": sorted(set(ref["seq"]) - set("ACGU")), "truth_empty": not ref["pairs"],
            "existing_frozen_coverage": sorted(pid for pid, names in existing.items() if name in names),
            "vienna_raw_strict_coverage": sorted(system for system in SYSTEMS[:4] if name in raw_maps[system]),
            "bpfold_official_release": release_members[name],
            "final_full_coverage_identities": [pid for pid, predictions in prediction_sets.items() if name in predictions],
            "blocked_recovered_identities": [pid for pid, predictions in prediction_sets.items() if name not in predictions],
        })
    if any(row["existing_frozen_coverage"] or row["vienna_raw_strict_coverage"] for row in per_id):
        raise ValueError("official-only ID unexpectedly present in original raw predictions")
    manifest = {
        "schema_version": "ts0.official_coverage_recovery.manifest.v1",
        "status": f"PASS_{len(final_ids)}_FULL1305_IDENTITIES_{len(partial_ids)}_PARTIAL", "dataset_id": "bprna_ts0",
        "official_panel": {"path": str(PANEL.relative_to(ROOT)), "sha256": PANEL_SHA, "n_sequences": 1305},
        "project_panel": {"path": str(PROJECT.relative_to(ROOT)), "sha256": PROJECT_SHA, "n_sequences": 1288},
        "official_only_n": 17, "truth_empty_name": "bpRNA_SRP_192.bpseq",
        "release": {"path": str(RELEASE), "sha256": RELEASE_SHA, "prefix": "BPfold_test_results/bpRNA/", "n_sequences": 1305},
        "raw_sources": {family: {"path": str(path), "sha256": expected} for family, (path, expected) in RAW.items()},
        "supplement": {"path": str(SUPPLEMENT.relative_to(ROOT)), "sha256": sha256(SUPPLEMENT),
                       "new_inference_records_per_system": 17, "projected_records_rerun": 0,
                       "bounded_replay_indices": list(REPLAY_INDICES), "deterministic_repeats": 3,
                       "device": "CPU_ONLY", "gpu_used": False},
        "prediction_entries": entries, "new_full_coverage_identities": final_ids,
        "recovered_partial_identities": partial_ids,
        "upgraded_existing_identities": [f"{system}/bprna_ts0" for system in ("contrafold", "eternafold")],
        "recovered_partial_existing_identities": [f"{system}/bprna_ts0" for system in ("rnastructure_fold", "rnastructure_maxexpect")],
        "added_identities": ["bpfold_official_release/bprna_ts0"] + [f"{system}/bprna_ts0/recovered_official17" for system in SYSTEMS[:4]],
        "unrecovered_neural_models": "NO_FULL1305_RAW_ASSET; PARTIAL_OR_BLOCKED; NO_GPU_INFERENCE",
        "per_id_audit": per_id, "external_test_model_selection": False,
    }
    write_json(MANIFEST, manifest)
    write_json(ART / "recovery_audit.json", manifest)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "full1305_identities": len(final_ids),
                      "official_only": len(missing), "truth_empty": manifest["truth_empty_name"],
                      "generated_missing17": args.generate_missing17, "gpu_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
