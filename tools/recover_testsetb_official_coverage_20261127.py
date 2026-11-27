#!/usr/bin/env python3
"""Recover TestSetB official-full frozen baselines without rerunning project 428."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
PANEL = PANEL_ROOT / "testsetb_official_430.jsonl.gz"
PROJECT = PANEL_ROOT / "testsetb_projected_428.jsonl.gz"
PRIOR = ROOT / "artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json"
PRIOR_MANIFEST = PANEL_ROOT / "source_manifest.json"
OUT = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127"
SOURCE = ROOT / "source_data/testsetb_official_coverage_recovery_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-testsetb-official-coverage-recovery.md"
ART = Path("/teams/HoulabData_1788502323/liucunyu/foldnar/testsetb_official_coverage_recovery_20261127")
SUPPLEMENT = ART / "missing2_cpu_predictions.json"
MANIFEST = ART / "recovery_manifest.json"
PREDICTIONS = ART / "predictions"
PANEL_SHA = "aed077ce5a913476c79084733d03e46ae4dd58c1cef1f1998b60f6cda4b86727"
PROJECT_SHA = "7bc6db613d8bf6d8099fe17283e3d25fe7725c84811fa5b11b7dd228dc5bbf44"
PROJECT_DATA_SHA = "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06"
BENCH = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
RAW = {
    "vienna": (BENCH / "vienna/raw_strict/testsetb.json", "b11fa255ad74bd5c007b111e421ae69b5dedb068bbaf492660f089bbc4345d25"),
    "contrafold": (BENCH / "contrafold/raw/testsetb.json", "1f3e14785abb9fbde0c3e2fcf953dd535f482804c621be746a6db86899cd9bd2"),
    "eternafold": (BENCH / "eternafold/raw/testsetb.json", "13185fe2545d8cad000406f6061facb62cdc544fbae699d1517979f811dc3556"),
    "mxfold2": (BENCH / "mxfold2/raw/testsetb.json", "921f9e5170726cb9f6f9014902a60446038fcac74c69e03c585f3fd384a1f452"),
    "rnastructure": (BENCH / "rnastructure/raw/testsetb.json", "271d1d8dae18897ca9efcbdc2daa302c131e09e260fdc446ed8ffab792d11f5d"),
}
SYSTEMS = (
    "vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner",
    "contrafold", "eternafold", "mxfold2", "rnastructure_fold", "rnastructure_maxexpect",
)
UPGRADED = ("contrafold", "eternafold", "mxfold2", "rnastructure_fold", "rnastructure_maxexpect")
ADDED = SYSTEMS[:4]
REPLAY_INDICES = (0, 17, 100, 200, 300, 427)
MX_PARAM = Path("/root/miniconda3/envs/lucaone/lib/python3.10/site-packages/mxfold2/models/TrainSetAB.pth")
MX_CONF = Path("/root/miniconda3/envs/lucaone/lib/python3.10/site-packages/mxfold2/models/TrainSetAB.conf")
MX_PARAM_SHA = "8ba1283e70c1c073033161adc120f67cdc467176476358d26a12cadaf63262ec"
EXPECTED_FULL = {f"{system}/testsetb" for system in ("contrafold", "eternafold", "mxfold2")} | {f"{system}/testsetb/recovered_official2" for system in ADDED}


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                text.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical(row) + "\n")
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


def raw_predictions(family, document):
    if family == "vienna":
        return {system: {row["name"]: row["pred_pairs"] for row in document["results"][system]["per_sequence"]}
                for system in ADDED}
    if family == "rnastructure":
        return {f"rnastructure_{key}": {row["name"]: row["pred_pairs"] for row in document["results"][key]["per_sequence"]}
                for key in ("fold", "maxexpect")}
    if family == "mxfold2":
        return {family: {row["name"]: row["pairs"] for row in document["predictions"]}}
    return {family: {row["name"]: row["pred_pairs"] for row in document["per_sequence"]}}


def predictions_from_entry(entry):
    document = load_json(entry["path"])
    prediction_id = entry["prediction_id"]
    if "results" not in document and "per_sequence" in document:
        source = document["per_sequence"]
    elif prediction_id.startswith("rnastructure_fold/"):
        source = document["results"]["fold"]["per_sequence"]
    elif prediction_id.startswith("rnastructure_maxexpect/"):
        source = document["results"]["maxexpect"]["per_sequence"]
    else:
        source = document.get("per_sequence", document.get("predictions"))
    if source is None:
        raise ValueError(f"record-level predictions required: {prediction_id}")
    result = {}
    for row in source:
        pairs = row.get("pred_pairs", row.get("pairs"))
        if row["name"] in result or pairs is None:
            raise ValueError(f"duplicate/missing pairs: {prediction_id}")
        result[row["name"]] = pairs
    return result


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
    pairs = []
    for index, line in enumerate(lines[1:block_size], 1):
        fields = line.split()
        if len(fields) != 6 or int(fields[0]) != index:
            raise ValueError("invalid CT record")
        if fields[1].upper().replace("T", "U") != row["seq"][index - 1].upper().replace("T", "U"):
            raise ValueError("CT sequence mismatch")
        partner = int(fields[4]) - 1
        if partner > index - 1:
            pairs.append((index - 1, partner))
    return sorted(pair_set(pairs, length))


def predict_mxfold2(row, directory, env):
    fasta = directory / "mxfold2.fa"
    fasta.write_text(f">{row['name']}\n{row['seq']}\n")
    command = ["/root/miniconda3/envs/lucaone/bin/python", "-m", "mxfold2", "predict",
               str(fasta), "--gpu", "-1", "--seed", "0", "@" + str(MX_CONF)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=300)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    structures = [line.split()[0] for line in lines if line and set(line.split()[0]) <= set(".()")]
    if len(structures) != 1:
        raise ValueError(f"MXfold2 output structure count={len(structures)}")
    return parse_dot(structures[0], len(row["seq"]))


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
        "mxfold2": lambda row: predict_mxfold2(row, directory, env),
        "rnastructure_fold": lambda row: predict_rnastructure(rna_exe / "Fold", row, directory, env),
        "rnastructure_maxexpect": lambda row: predict_rnastructure(rna_exe / "MaxExpect", row, directory, env),
    }


def generate_supplement(full, projected, raw_maps):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["DATAPATH"] = "/home/rna_jev/packages/rnastructure-6.6/RNAstructure/data_tables"
    missing = [row for row in full if row["name"] not in {item["name"] for item in projected}]
    observations = {system: [] for system in SYSTEMS}
    results = {system: {"per_sequence": [], "blocked": [], "repeat_prediction_sha256": []} for system in SYSTEMS}
    with tempfile.TemporaryDirectory(prefix="testsetb_official2_cpu_") as temporary:
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
            reference_blocked = None
            for repeat in range(3):
                current = []
                blocked = []
                for row in missing:
                    try:
                        pairs = sorted(pair_set(predictor(row), len(row["seq"])))
                    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as error:
                        detail = ""
                        if isinstance(error, subprocess.CalledProcessError):
                            detail = (error.stderr or error.stdout or "").strip().splitlines()
                            invalid = [line for line in detail if "Invalid nucleobase" in line]
                            detail = invalid[0] if invalid else detail[-1] if detail else str(error)
                        elif isinstance(error, FileNotFoundError):
                            detail = "Official CPU tool rejected raw S input and produced no CT output"
                        else:
                            detail = str(error)
                        blocked.append({"name": row["name"], "status": "BLOCKED_OFFICIAL_CPU_TOOL_REJECTED_INPUT",
                                        "detail": detail, "sequence_sha256": row["sequence_sha256"]})
                        continue
                    current.append({"name": row["name"], "seq": row["seq"],
                                    "sequence_sha256": row["sequence_sha256"], "pred_pairs": pairs})
                identity = {"predictions": [[row["name"], row["pred_pairs"]] for row in current],
                            "blocked": [row["name"] for row in blocked]}
                results[system]["repeat_prediction_sha256"].append(value_sha(identity))
                if reference is None:
                    reference = current
                    reference_blocked = blocked
                elif current != reference or blocked != reference_blocked:
                    raise ValueError(f"nondeterministic CPU prediction/block {system} repeat {repeat}")
            results[system]["per_sequence"] = reference
            results[system]["blocked"] = reference_blocked
    return {
        "schema_version": "testsetb.official_coverage_recovery.missing2_cpu.v1",
        "status": "PASS_2_ONLY_CPU_AND_BOUNDED_FROZEN_REPLAY_WITH_EXPLICIT_BLOCKS",
        "device": "CPU_ONLY", "gpu_used": False, "cuda_visible_devices": "",
        "mxfold2_gpu_argument": -1, "cpu_threads": 1, "missing_records": len(missing),
        "deterministic_repeats": 3, "replay_indices": list(REPLAY_INDICES),
        "frozen_replay": observations, "systems": results,
        "protocols": {
            "vienna": "ViennaRNA 2.7.2 RNA.fold / fold_compound.pf.centroid / MEA",
            "nussinov_turner": "project Turner score matrix MLP_T=0 with max-product decode",
            "contrafold": "official serial contrafold predict native parameters, posterior MEA",
            "eternafold": "official serial contrafold predict --params EternaFoldParams.v1, posterior MEA",
            "mxfold2": "official @TrainSetAB.conf + TrainSetAB.pth; python -m mxfold2 predict --gpu -1 --seed 0",
            "rnastructure_fold": "RNAstructure 6.6 Fold --MFE; single CT block",
            "rnastructure_maxexpect": "RNAstructure 6.6 MaxExpect --sequence; first CT block",
        },
        "mxfold2_parameter": {"path": str(MX_PARAM), "sha256": MX_PARAM_SHA},
        "mxfold2_config": {"path": str(MX_CONF), "sha256": sha256(MX_CONF)},
    }


def prf(truth, predicted):
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    if not truth and not predicted:
        return tp, fp, fn, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1


def score(panel, predictions):
    totals = [0, 0, 0]
    values, records, missing = [], {}, []
    panel_names = {row["name"] for row in panel}
    for label in panel:
        name = label["name"]
        if name not in predictions:
            missing.append(name)
            continue
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(predictions[name], len(label["seq"]))
        tp, fp, fn, p, r, f = prf(truth, predicted)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        values.append((p, r, f))
        records[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
                         "truth_empty": not truth, "prediction_empty": not predicted,
                         "empty_empty": not truth and not predicted}
    n = len(values)
    tp, fp, fn = totals
    metrics = {"n_scored": n, "tp": tp, "fp": fp, "fn": fn,
               "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
               "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
               "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "macro_precision": math.fsum(value[0] for value in values) / n if n else None,
               "macro_recall": math.fsum(value[1] for value in values) / n if n else None,
               "macro_f1": math.fsum(value[2] for value in values) / n if n else None,
               "truth_empty_records": sum(row["truth_empty"] for row in records.values()),
               "empty_empty_records": sum(row["empty_empty"] for row in records.values())}
    extra = sorted(set(predictions) - panel_names)
    return metrics, records, missing, extra


def normalized_document(prediction_id, origin, full, predictions):
    rows = [{"name": ref["name"], "seq": ref["seq"], "sequence_sha256": ref["sequence_sha256"],
             "pred_pairs": sorted(pair_set(predictions[ref["name"]], len(ref["seq"])))}
            for ref in full if ref["name"] in predictions]
    document = {"schema_version": "testsetb.official_coverage_recovery.prediction.v1",
                "prediction_id": prediction_id, "dataset_id": "testsetb",
                "data": str(PANEL.relative_to(ROOT)), "data_sha256": PANEL_SHA,
                "n_sequences": len(rows), "prediction_origin": origin, "per_sequence": rows}
    scope = "full430" if len(rows) == 430 else f"partial{len(rows)}"
    filename = prediction_id.replace("/", "__") + f"__{scope}.json.gz"
    path = PREDICTIONS / filename
    write_gzip_json(path, document)
    return path


def render_report(document):
    lines = ["# TestSetB 官方 430 覆盖恢复", "",
             "仅恢复最终 frozen baseline 覆盖，不训练、不选模。保留原冻结 project 428 预测，只对官方多出的 `testsetb_104.bpseq`（S）与 `testsetb_36.bpseq`（N）做显式 CPU 补充。原始字符不替换；缺预测不补 0，部分覆盖不排名。大额补充与归一化预测位于 `/teams`。", "",
             "CPU 补充前对 6 条既有记录逐系统回放，9 个 decoder 均逐对精确匹配冻结预测；两条补充均做 3 次确定性重复。MXfold2 使用 `--gpu -1`，所有进程 `CUDA_VISIBLE_DEVICES=''`。", "",
             "RNAstructure Fold/MaxExpect 官方工具拒绝 `testsetb_104.bpseq` 的原始 S 字符，二者仅补到 429/430，保持 `PARTIAL_COVERAGE_NOT_RANKED`；该条缺预测不补 0。", "",
             "## 新增全覆盖身份", "",
             "| Full430 prediction identity | 来源 |",
             "|---|---|"]
    for prediction_id in document["new_full_coverage_identities"]:
        origin = "frozen 428 + official-only 2 CPU completion"
        lines.append(f"| `{prediction_id}` | {origin} |")
    for title, panel_id, full in (("官方 full 430", "testsetb_official_reference_430", True),
                                  ("项目 project 428", "testsetb_projected_reference_428", False)):
        lines += ["", f"## {title}", "",
                  "按 identity 字典序列出，不按指标排序。full 部分覆盖只展示 present-only 指标，状态保持 `PARTIAL_COVERAGE_NOT_RANKED`。", "",
                  "| Prediction identity | 状态 | n | micro F1 | macro F1 | 训练暴露 |",
                  "|---|---|---:|---:|---:|---|"]
        for row in document["rows"]:
            result = row["panels"][panel_id]
            if result["metrics"] is None:
                lines.append(f"| `{row['prediction_id']}` | {result['status']} | 0 | null | null | `{row['training_exposure']['full_project_panel_exposure_status']}` |")
            else:
                metric = result["metrics"]
                lines.append(f"| `{row['prediction_id']}` | {result['status']} | {metric['n_scored']} | {metric['micro_f1']:.10f} | {metric['macro_f1']:.10f} | `{row['training_exposure']['full_project_panel_exposure_status']}` |")
    lines += ["", "## 逐 ID 覆盖核查", "",
              "| Official-only ID | 非 ACGU | 原冻结覆盖 | 最终全覆盖身份数 |",
              "|---|---|---:|---:|"]
    for row in document["per_id_audit"]:
        lines.append(f"| `{row['name']}` | {''.join(row['non_acgu_characters'])} | {len(row['existing_frozen_coverage'])} | {len(row['final_full_coverage_identities'])} |")
    lines += ["", "未恢复的神经模型没有 full430 原始资产，保持 428/430 或原阻塞状态；不使用 GPU，不根据外部测试指标选模。训练暴露列逐行保留，不能用标签独立替代训练独立。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-missing2", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sha256(PANEL) != PANEL_SHA or sha256(PROJECT) != PROJECT_SHA or sha256(MX_PARAM) != MX_PARAM_SHA:
        raise ValueError("panel or MXfold2 parameter identity changed")
    full, projected = gzip_rows(PANEL), gzip_rows(PROJECT)
    if len(full) != 430 or len(projected) != 428 or any(not row["pairs"] for row in full):
        raise ValueError("panel population changed")
    full_by_name = {row["name"] for row in full}
    project_names = {row["name"] for row in projected}
    missing = [row for row in full if row["name"] not in project_names]
    if {row["name"] for row in missing} != {"testsetb_104.bpseq", "testsetb_36.bpseq"}:
        raise ValueError("official-only population changed")
    raw_maps = {}
    for family, (path, expected) in RAW.items():
        if sha256(path) != expected:
            raise ValueError(f"raw source changed {family}")
        document = load_json(path)
        data_sha = document["protocol"]["data_sha256"] if family == "mxfold2" else document["data_sha256"]
        n_sequences = document["protocol"]["n_sequences"] if family == "mxfold2" else document["n_sequences"]
        if data_sha != PROJECT_DATA_SHA or n_sequences != 428:
            raise ValueError(f"raw data contract changed {family}")
        raw_maps.update(raw_predictions(family, document))
    for system in SYSTEMS:
        if set(raw_maps[system]) != project_names:
            raise ValueError(f"raw coverage changed {system}")
    if args.generate_missing2:
        write_json(SUPPLEMENT, generate_supplement(full, projected, raw_maps))
    if not SUPPLEMENT.is_file():
        raise FileNotFoundError("missing2 supplement is blocked; explicit --generate-missing2 is required")
    supplement = load_json(SUPPLEMENT)
    if supplement["status"] != "PASS_2_ONLY_CPU_AND_BOUNDED_FROZEN_REPLAY_WITH_EXPLICIT_BLOCKS" or supplement["gpu_used"]:
        raise ValueError("invalid CPU supplement")
    missing_names = {row["name"] for row in missing}
    for system in SYSTEMS:
        rows = supplement["systems"][system]["per_sequence"]
        blocked = supplement["systems"][system]["blocked"]
        if {row["name"] for row in rows} | {row["name"] for row in blocked} != missing_names:
            raise ValueError(f"supplement coverage changed {system}")
        if {row["name"] for row in rows} & {row["name"] for row in blocked}:
            raise ValueError(f"supplement prediction/block overlap {system}")
        if len(set(supplement["systems"][system]["repeat_prediction_sha256"])) != 1:
            raise ValueError(f"supplement repeats differ {system}")
        for row in rows:
            ref = next(item for item in missing if item["name"] == row["name"])
            if row["seq"] != ref["seq"] or row["sequence_sha256"] != ref["sequence_sha256"]:
                raise ValueError(f"supplement sequence mismatch {system}/{row['name']}")
            pair_set(row["pred_pairs"], len(row["seq"]))
    prior = json.loads(PRIOR.read_text())
    prior_by_id = {row["prediction_id"]: row for row in prior["rows"]}
    rows = []
    normalized_entries = []
    for system in SYSTEMS:
        prediction_id = f"{system}/testsetb" if system in UPGRADED else f"{system}/testsetb/recovered_official2"
        merged = dict(raw_maps[system])
        merged.update({row["name"]: row["pred_pairs"] for row in supplement["systems"][system]["per_sequence"]})
        origin = "FROZEN_RAW428_PLUS_OFFICIAL2_CPU_COMPLETION"
        path = normalized_document(prediction_id, origin, full, merged)
        if system in UPGRADED:
            row = dict(prior_by_id[prediction_id])
        else:
            template = prior_by_id[f"{system}/testsetb/repeat0"]
            row = {"prediction_id": prediction_id, "baseline_id": template["baseline_id"],
                   "source_group": "FROZEN_WITH_OFFICIAL2_CPU_COMPLETION", "registry_status": "RECOVERED_RAW_STRICT_PLUS_COMPLETION",
                   "model_form": "deterministic_single_predictor", "training_exposure": template["training_exposure"]}
        row.update({"path": str(path), "sha256": sha256(path), "data_path": str(PANEL.relative_to(ROOT)),
                    "available": True, "prediction_origin": origin, "source_group": "FROZEN_WITH_OFFICIAL2_CPU_COMPLETION",
                    "registry_status": "RECOVERED_FROZEN_WITH_COMPLETION", "n_prediction_records": len(merged),
                    "sequence_join": "official panel ID plus sequence SHA256"})
        rows.append(row)
        normalized_entries.append({"prediction_id": prediction_id, "path": str(path), "sha256": sha256(path),
                                   "n_sequences": len(merged), "prediction_origin": origin})
    recovered_ids = {row["prediction_id"] for row in rows}
    for old in prior["rows"]:
        if old["prediction_id"] in recovered_ids:
            continue
        row = dict(old)
        if row["available"]:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"registered raw SHA changed {row['prediction_id']}")
            row["n_prediction_records"] = len(predictions_from_entry(row))
        rows.append(row)
    rows.sort(key=lambda row: row["prediction_id"])
    full_id, project_id = "testsetb_official_reference_430", "testsetb_projected_reference_428"
    all_records = []
    for row in rows:
        row["panels"] = {}
        if not row["available"]:
            status = "BLOCKED_RECORD_LEVEL_PREDICTIONS_REQUIRED" if row["registry_status"] == "SUMMARY_ONLY" else "BLOCKED_REGISTERED_PREDICTION_MISSING"
            for panel_id, panel in ((full_id, full), (project_id, projected)):
                row["panels"][panel_id] = {"status": status, "n_panel_sequences": len(panel), "n_with_prediction": 0,
                                           "coverage": 0.0, "metrics": None, "delta_vs_xens2": None,
                                           "missing_predictions_imputed_as_zero": False}
            continue
        predictions = predictions_from_entry(row)
        for panel_id, panel in ((full_id, full), (project_id, projected)):
            metrics, records, missing_names, extra = score(panel, predictions)
            row["panels"][panel_id] = {"status": "PASS_FULL_COVERAGE" if not missing_names else "PARTIAL_COVERAGE_NOT_RANKED",
                "n_panel_sequences": len(panel), "n_with_prediction": metrics["n_scored"],
                "coverage": metrics["n_scored"] / len(panel), "missing_count": len(missing_names),
                "missing_names": missing_names, "extra_count": len(extra), "extra_names": extra,
                "metrics": metrics, "delta_vs_xens2": None, "missing_predictions_imputed_as_zero": False}
            for name, values in records.items():
                all_records.append({"dataset": "testsetb", "panel_id": panel_id,
                                    "prediction_id": row["prediction_id"], "name": name, **values})
    xens = next(row for row in rows if row["prediction_id"] == "xens2/testsetb")["panels"][project_id]["metrics"]
    for row in rows:
        result = row["panels"][project_id]
        if result["status"] == "PASS_FULL_COVERAGE":
            result["delta_vs_xens2"] = {key: result["metrics"][key] - xens[key] for key in (
                "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
    full_actual = {row["prediction_id"] for row in rows if row["panels"][full_id]["status"] == "PASS_FULL_COVERAGE"}
    if full_actual != EXPECTED_FULL:
        raise ValueError(f"full coverage identity mismatch: {full_actual}")
    for old_id, old in prior_by_id.items():
        if old["available"]:
            new = next(row for row in rows if row["prediction_id"] == old_id)
            if new["panels"][project_id]["metrics"] != old["panels"][project_id]["metrics"]:
                raise ValueError(f"project metric changed {old_id}")
    manifest = {
        "schema_version": "testsetb.official_coverage_recovery.manifest.v1",
        "status": "PASS_7_FULL430_IDENTITIES_AND_2_PARTIAL429_WITH_EXPLICIT_BLOCKS", "dataset_id": "testsetb", "base_commit": "9aeec63d73f90360bd69f7a65fdbb19d5d651908",
        "official_panel": {"path": str(PANEL.relative_to(ROOT)), "sha256": PANEL_SHA, "n_sequences": 430},
        "project_panel": {"path": str(PROJECT.relative_to(ROOT)), "sha256": PROJECT_SHA, "n_sequences": 428},
        "official_only_n": 2, "raw_sources": {family: {"path": str(path), "sha256": expected} for family, (path, expected) in RAW.items()},
        "supplement": {"path": str(SUPPLEMENT), "sha256": sha256(SUPPLEMENT), "new_inference_records_per_system": 2,
                       "projected_records_rerun": 0, "bounded_replay_indices": list(REPLAY_INDICES),
                       "bounded_replay_sequences_per_system": len(REPLAY_INDICES), "deterministic_repeats": 3,
                       "device": "CPU_ONLY", "gpu_used": False, "cuda_visible_devices": "", "mxfold2_gpu_argument": -1},
        "prediction_entries": normalized_entries, "new_full_coverage_identities": sorted(EXPECTED_FULL),
        "upgraded_existing_identities": [f"{system}/testsetb" for system in ("contrafold", "eternafold", "mxfold2")],
        "partially_completed_existing_identities": ["rnastructure_fold/testsetb", "rnastructure_maxexpect/testsetb"],
        "added_identities": [f"{system}/testsetb/recovered_official2" for system in ADDED],
        "unrecovered_neural_models": "NO_FULL430_RAW_ASSET; PARTIAL_OR_BLOCKED; NO_GPU_INFERENCE",
        "external_test_model_selection": False, "new_training": False,
    }
    prior_manifest = json.loads(PRIOR_MANIFEST.read_text())
    existing = {}
    for entry in prior_manifest["datasets"]["testsetb"]["predictions"]:
        if entry["available"]:
            existing[entry["prediction_id"]] = set(predictions_from_entry(entry))
    per_id = []
    for ref in missing:
        per_id.append({"name": ref["name"], "sequence_sha256": ref["sequence_sha256"], "pair_set_sha256": ref["pair_set_sha256"],
            "non_acgu_characters": sorted(set(ref["seq"]) - set("ACGU")), "truth_empty": not ref["pairs"],
            "existing_frozen_coverage": sorted(pid for pid, names in existing.items() if ref["name"] in names),
            "vienna_raw_strict_coverage": sorted(system for system in ADDED if ref["name"] in raw_maps[system]),
            "final_full_coverage_identities": sorted(EXPECTED_FULL)})
    if any(row["existing_frozen_coverage"] or row["vienna_raw_strict_coverage"] for row in per_id):
        raise ValueError("official-only ID unexpectedly present in original raw predictions")
    manifest["per_id_audit"] = per_id
    document = {"schema_version": "testsetb.official_coverage_recovery.metrics.v1", "dataset": "testsetb",
        "audit_date": "2026-11-27", "scope": "official-full and projected frozen baseline coverage recovery; no training or model selection",
        "label_panels": {full_id: manifest["official_panel"], project_id: manifest["project_panel"]},
        "metric_contract": prior["metric_contract"], "coverage_accounting": {
            "inventory_entries": len(rows), "scored_identities": sum(row["available"] for row in rows),
            "blocked_identities": sum(not row["available"] for row in rows),
            "projected_full_coverage": sum(row["panels"][project_id]["status"] == "PASS_FULL_COVERAGE" for row in rows),
            "official_full_coverage": len(full_actual),
            "official_partial_not_ranked": sum(row["panels"][full_id]["status"] == "PARTIAL_COVERAGE_NOT_RANKED" for row in rows),
            "official_records_missing_from_project": 2},
        "recovery_manifest": str(MANIFEST), "recovery_manifest_sha256": None,
        "new_full_coverage_identities": manifest["new_full_coverage_identities"],
        "per_id_audit": manifest["per_id_audit"],
        "rows": rows, "external_test_model_selection": False, "new_training": False,
        "new_inference_scope": "official-only two records for nine CPU decoders; bounded six-record protocol replay not used for metrics",
        "gpu_used": False}
    write_json(MANIFEST, manifest)
    document["recovery_manifest_sha256"] = sha256(MANIFEST)
    write_json(OUT / "system_metrics.json", document)
    write_gzip_jsonl(ART / "per_record_metrics.jsonl.gz", all_records)
    write_json(ART / "recovery_audit.json", manifest)
    source_manifest = {"schema_version": "testsetb.official_coverage_recovery.source_manifest.v1",
        "base_commit": manifest["base_commit"], "panels": manifest["official_panel"] | {"project": manifest["project_panel"]},
        "raw_sources": manifest["raw_sources"], "supplement": manifest["supplement"],
        "prediction_entries": normalized_entries, "artifact_root": str(ART),
        "per_record_metrics": {"path": str(ART / "per_record_metrics.jsonl.gz"), "sha256": sha256(ART / "per_record_metrics.jsonl.gz"), "n_rows": len(all_records)},
        "shared_registries_modified": False, "external_test_model_selection": False, "new_training": False, "gpu_used": False}
    write_json(SOURCE / "source_manifest.json", source_manifest)
    write_json(OUT / "recovery_audit.json", manifest)
    write_json(OUT / "build_receipt.json", {"schema_version": "testsetb.official_coverage_recovery.receipt.v1", "status": "PASS",
        "full430_identities": len(full_actual), "scored_identities": document["coverage_accounting"]["scored_identities"],
        "per_record_metric_rows": len(all_records), "system_metrics_sha256": sha256(OUT / "system_metrics.json"),
        "source_manifest_sha256": sha256(SOURCE / "source_manifest.json"), "artifact_root": str(ART)})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(document), encoding="utf-8")
    write_json(ART / "build_receipt.json", json.loads((OUT / "build_receipt.json").read_text()))
    print(json.dumps({"status": manifest["status"], "full430_identities": len(full_actual),
                      "scored_identities": document["coverage_accounting"]["scored_identities"],
                      "per_record_metric_rows": len(all_records), "generated_missing2": args.generate_missing2,
                      "gpu_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
