#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("/home/rna_jev/third_party/structRFM")
DEFAULT_DATASET = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
MATRIX_SCORES = (
    "score_basepair",
    "score_helix_stacking",
    "score_mismatch_external",
    "score_mismatch_hairpin",
    "score_mismatch_internal",
    "score_mismatch_multi",
    "score_base_hairpin",
    "score_base_internal",
    "score_base_multi",
    "score_base_external",
)
LENGTH_SHAPES = {
    "score_hairpin_length": (31,),
    "score_bulge_length": (31,),
    "score_internal_length": (31,),
    "score_internal_explicit": (5, 5),
    "score_internal_symmetry": (16,),
    "score_internal_asymmetry": (29,),
    "score_helix_length": (31,),
}


def load_interface(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("interface", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load interface: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interface"] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("Loaded interface identity mismatch")
    return module


def parameters(sequence, seed, source, parameter_file=None):
    import numpy as np
    if parameter_file is not None:
        archive = np.load(parameter_file)
        result = {branch: {name.split("__", 1)[1]: archive[name].copy() for name in archive.files if name.startswith(f"{branch}__")} for branch in ("turner", "positional")}
        for branch in result.values():
            for name, value in list(branch.items()):
                if name.startswith("score_"):
                    branch["count_" + name[6:]] = np.zeros_like(value)
        return result
    sys.path.insert(0, str(Path(source) / "tasks/seqcls_ssp"))
    import param_turner2004
    rng = np.random.RandomState(seed)
    turner = {name: np.asarray(getattr(param_turner2004, name), dtype=np.float32) for name in dir(param_turner2004) if name.startswith("score_")}
    for name, value in list(turner.items()):
        turner["count_" + name[6:]] = np.zeros_like(value)
    length = len(sequence)
    positional = {name: rng.normal(0.0, 0.2, (length, length)).astype(np.float32) for name in MATRIX_SCORES}
    positional.update({name: rng.normal(0.0, 0.2, shape).astype(np.float32) for name, shape in LENGTH_SHAPES.items()})
    for name, value in list(positional.items()):
        positional["count_" + name[6:]] = np.zeros_like(value)
    return {"turner": turner, "positional": positional}


def cases(source):
    rng = __import__("numpy").random.RandomState(20261126)
    alphabet = __import__("numpy").array(list("ACGU"))
    sequences = ["".join(rng.choice(alphabet, size=length)) for length in (33, 34, 37, 41, 47, 55, 63, 72, 89, 101, 127, 160)]
    output = []
    for index, sequence in enumerate(sequences):
        constraint = None
        reference = None
        output.append({"case_id": f"case_{index:02d}", "sequence": sequence, "seed": 1000 + index, "constraint": constraint, "reference": reference})
    dataset_rows = [json.loads(line) for line in DEFAULT_DATASET.read_text().splitlines() if line.strip()]
    selected = [min(dataset_rows, key=lambda row: len(row["seq"])), max(dataset_rows, key=lambda row: len(row["seq"]))]
    selected.extend(dataset_rows[index] for index in (0, 100, 1000, 3000))
    for offset, row in enumerate(selected, start=len(output)):
        output.append({"case_id": f"case_{offset:02d}", "sequence": row["seq"].upper().replace("T", "U"), "seed": 1000 + offset, "constraint": None, "reference": None, "dataset_name": row["name"]})
    return output


def run(interface_path, output_path, source, case_id=None, cases_path=None):
    interface = load_interface(interface_path)
    rows = []
    parameter_manifest = json.loads(Path(cases_path).read_text()) if cases_path else None
    source_cases = parameter_manifest["cases"] if parameter_manifest else cases(source)
    selected_cases = [case for case in source_cases if case_id is None or case["case_id"] == case_id]
    if not selected_cases:
        raise ValueError(f"Unknown case: {case_id}")
    for case in selected_cases:
        print(f"running {case['case_id']} length={len(case['sequence'])}", flush=True)
        value, dot_bracket, pairs = interface.predict_mxfold(
            case["sequence"],
            parameters(case["sequence"], case["seed"], source, case.get("parameter_file")),
            min_hairpin_length=3,
            max_internal_length=30,
            max_helix_length=30,
            constraint=case["constraint"],
            reference=case["reference"],
            loss_pos_paired=0.0,
            loss_neg_paired=0.0,
            loss_pos_unpaired=0.0,
            loss_neg_unpaired=0.0,
        )
        rows.append({**case, "energy": float(value), "dot_bracket": dot_bracket, "pairs": [int(value) for value in pairs]})
    document = {"python": platform.python_version(), "interface": str(Path(interface_path).resolve()), "parameter_manifest": str(Path(cases_path).resolve()) if cases_path else None, "parameter_manifest_sha256": hashlib.sha256(Path(cases_path).read_bytes()).hexdigest() if cases_path else None, "n_cases": len(rows), "rows": rows}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"status": "PASS_CASE_GENERATION", "python": document["python"], "n_cases": len(rows), "output": str(output_path)}, indent=2))


def compare(reference_path, candidate_path, report_path):
    reference = json.loads(Path(reference_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    if reference["n_cases"] != candidate["n_cases"] or len(reference["rows"]) != 18:
        raise AssertionError("Unexpected differential case count")
    if reference.get("parameter_manifest_sha256") != candidate.get("parameter_manifest_sha256"):
        raise AssertionError("DP parameter manifest identity mismatch")
    checks = []
    for expected, actual in zip(reference["rows"], candidate["rows"]):
        for key in ("case_id", "sequence", "seed", "constraint", "reference"):
            assert expected[key] == actual[key], (key, expected["case_id"])
        assert expected["pairs"] == actual["pairs"], expected["case_id"]
        assert expected["dot_bracket"] == actual["dot_bracket"], expected["case_id"]
        energy_delta = abs(expected["energy"] - actual["energy"])
        assert math.isclose(expected["energy"], actual["energy"], rel_tol=0.0, abs_tol=1e-5), (expected["case_id"], energy_delta)
        checks.append({"case_id": expected["case_id"], "length": len(expected["sequence"]), "pairs_equal": True, "dot_bracket_equal": True, "energy_delta": energy_delta})
    report = {
        "schema_version": 1,
        "status": "PASS_OFFICIAL_CP38_VS_REBUILT_CP310",
        "reference_python": reference["python"],
        "candidate_python": candidate["python"],
        "reference_interface": reference["interface"],
        "reference_interface_sha256": hashlib.sha256(Path(reference["interface"]).read_bytes()).hexdigest(),
        "candidate_interface": candidate["interface"],
        "candidate_interface_sha256": hashlib.sha256(Path(candidate["interface"]).read_bytes()).hexdigest(),
        "parameter_manifest_sha256": reference.get("parameter_manifest_sha256"),
        "real_bprna_new_cases": sum("dataset_name" in row for row in reference["rows"]),
        "n_cases": len(checks),
        "max_energy_delta": max(row["energy_delta"] for row in checks),
        "pair_and_dot_bracket_exact": True,
        "energy_absolute_tolerance": 1e-5,
        "inference_scope": "constraint=None, reference=None, all loss weights=0; exact structRFM inference path",
        "checks": checks,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--interface", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    run_parser.add_argument("--case-id")
    run_parser.add_argument("--cases", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.interface, args.output, args.source, args.case_id, args.cases)
    else:
        compare(args.reference, args.candidate, args.report)


if __name__ == "__main__":
    main()
