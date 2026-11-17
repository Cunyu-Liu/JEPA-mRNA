import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "src/rnajepa/dual_view.py": "908b3a52f96fef5d0737e4b362077c2433ecf3db441ba3d4c86b4aa3505a6266",
    "tools/train_plan_a.py": "67cc71d012111220a61e6be76b653cc9fea313cc4dbed26499628e6fad8392a4",
    "tests/test_dual_view.py": "66331fd61a20401daba3eeb31bed82d6664974ffc4fbcf2b413ec848fc90b208",
    "tools/validate_dual_view_cuda.py": "975cd9ede0406130502ee5655a2208af16fe2d36148ed4dc0a1cb5fd81de8e0e",
    "tools/validate_dual_view_deployment.py": "d1761f2828fb1666714b5b883d96679a9d196b9bf42912b844ca4f7cfb3a0266",
}


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root)
    repo = Path(args.repo)
    manifest = root / "manifests/p1/candidate_cu128_v1"
    host = root / "artifacts/p1/host_integration_candidate_v1"
    for relative, expected in EXPECTED.items():
        actual = sha256(repo / relative)
        if actual != expected:
            raise AssertionError(f"source identity mismatch for {relative}: {actual}")
    pytest_log = (manifest / "pytest.log").read_text()
    if "28 passed" not in pytest_log or " failed" in pytest_log:
        raise AssertionError("28-test CPU gate did not pass")
    cuda = load(manifest / "cuda_forward_backward.json")
    if cuda["status"] != "PASS_DUAL_VIEW_CUDA_FORWARD_BACKWARD":
        raise AssertionError("CUDA gate did not pass")
    if set(cuda["arms"]) != {"H1", "H2", "H3", "H4", "H5", "H6"}:
        raise AssertionError("not all H1-H6 CUDA arms passed")
    if cuda["source_sha256"] != {key: EXPECTED[key] for key in (
            "src/rnajepa/dual_view.py", "tools/train_plan_a.py",
            "tests/test_dual_view.py", "tools/validate_dual_view_cuda.py")}:
        raise AssertionError("CUDA report is not bound to the candidate source")
    if cuda["torch_geometric_imported"] or cuda["flops"]["profiler_reported_forward_backward_flops"] <= 0:
        raise AssertionError("CUDA dependency or FLOPs check failed")
    if max(arm["peak_memory_bytes"] for arm in cuda["arms"].values()) >= 24 * 10**9:
        raise AssertionError("synthetic CUDA peak exceeds 24 GB")
    runs = {}
    for name, arm, steps in (("h0", "H0", 1), ("h2_seed0", "H2", 1), ("h2_seed1_resume", "H2", 2)):
        run_root = (root / "artifacts/p1/host_integration_resume_fixture_v2" / name
                    if name == "h2_seed1_resume" else host / name)
        meta = load(run_root / "run_meta.json")
        if meta["status"] != "completed" or meta["steps_completed"] != steps:
            raise AssertionError(f"unexpected host terminal state for {name}")
        if meta["converged"] or meta["early_stopped"] or meta["stop_reason"] != "step_cap":
            raise AssertionError("engineering step caps must not be called convergence")
        if meta["scorer"] != "resnet2d" or meta["motif"] is not None or meta["thermodynamics"] is not None:
            raise AssertionError("host or independent input arm changed")
        if meta["objective"]["lambda_jepa"] != 0 or meta["dual_view"]["arm"] != arm:
            raise AssertionError("stopped Pair-JEPA or wrong dual-view arm")
        records = [json.loads(line) for line in (run_root / "train_log.jsonl").read_text().splitlines()]
        if not records or max(record["peak_gb"] for record in records) >= 24:
            raise AssertionError(f"invalid host memory ledger for {name}")
        runs[name] = {"steps": steps, "converged": False, "stop_reason": "step_cap"}
    legacy = load(root / "artifacts/p1/host_integration_resume_fixture_v2/h2_seed1_resume/legacy_resume_preparation.json")
    if legacy["sigreg_global_step_preserved_in_state_dict"] < 1:
        raise AssertionError("legacy SIGReg fixture was not constructed correctly")
    deployment = load(manifest / "deployment_audit.json")
    if deployment["status"] != "PASS_SEQUENCE_ONLY_DEPLOYMENT":
        raise AssertionError("sequence-only deployment failed")
    if (deployment["dual_view_state_present"] or any(deployment["training_only_call_counts"].values())
            or not deployment["deterministic_repeat_equal"] or not deployment["legal_scores_finite"]):
        raise AssertionError("deployment contains structure training state or calls")
    result = {
        "status": "PASS_P1_ENGINEERING_ONLY",
        "p1_accepted": True,
        "p2_started": False,
        "scientific_claim": "none; no pilot, accuracy, convergence, or downstream claim",
        "source_sha256": EXPECTED,
        "cpu_tests": "28 passed",
        "cuda": "H1-H6 forward/backward passed; H0 checked in real host",
        "host_runs": runs,
        "legacy_resume": "explicit scalar removed; state-dict SIGReg step restored",
        "deployment": "sequence-only; training-only call counts all zero",
        "next_required_action": "Freeze pilot initialization, limited budget and model-dev protocol before H0-H6.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
