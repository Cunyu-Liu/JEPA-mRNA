import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nb_secstr_queue_round35 import completed_runs_needing_qa, full_result_pass, sha256


def write_run(root, head, seed, status="PASS", converged=True):
    run_dir = root / f"{head}_s{seed}"
    run_dir.mkdir(parents=True)
    run = run_dir / "run.json"
    run.write_text(json.dumps({
        "status": status, "converged": converged,
        "early_stopped": converged, "scientific_eligibility": converged,
    }) + "\n")
    return run_dir, run


def test_completed_runs_need_missing_or_stale_qa(tmp_path):
    official, run = write_run(tmp_path, "official", 0)
    assert completed_runs_needing_qa(tmp_path) == [("official", 0)]

    (official / "qa.json").write_text(json.dumps({
        "status": "PASS", "run_sha256": sha256(run),
    }) + "\n")
    assert completed_runs_needing_qa(tmp_path) == []

    (official / "qa.json").write_text(json.dumps({
        "status": "PASS", "run_sha256": "stale",
    }) + "\n")
    assert completed_runs_needing_qa(tmp_path) == [("official", 0)]


def test_only_passed_completed_runs_are_backfilled(tmp_path):
    write_run(tmp_path, "official", 0, status="FAIL")
    _, run = write_run(tmp_path, "jev", 3)
    assert completed_runs_needing_qa(tmp_path) == [("jev", 3)]

    qa = tmp_path / "jev_s3" / "qa.json"
    qa.write_text(json.dumps({"status": "FAIL", "run_sha256": sha256(run)}) + "\n")
    assert completed_runs_needing_qa(tmp_path) == [("jev", 3)]


def test_nonconverged_pass_is_not_complete_or_backfilled(tmp_path):
    _, run = write_run(tmp_path, "official", 2, converged=False)
    assert not full_result_pass(run)
    assert completed_runs_needing_qa(tmp_path) == []
