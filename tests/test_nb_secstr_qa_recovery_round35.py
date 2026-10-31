import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path("/home/rna_jev/tools/run_nb_secstr_qa_recovery_round35.sh")


def run_recovery(tmp_path, result=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if result is not None:
        (run_dir / "run.json").write_text(json.dumps(result) + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    flock_log = tmp_path / "flock_args.json"
    fake_flock = bin_dir / "flock"
    fake_flock.write_text(
        "#!/root/miniconda3/envs/lucaone/bin/python\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_FLOCK_LOG'], 'w') as handle:\n"
        "    json.dump(sys.argv[1:], handle)\n"
    )
    fake_flock.chmod(0o755)
    finished = subprocess.Popen(["sleep", "0.01"])
    finished.wait()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_FLOCK_LOG": str(flock_log)}
    completed = subprocess.run(
        ["bash", str(SCRIPT), str(finished.pid), str(run_dir)],
        env=env, text=True, capture_output=True, check=False,
    )
    args = json.loads(flock_log.read_text()) if flock_log.exists() else None
    return completed, args


def test_missing_run_fails_without_starting_queue(tmp_path):
    completed, args = run_recovery(tmp_path)
    assert completed.returncode != 0
    assert "refusing automatic retry" in completed.stderr
    assert args is None


def test_nonconverged_run_fails_without_starting_queue(tmp_path):
    completed, args = run_recovery(tmp_path, {
        "status": "PASS", "converged": False,
        "early_stopped": False, "scientific_eligibility": False,
    })
    assert completed.returncode != 0
    assert "training completion gate failed" in completed.stderr
    assert args is None


def test_converged_run_starts_single_serial_queue(tmp_path):
    completed, args = run_recovery(tmp_path, {
        "status": "PASS", "converged": True,
        "early_stopped": True, "scientific_eligibility": True,
    })
    assert completed.returncode == 0, completed.stderr
    assert "COMPLETION_GATE_PASS" in completed.stdout
    assert args is not None
    assert any(arg.endswith("run_nb_secstr_queue_round35.py") for arg in args)
    assert args[-6:] == ["--parallel", "1", "--min-free-mib", "8000", "--interval", "30"]
