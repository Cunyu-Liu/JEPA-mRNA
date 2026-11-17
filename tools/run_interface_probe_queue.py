import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
MODES = ("rinalmo", "h0", "h2", "handcrafted", "random", "length", "gc")


def main():
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    while True:
        paths = [ROOT / "features" / f"{mode}_{split}_part0.json"
                 for mode in ("rinalmo", "h0", "h2") for split in ("train", "holdout")]
        if all(path.exists() for path in paths):
            break
        time.sleep(30)
    qa = json.loads((ROOT / "label_qa_summary.json").read_text())
    assert qa["status"] == "PASS_INDEPENDENT_INTERFACE_LABEL_QA"
    assert qa["splits"]["train"]["n_valid"] == 804 and qa["splits"]["holdout"]["n_valid"] == 202
    qa_sha = hashlib.sha256((ROOT / "label_qa_summary.json").read_bytes()).hexdigest()
    for path in paths:
        result = json.loads(path.read_text())
        assert result["status"] == "PASS_FROZEN_INTERFACE_FEATURES"
        assert result["label_qa_sha256"] == qa_sha
    jobs = [(mode, seed) for mode in MODES for seed in (42, 2026, 3407)]
    running = []
    failed = []
    while jobs or running:
        while jobs and len(running) < 2 and not failed:
            mode, seed = jobs.pop(0)
            directory = ROOT / "converged_baselines" / f"{mode}_seed{seed}"
            result_path = directory / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["converged"] and result["label_qa_sha256"] == qa_sha:
                    continue
                raise RuntimeError(f"existing result identity requires review: {result_path}")
            if (directory / "best.pt").exists() and not (directory / "resume.pt").exists():
                raise RuntimeError(f"best checkpoint exists without resumable optimizer: {directory}")
            command = [PYTHON, "tools/train_interface_baseline.py", "--mode", mode, "--seed", str(seed)]
            if (directory / "resume.pt").exists():
                command.append("--resume")
            log = (ROOT / "converged_baselines" / f"{mode}_seed{seed}.stdout.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a")
            process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
            running.append((process, handle, mode, seed))
        time.sleep(15)
        for item in list(running):
            process, handle, mode, seed = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            running.remove(item)
            if code != 0:
                failed.append((mode, seed, code))
                continue
            result = json.loads((ROOT / "converged_baselines" / f"{mode}_seed{seed}" / "result.json").read_text())
            assert result["converged"] and result["stop_reason"] == "internal_validation_plateau_30_epochs"
            print(json.dumps({"completed": mode, "seed": seed, "best_epoch": result["best_epoch"],
                              "micro_auprc": result["micro_auprc"], "macro_f1_binary": result["macro_f1_binary"]}), flush=True)
    if failed:
        raise RuntimeError(f"interface probe failures preserved: {failed}")
    subprocess.run([PYTHON, "tools/aggregate_interface_results.py"], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
