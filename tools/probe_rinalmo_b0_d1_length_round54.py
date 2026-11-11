import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, required=True, choices=(1022, 2048, 3000))
    args = parser.parse_args()
    source = ROOT / f"data/rna_complex_round49/rinalmo_d1_length_round54/{args.threshold}"
    rows = [json.loads(line) for line in (source / "train_d1.jsonl").read_text().splitlines() if line.strip()]
    selected = max(rows[42564:], key=lambda row: len(row["seq"]))
    out = ROOT / f"audits/round54/d1_length_{args.threshold}_longest"
    run = out / "run"
    if (run / "run_meta.json").exists():
        raise FileExistsError(f"refusing to overwrite {run}")
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    data = assets / "input.jsonl"
    data.write_text(json.dumps(selected) + "\n")
    command = [
        sys.executable, "/home/rna_jev/tools/train_plan_a.py",
        "--data", str(data),
        "--teacher-dir", "/root/rna_jepa_data/ss_data/teacher/bprna_tr1c",
        "--aux-teacher-dir", str(ROOT / "data/rna_complex_round49/rinalmo_full_d1/teacher_complex"),
        "--lazy-teacher", "--out", str(run), "--steps", "1", "--batch-size", "1",
        "--seed", "42", "--warmup-head-steps", "0", "--unfreeze-every", "1",
        "--unfreeze-per-step", "33", "--head-lr", "1e-4", "--backbone-lr", "1e-5",
        "--nll-normalization", "sum", "--scorer", "resnet2d", "--pair-latent-dim", "64",
        "--save-every", "1", "--snapshot-every", "2000", "--snapshot-dir", str(out / "snapshots"),
        "--gpu-reserve-gb", "0",
    ]
    log = out / "train.log"
    with log.open("w") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    result = {
        "status": "PASS_D1_LENGTH_FULL_B0_ENGINEERING" if process.returncode == 0 else "PARTIAL_D1_LENGTH_FULL_B0_FAILURE",
        "scientific_conclusion": False,
        "threshold": args.threshold,
        "length": len(selected["seq"]),
        "returncode": process.returncode,
        "command": command,
        "input_sha256": sha256(data),
        "teacher_manifest_sha256": sha256(ROOT / "data/rna_complex_round49/rinalmo_full_d1/teacher_complex/manifest.json"),
        "train_log_sha256": sha256(log),
    }
    if (run / "run_meta.json").exists():
        meta = json.loads((run / "run_meta.json").read_text())
        result["run_meta"] = meta
        result["run_meta_sha256"] = sha256(run / "run_meta.json")
        result["checkpoint_sha256"] = sha256(run / "resume.pt") if (run / "resume.pt").exists() else None
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
