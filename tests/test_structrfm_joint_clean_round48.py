import json
import subprocess
import sys
from pathlib import Path


NAMES = ["bprna_ts0", "bprna_new", "archiveii", "ref_pdb_ts1", "ref_pdb_ts2",
         "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "rfam_fam", "chanrg_test",
         "chanrg_gena", "chanrg_genc", "chanrg_genf"]


def test_joint_clean_matches_independent_mmseqs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    mmseqs = root / "tools/bin/mmseqs"
    if not mmseqs.is_file():
        import pytest
        pytest.skip("MMseqs binary is required")
    fasta = tmp_path / "pretraining.fasta"
    fasta.write_text(">exact\nACGUACGU\n>homolog\nAAAAAAAA\n>other\nCCCCCCCC\n")
    arguments = []
    for index, name in enumerate(NAMES):
        rows = [{"name": f"{name}_exact", "seq": "ACGUACGU", "pairs": []},
                {"name": f"{name}_homolog", "seq": "AAAAAAAC", "pairs": []},
                {"name": f"{name}_clean", "seq": "AAAACCCC", "pairs": []}]
        path = tmp_path / f"{name}.raw.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        arguments.extend(["--split", f"{name}={path}"])
        separate = tmp_path / f"separate_{name}"
        subprocess.run([sys.executable, str(root / "tools/audit_structrfm_clean_round48.py"),
                        "--train-fasta", str(fasta), "--split", str(path),
                        "--out", str(separate / "clean.jsonl"), "--audit", str(separate / "audit.json"),
                        "--work-dir", str(separate / "work"), "--mmseqs", str(mmseqs)],
                       cwd=root, check=True, capture_output=True, text=True)
    joint = tmp_path / "joint"
    subprocess.run([sys.executable, str(root / "tools/audit_structrfm_clean_all_round48.py"),
                    "--train-fasta", str(fasta), *arguments, "--out-dir", str(joint),
                    "--work-dir", str(tmp_path / "joint_work"), "--mmseqs", str(mmseqs)],
                   cwd=root, check=True, capture_output=True, text=True)
    summary = json.loads((joint / "summary.json").read_text())
    assert summary["status"] == "PASS_STRUCTRFM_PRETRAIN_CLEAN_ABC_13_SPLITS"
    assert set(summary["splits"]) == set(NAMES)
    for index, name in enumerate(NAMES):
        expected = json.loads((tmp_path / f"separate_{name}/audit.json").read_text())
        actual = json.loads((joint / f"{name}.audit.json").read_text())
        assert actual["query_start"] == index * 3
        for key in ("raw_n", "clean_n", "removed_exact", "removed_homology_only",
                    "removed_exact_indices", "removed_homology_only_indices",
                    "train_sha256", "split_sha256", "clean_sha256"):
            assert actual[key] == expected[key], (name, key)
        assert (joint / f"{name}.jsonl").read_bytes() == (
            tmp_path / f"separate_{name}/clean.jsonl").read_bytes()
