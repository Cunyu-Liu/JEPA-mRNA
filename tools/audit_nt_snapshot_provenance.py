import json
from pathlib import Path

from audit_pretrain_exact_overlap import sha256

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


def main():
    tree = json.loads((ROOT / "rinalmo_github_tree.json").read_text())
    commit = json.loads((ROOT / "rinalmo_github_main_commit.json").read_text())
    remote = json.loads((ROOT / "rinalmo_remote_data_decoded.txt").read_text())
    history = json.loads((ROOT / "rinalmo_remote_data_history.json").read_text())
    initial = json.loads((ROOT / "source_manifest.json").read_text())
    if tree["truncated"] or tree["sha"] != commit["sha"]:
        raise AssertionError("incomplete or inconsistent official repository tree")
    if remote["FTP"]["PATH"]["NT_SEQS"] != "/blast/db/FASTA/nt.gz":
        raise AssertionError("unexpected NCBI nt path")
    if initial["nt"]["status"] != "NOT_DOWNLOADED_HISTORICAL_SNAPSHOT_UNPINNED":
        raise AssertionError("unexpected nt download state")
    paths = [item["path"] for item in tree["tree"]]
    corpus_scripts = [path for path in paths if any(key in path.lower() for key in ("pretrain_corpus", "prepare_pretrain", "prepare_corpus"))]
    report = {"status": "BLOCKED_HISTORICAL_NT_SNAPSHOT_UNPINNED",
              "scientific_scope": "provenance assessment only; no complete historical pretraining contamination or clean-OOD conclusion",
              "official_repository": "https://github.com/lbcb-sci/RiNALMo",
              "main_commit": commit["sha"], "tree_sha": tree["sha"], "tree_truncated": tree["truncated"],
              "corpus_manifest_or_preparation_scripts_found": corpus_scripts,
              "remote_data_path": "rinalmo/resources/remote_data.json",
              "remote_data_nt": f"https://{remote['FTP']['ADDRESS']['NCBI']}{remote['FTP']['PATH']['NT_SEQS']}",
              "live_aliases": {"nt": remote["FTP"]["PATH"]["NT_SEQS"],
                               "RNAcentral": remote["FTP"]["PATH"]["RNACENTRAL_ACTIVE_SEQS"],
                               "Rfam": remote["FTP"]["PATH"]["RFAM_SEQS"]},
              "current_nt": {"bytes": initial["nt"]["current_bytes"], "downloaded": False,
                             "not_a_historical_substitute": True},
              "paper_preprocessing": initial["paper"]["reported_preprocessing"],
              "official_commit_history": [{"sha": item["sha"], "date": item["commit"]["committer"]["date"],
                                           "message": item["commit"]["message"]} for item in history],
              "limitations": ["nt URL has no historical release, date or content hash",
                              "commit timestamps do not prove which nt snapshot was trained on",
                              "current nt.gz would be a new sensitivity analysis, not the original pretraining corpus",
                              "nt absence prevents a complete exposure denominator but cannot negate positive overlap in known sources"],
              "evidence_sha256": {name: sha256(ROOT / name) for name in (
                  "source_manifest.json", "rinalmo_github_tree.json", "rinalmo_github_main_commit.json",
                  "rinalmo_remote_data_decoded.txt", "rinalmo_remote_data_history.json",
                  "rinalmo_readme_api.json", "rinalmo_ss_dataset_decoded.py")},
              "next_step": "retain the nt limitation explicitly; do not download current nt to claim historical pretraining coverage"}
    output = ROOT / "nt_snapshot_assessment.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
