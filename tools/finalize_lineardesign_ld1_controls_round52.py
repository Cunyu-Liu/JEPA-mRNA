import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    path = root / "audits/round52/lineardesign_ld1_controls_qa.json"
    report = json.loads(path.read_text())
    assert report["status"] == "PASS_LD1_798_CONTROL_INDEPENDENT_VIENNA_QA_AND_AGGREGATION"
    assert report["n_recomputed"] == 798 and report["n_proteins"] == 114
    assert not report["scientific_conclusion"]
    result = {
        "status": "PASS_LD1_CONTROLS_THREE_CHECK_ACCEPTANCE",
        "scientific_conclusion": False,
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        "paired_comparisons": report["paired_comparisons"],
        "limitations": ["Full computational QA is complete; wild-type assets and LD2 frozen-model evaluation remain separate.",
                        "No biological efficacy or PRISM benefit is claimed.",
                        "Shell acceptance finalization failed; this independent Python finalizer did not rerun ViennaRNA."],
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
