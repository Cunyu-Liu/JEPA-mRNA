import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TITLE = "Contamination-aware evaluation of RNA sequence representations: pretraining exposure, homology, and limits of structure-assisted learning"
HEADER = """% Evidence draft only; not submission-ready.
% This source requires XeLaTeX with xeCJK support. No LaTeX compiler was available locally;
% no compiled PDF from this source has been validated. The accepted reading PDF is
% artifacts/v2/reporting/rna_evidence_v2_manuscript_draft.pdf (SHA256=473eaa07…52d0b).
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    manuscript = Path(args.manuscript)
    source_hash = sha256(manuscript)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise AssertionError("manuscript changed after evidence QA")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is required")
    if not shutil.which("xelatex") and not shutil.which("pdflatex"):
        compiler = "unavailable: xelatex and pdflatex were not found"
    else:
        compiler = "present but not invoked; source generation is not compilation QA"
    output = Path(args.output)
    qa_path = Path(args.qa)
    for path in (output, qa_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([pandoc, str(manuscript), "--from=gfm", "--to=latex", "--standalone",
                    "--metadata=documentclass=article", "--metadata=classoption=11pt",
                    "--metadata=geometry:margin=1in", f"--metadata=title={TITLE}",
                    "--metadata=author=Anonymous evidence draft", "--metadata=date=2026-11-18",
                    f"--output={output}"], check=True)
    text = output.read_text()
    text = text.replace("\\else % if luatex or xetex\n  \\usepackage{unicode-math}",
                        "\\else % if luatex or xetex\n  \\usepackage{unicode-math}\n  \\usepackage{xeCJK}", 1)
    output.write_text(HEADER + text)
    text = output.read_text()
    for token in ["\\documentclass[", "11pt]{article}", "\\title{Contamination-aware evaluation",
                  "\\begin{document}", "\\end{document}", "\\usepackage{xeCJK}"]:
        assert token in text, token
    assert text.count("\\begin{longtable}") == text.count("\\end{longtable}")
    for number in range(1, 9):
        assert f"{{[}}{number}{{]}}" in text, number
    for value in ["0.844264", "0.851460", "162,489", "0.7715", "199,494,385,321",
                  "1b4b5432758290c93d4d95063b257f751ed5d84694bd4f6c79c6a004be832bbd"]:
        assert value in text, value
    report = {"status": "PASS_ARXIV_SOURCE_CONTENT_QA_COMPILE_NOT_AVAILABLE",
              "source": str(output), "source_sha256": sha256(output), "manuscript_sha256": source_hash,
              "tables": text.count("\\begin{longtable}"), "references": 8, "compiler": compiler,
              "reading_pdf_sha256": "473eaa07d6666e683c1b4954af8f8d67303f93d11ea3481d8d5c8744e4852d0b",
              "limitations": ["LaTeX source is generated for editing; no compiled PDF from this source has been validated",
                              "XeLaTeX and xeCJK are required for the Chinese stacking-energy phrase in Q4",
                              "Evidence draft remains not submission-ready"]}
    qa_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
