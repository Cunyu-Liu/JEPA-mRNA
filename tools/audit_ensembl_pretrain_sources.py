import json
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def fetch(url):
    request = Request(url, headers={"User-Agent": "RNA-JEV-pretraining-exposure-audit/1.0"})
    try:
        with urlopen(request, timeout=40) as response:
            parser = Links()
            parser.feed(response.read().decode("utf-8", errors="replace"))
            return parser.hrefs, None
    except HTTPError as error:
        if error.code == 404:
            return [], None
        return [], str(error)
    except Exception as error:
        return [], str(error)


def main():
    jobs = [("ensembl_109", "https://ftp.ensembl.org/pub/release-109/fasta/", 2),
            ("ensembl_bacteria_56", "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-56/fasta/", 3)]
    output = {"status": "PASS_ENSEMBL_EXACT_NCRNA_SOURCE_LIST_WITH_DOWNLOAD_PENDING",
              "license_evidence": "https://www.ensembl.org/info/about/legal/disclaimer.html",
              "third_party_constraint_note": "Ensembl states no project-data restrictions; users must separately establish any third-party constraints.",
              "sources": {}}
    for name, root, depth in jobs:
        directories = [root]
        errors = []
        for _ in range(depth - 1):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(fetch, directories))
            directories = []
            for base, (hrefs, error) in zip([root] if _ == 0 else previous, results):
                if error:
                    errors.append({"url": base, "error": error})
                for href in hrefs:
                    if href.endswith("/") and not href.startswith(("?", "/", "http", "..")):
                        directories.append(base + href)
            previous = directories
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(fetch, directories))
        files = []
        for base, (hrefs, error) in zip(directories, results):
            if error:
                errors.append({"url": base, "error": error})
            files.extend(base + href for href in hrefs if href.endswith(".ncrna.fa.gz"))
        output["sources"][name] = {"root": root, "n_files": len(files), "files": sorted(files), "errors": errors}
        if errors:
            output["status"] = "BLOCKED_INCOMPLETE_ENSEMBL_SOURCE_LIST"
    (ROOT / "ensembl_sources.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "counts": {name: source["n_files"] for name, source in output["sources"].items()},
                      "errors": {name: len(source["errors"]) for name, source in output["sources"].items()}}, indent=2))


if __name__ == "__main__":
    main()
