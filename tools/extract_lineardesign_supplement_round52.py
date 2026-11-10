import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    lines = Path(args.text).read_text().splitlines()
    entries = []
    current = None
    for line in lines:
        if line.startswith(">"):
            if current:
                entries.append(current)
            header = line[1:].strip()
            if header.startswith("Sequence ") or header.startswith("BNT") or header.startswith("gE-") or "UTR" in header:
                current = {"header": header, "lines": []}
            else:
                current = None
        elif current and line.strip() and set(line.strip()) <= set("ACGU"):
            current["lines"].append(line.strip())
    if current:
        entries.append(current)
    records = []
    skipped = []
    for entry in entries:
        header = entry["header"]
        sequence = "".join(entry["lines"]).upper()
        if not sequence:
            continue
        if header.startswith("Sequence "):
            name = "covid_" + header[len("Sequence "):].split(":")[0]
        elif header.startswith("BNT"):
            name = "covid_BNT"
        elif header.startswith("gE-"):
            name = "vzv_" + header.split(":")[0]
        else:
            name = ("covid" if records and not any(r["name"].startswith("vzv_") for r in records) else "vzv") + "_" + header.replace(" ", "_")
        if "UTR" not in name and (len(sequence) % 3 or (name.startswith("covid_") and len(sequence) != 3819)):
            skipped.append({"name": name, "length": len(sequence), "reason": "coding sequence is incomplete or not triplet-preserving"})
            continue
        records.append({"name": name, "sequence": sequence})
    names = [record["name"] for record in records]
    assert len(records) == 19 and len(set(names)) == 19
    coding = [record for record in records if "UTR" not in record["name"]]
    assert len(coding) == 15
    assert all(len(record["sequence"]) == 3819 for record in coding if record["name"].startswith("covid_"))
    assert len({len(record["sequence"]) for record in coding if record["name"].startswith("vzv_")}) == 1
    output = "".join(f">{record['name']}\n{record['sequence']}\n" for record in records)
    fasta = Path(args.fasta)
    fasta.write_text(output)
    manifest = {
        "status": "PASS_LINEARDESIGN_SUPPLEMENT_SEQUENCE_EXTRACTION",
        "n_coding_sequences": len(coding),
        "n_utr_sequences": len(records) - len(coding),
        "records": [{"name": record["name"], "length": len(record["sequence"]),
                     "sequence_sha256": hashlib.sha256(record["sequence"].encode()).hexdigest()}
                    for record in records],
        "fasta_sha256": hashlib.sha256(fasta.read_bytes()).hexdigest(),
        "skipped": skipped,
        "limitations": [
            "COVID wild-type and the length-inconsistent BNT text are not accepted as verified coding sequences.",
            "The figshare FASTA and uniprot_114.zip remain unavailable; this is a partial asset recovery.",
            "No new design or biological validation has been performed.",
        ],
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
