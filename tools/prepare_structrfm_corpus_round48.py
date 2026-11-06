import argparse
import csv
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    count = 0
    with gzip.open(args.archive, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            member = archive.next()
            if member is None or not member.isfile() or not member.name.endswith(".csv"):
                raise RuntimeError("expected one CSV in archive")
            source = archive.extractfile(member)
            with temporary.open("w", encoding="utf-8") as dst:
                for name, sequence, *_ in csv.reader((line.decode() for line in source)):
                    if name == "name":
                        continue
                    sequence = sequence.upper().replace("T", "U")
                    if not sequence or set(sequence) - set("ACGUN"):
                        raise ValueError(f"invalid sequence at row {count}: {name}")
                    dst.write(f">pretrain_{count}\n{sequence}\n")
                    count += 1
            if archive.next() is not None:
                raise RuntimeError("expected exactly one CSV in archive")
    temporary.replace(output)
    sha256 = hashlib.sha256()
    with output.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            sha256.update(block)
    metadata = {"status": "PASS", "n_sequences": count, "fasta": str(output),
                "fasta_sha256": sha256.hexdigest(), "archive": args.archive}
    metadata_path = Path(args.metadata)
    temporary_metadata = metadata_path.with_suffix(".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
