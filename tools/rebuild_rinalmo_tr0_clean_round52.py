import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ("bprna_ts0", "bprna_new", "archiveii_embok_clean", "ref_pdb_ts1",
          "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    audit_root = Path(args.audit_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        audit_path = audit_root / f"{split}.audit.json"
        audit = json.loads(audit_path.read_text())
        source = Path(audit["split"])
        raw_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        assert raw_sha256 == audit["split_sha256"]
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        assert len(records) == audit["raw_n"]
        removed = set(audit["removed_exact_indices"]) | set(audit["removed_homology_only_indices"])
        assert len(records) - len(removed) == audit["clean_n"]
        clean = [record for index, record in enumerate(records) if index not in removed]
        assert len(clean) == audit["clean_n"]
        serialized = "".join(json.dumps(record, sort_keys=True) + "\n" for record in clean)
        output = out_dir / f"{split}.jsonl"
        output.write_text(serialized)
        assert len(output.read_text().splitlines()) == audit["clean_n"]
        rows.append({
            "split": split, "source": str(source), "source_sha256": raw_sha256,
            "audit": str(audit_path), "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "raw_n": audit["raw_n"], "clean_n": audit["clean_n"],
            "removed_exact": audit["removed_exact"],
            "removed_homology_only": audit["removed_homology_only"],
            "clean_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        })
    manifest = {
        "status": "PASS_RINALMO_TR0_CLEAN_ASSETS",
        "n_splits": len(rows), "rows": rows,
        "protocol": "Rebuilt from previously accepted exact+80% identity/query-coverage>=0.8 audit indices; original order and GT preserved.",
        "pretrain_overlap_risk": "RiNALMo pretraining also includes RNAcentral/nt/Rfam/Ensembl; these are TR0-clean downstream sets, not a guarantee of zero pretraining overlap.",
        "rfam_protocol": "Rfam remains a separate raw official evaluation with explicit pretraining-overlap risk, not silently relabeled TR0-clean.",
    }
    path = Path(args.manifest)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
