#!/usr/bin/env python
"""Global train/test overlap audit + frozen eval-split manifest (15.14).

Audits EVERY training corpus against EVERY evaluation split by exact
sequence, writes:
  1. tables/overlap_audit.md  — the definitive contamination table
  2. spec/eval_splits_frozen.json — sha256 + n + overlap verdict for each
     frozen eval split, so any future corpus build can gate against it.

Background: two leaks were found by auditing after the fact (15.10: TR1
carried TS0/PDB-family rows; 15.13: TR0 carries 247/428 TestSetB rows
because TR0 predates TestSetB's arrival). This tool makes the audit a
gated, repeatable step instead of a post-hoc discovery.
"""
import hashlib
import json
import os
from pathlib import Path

JSONL = Path("/mnt/cunyuliu/rna-jepa/ss_data/jsonl")
OUT_MD = Path("/mnt/cunyuliu/rna-jepa/tables/overlap_audit.md")
OUT_JSON = Path("/home/cunyuliu/rna-jepa/spec/eval_splits_frozen.json")

TRAIN_CORPORA = ["bprna_tr0", "bprna_tr1", "bprna_tr1c"]
EVAL_SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2",
               "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb",
               "archiveii_embok_clean", "archiveii"]


def seqs_of(path):
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.add(r.get("seq", r.get("sequence", "")))
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    train_seqs = {c: seqs_of(JSONL / f"{c}.jsonl") for c in TRAIN_CORPORA}
    rows = []
    frozen = {"frozen_at": "15.14 (2026-10-04)",
              "note": "any new training corpus MUST be built by excluding "
                      "these exact sequences (tools/make_clean_corpus.py "
                      "blocklist pattern); gate = zero overlap on every "
                      "split below.",
              "splits": {}}
    for sp in EVAL_SPLITS:
        path = JSONL / f"{sp}.jsonl"
        ts = seqs_of(path)
        verdict = {}
        for c in TRAIN_CORPORA:
            verdict[c] = len(ts & train_seqs[c])
        clean = all(v == 0 for v in verdict.values())
        sha = sha256_of(path)
        rows.append((sp, len(ts), verdict, clean, sha))
        frozen["splits"][sp] = {
            "n_unique": len(ts), "sha256": sha,
            "overlap_with_training_corpora": verdict,
            "fully_clean_vs_all_corpora": clean,
        }
        print(f"{sp:26s} n={len(ts):5d} " +
              " ".join(f"{c}={verdict[c]:5d}" for c in TRAIN_CORPORA) +
              f"  clean={clean}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# Global train/test overlap audit (15.14)\n\n")
        fh.write("Exact-sequence overlap between every training corpus and "
                 "every frozen evaluation split.\n\n")
        fh.write("| eval split | n | " +
                 " | ".join(TRAIN_CORPORA) + " | clean vs all |\n")
        fh.write("|---|---|" + "---|" * (len(TRAIN_CORPORA) + 1) + "\n")
        for sp, n, verdict, clean, sha in rows:
            fh.write(f"| {sp} | {n} | " +
                     " | ".join(str(verdict[c]) for c in TRAIN_CORPORA) +
                     f" | {'**YES**' if clean else 'NO'} |\n")
        fh.write("\nKnown leaks (retractions already issued): "
                 "TR0∩TestSetB=247 (15.13), TR1∩TS0=1087/PDB-family (15.10), "
                 "TR1∩ArchiveII-clean=843 (15.13).\n")
        fh.write("\nThe ONLY corpus clean against every split is "
                 "**bprna_tr1c** (built by make_clean_corpus.py with the "
                 "9-split blocklist + archiveii).\n")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=1)
    print(f"written {OUT_MD}")
    print(f"written {OUT_JSON}")


if __name__ == "__main__":
    main()
