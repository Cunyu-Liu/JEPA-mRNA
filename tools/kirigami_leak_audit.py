#!/usr/bin/env python3
"""T-A50b addendum: sequence-level leak audit of Kirigami's TR0 training corpus
against our ArchiveII-clean (2544) and TestSetB (428) board splits.

Kirigami main.ckpt was trained on TR0.dbn ONLY (DataModule hardcodes
TR0/VL0/TS0; archiveII/RNAStrAlign/bpRNAnew files in their data/ dir are
paper-benchmark extras, not training data). But TR0 itself is known to overlap
TestSetB (RiNALMo-ft case: 247/428) — so we audit directly, sequence-level:
- exact match of normalized sequences (T->U, uppercase)
- also report TR0 count and sha256 of the file for provenance
Output: eval_decision/kirigami_leak_audit.json
"""
import hashlib, json, re, sys

TR0 = "/mnt/cunyuliu/relocated_home/rna_baselines_src/kirigami/data/TR0.dbn"
SPLITS = {
    "archiveii_embok_clean": "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/archiveii_embok_clean.jsonl",
    "testsetb": "/mnt/cunyuliu/rna-jepa/ss_data/jsonl/testsetb.jsonl",
}
OUT = "/mnt/cunyuliu/rna-jepa/eval_decision/kirigami_leak_audit.json"

def norm(seq):
    return seq.upper().replace("T", "U")

def main():
    # load TR0 sequences
    lines = open(TR0).read().splitlines()
    n_rec = len(lines) // 3
    tr0_seqs = set()
    for i in range(n_rec):
        s = lines[3 * i + 1]
        if s and not s.startswith(">"):
            tr0_seqs.add(norm(s))
    print("TR0 records:", n_rec, "unique normalized seqs:", len(tr0_seqs))

    # provenance
    h = hashlib.sha256(open(TR0, "rb").read()).hexdigest()
    print("TR0.dbn sha256:", h)

    results = {}
    for split, path in SPLITS.items():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        n = len(rows)
        hits = [i for i, r in enumerate(rows) if norm(r["seq"]) in tr0_seqs]
        results[split] = {
            "n_total": n,
            "n_leak": len(hits),
            "leak_fraction": round(len(hits) / n, 4),
            "leak_row_idxs_first20": hits[:20],
        }
        print(split, "n:", n, "leak:", len(hits))

    json.dump({
        "audit": "Kirigami TR0 training-corpus leak check (sequence-level, exact match after T->U/upper)",
        "tr0_dbn": TR0,
        "tr0_records": n_rec,
        "tr0_unique_seqs": len(tr0_seqs),
        "tr0_sha256": h,
        "results": results,
    }, open(OUT, "w"), indent=1)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
