import json, hashlib
from collections import Counter
JSONL = "/mnt/cunyuliu/rna-jepa/ss_data/jsonl"

BLK = set()
BLK_FILES = ["ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard",
             "bprna_ts0", "bprna_new", "bprna_vl0", "testsetb", "archiveii"]
for s in BLK_FILES:
    for l in open(f"{JSONL}/{s}.jsonl"):
        r = json.loads(l)
        BLK.add(r.get("seq", r.get("sequence", "")))
print("blocklist:", len(BLK), "from", len(BLK_FILES), "eval splits")

kept, dropped = [], 0
for l in open(f"{JSONL}/bprna_tr1.jsonl"):
    r = json.loads(l)
    if r.get("seq") in BLK:
        dropped += 1
        continue
    kept.append(r)
print(f"tr1 45865 -> kept {len(kept)}, dropped {dropped}")

out = f"{JSONL}/bprna_tr1c.jsonl"
with open(out, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
h = hashlib.sha256(open(out, "rb").read()).hexdigest()

c = Counter()
for r in kept:
    L = len(r["seq"])
    c["<=100" if L <= 100 else ("100-200" if L <= 200 else ("200-400" if L <= 400 else ">400"))] += 1
src = Counter()
for r in kept:
    nm = r.get("name", "")
    if nm.startswith("bpRNA_RFAM"): src["RFAM"] += 1
    elif nm.startswith("bpRNA_CRW"): src["CRW"] += 1
    else: src["plk"] += 1
tp = sum(r.get("n_pairs", 0) for r in kept)

man = {
    "generated_by": "tools/make_clean_corpus.py (15.10 decontamination)",
    "derived_from": "bprna_tr1.jsonl (sha256 473f62458695554335a5a6911bcc590aa788fd1d70b2ebc6ab4830baf0e93929)",
    "decontamination": {
        "method": "exact-sequence removal",
        "blocked_splits": BLK_FILES,
        "n_dropped": dropped,
    },
    "max_length": max(len(r["seq"]) for r in kept),
    "mean_length": sum(len(r["seq"]) for r in kept) / len(kept),
    "n_records_written": len(kept),
    "n_unique_sequences": len({r["seq"] for r in kept}),
    "out": out,
    "out_sha256": h,
    "length_histogram": dict(c),
    "source_mix": dict(src),
    "total_pairs": tp,
}
with open(f"{JSONL}/bprna_tr1c.manifest.json", "w") as fh:
    json.dump(man, fh, indent=1)
print("written", out)
print(json.dumps(man, indent=1)[:900])
