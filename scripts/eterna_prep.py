import json
import sys

sp, fa, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
import os
os.makedirs(outdir, exist_ok=True)
recs = []
with open(f"/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        seq = str(r["seq"]).upper().replace("T", "U")
        recs.append((r["name"], seq))
with open(fa, "w") as out:
    for n, s in recs:
        out.write(f">{n}\n{s}\n")
for i, (n, s) in enumerate(recs):
    safe = f"{i:06d}"
    with open(os.path.join(outdir, safe + ".fa"), "w") as fh:
        fh.write(f">{n}\n{s}\n")
print(f"fasta {fa} written; {len(recs)} per-seq files in {outdir}")
