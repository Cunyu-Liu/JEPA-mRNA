import glob
import os
import sys

jobdir, dbn = sys.argv[1], sys.argv[2]
parts = []
for f in sorted(glob.glob(os.path.join(jobdir, "*.fa"))):
    fa_lines = [l.strip() for l in open(f) if l.strip()]
    seq = fa_lines[1]
    stem = f[:-3]
    dpath = stem + ".dbn"
    struct = None
    if os.path.exists(dpath):
        dlines = [l.strip() for l in open(dpath) if l.strip()]
        for i, l in enumerate(dlines):
            if l == ">structure":
                struct = dlines[i + 1] if i + 1 < len(dlines) else None
                break
    parts.append((fa_lines[0], seq, struct))
n_missing = sum(1 for _, _, s in parts if not s)
with open(dbn, "w") as out:
    for hdr, seq, struct in parts:
        out.write(f"{hdr}\n{seq}\n{struct or ''}\n")
print(f"dbn {dbn}: {len(parts)} records, {n_missing} missing structures")
