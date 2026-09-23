#!/bin/bash
# Extract the corpus FASTAs once, so pre-processing can read plain files.
#
# The archive is a zip of five zips, each holding one FASTA (~100 GB of FASTA in total
# from 20 GB of archive).  Reading those FASTAs straight out of the nested zip makes the
# reader single-threaded and CPU-bound: it has to inflate the stream itself, and on a
# node shared with dozens of other users it managed about 10 MB/s, i.e. close to three
# hours before the 48 worker processes could even be fed.
#
# Extracting first moves that work into zlib's C path in one pass, after which
# pre-processing reads ordinary files at I/O speed.  It costs ~100 GB of extra disk
# (there are terabytes) and about fifteen minutes.
#
# Idempotent: a FASTA whose size already matches is left alone.
#
# Usage: extract_corpus.sh [dest_dir]
set -uo pipefail

ROOT="${RNAJEPA_ROOT:-/home/cunyuliu/rna-jepa}"
ART="${RNAJEPA_ART:-/mnt/cunyuliu/rna-jepa}"
ZIP="${1:-$ART/data/raw/mRNAdataset.zip}"
DEST="${2:-$ART/data/pretrain_fasta}"
WORK="$ART/data/pretrain_work"
PY="${RNAJEPA_PYTHON:-/home/cunyuliu/miniconda3/envs/lucaone/bin/python}"

mkdir -p "$DEST" "$WORK"
echo "[extract] $(date '+%F %T') archive=$ZIP dest=$DEST"

"$PY" - "$ZIP" "$DEST" "$WORK" <<'PYEOF'
import os, shutil, sys, time, zipfile

zip_path, dest, work = sys.argv[1], sys.argv[2], sys.argv[3]

def copy_member(zf, member, out_path, chunk=1 << 23):
    """Stream one zip member to a file, resuming by size if already complete."""
    need = zf.getinfo(member).file_size
    if os.path.isfile(out_path) and os.path.getsize(out_path) == need:
        print(f"  skip {os.path.basename(out_path)} ({need/1e9:.2f} GB already present)")
        return False
    part = out_path + ".part"
    t0 = time.time()
    with zf.open(member) as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, chunk)
    os.replace(part, out_path)
    dt = max(1e-9, time.time() - t0)
    print(f"  {os.path.basename(out_path)} {need/1e9:.2f} GB in {dt:.0f}s "
          f"({need/1e6/dt:.0f} MB/s)", flush=True)
    return True

with zipfile.ZipFile(zip_path) as outer:
    members = [m for m in outer.namelist() if m.lower().endswith(".zip")]
    print(f"  {len(members)} inner archive(s)")
    for m in members:
        inner_path = os.path.join(work, os.path.basename(m))
        copy_member(outer, m, inner_path)
        with zipfile.ZipFile(inner_path) as inner:
            for fasta in inner.namelist():
                if fasta.endswith("/"):
                    continue
                idx = os.path.splitext(os.path.basename(m))[0].split("_")[-1]
                out = os.path.join(dest, f"{idx}_{os.path.basename(fasta)}")
                copy_member(inner, fasta, out)
        # the inner archive has served its purpose; the FASTA supersedes it
        try:
            os.remove(inner_path)
            print(f"  released {os.path.basename(inner_path)}")
        except OSError:
            pass
print("extraction complete")
PYEOF

echo "[extract] $(date '+%F %T') listing"
ls -la "$DEST" | awk '{print "  ", $5, $9}'