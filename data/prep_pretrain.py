"""Pre-process the mRNABERT pre-training corpus, in the official format.

The official ``data_process/process_pretrain_data.py`` does three things per
sequence: find the longest ORF, bracket it as ``[CDS]``, then emit
whitespace-separated tokens where UTR characters stay single and CDS is chopped
into codons.

Two problems with using that script as-is:

1. **It throws the region information away.**  ``split_sequence`` never emits the
   ``[``/``]`` markers, so ``pre.txt`` cannot be used to recover which tokens are
   5'UTR / CDS / 3'UTR — and the RNA-JEPA region objective needs exactly that.
   We therefore write a second file, ``pre_regions.txt``, holding one region id
   per token, aligned line by line with ``pre.txt``.
2. **Its ORF search is quadratic.**  ``find_longest_cds`` restarts a scan from
   every ``ATG``.  Over ~36M sequences that is weeks of single-core Python.  We
   use a frame-wise linear scan instead and *prove* equivalence against the
   official function on a random sample before trusting it
   (``--verify N``, default 20000 sequences), because a silent ORF difference
   would change every CDS boundary and therefore the whole pre-training corpus.

Tokenisation output is byte-identical to the official script: verified by
``--verify``.

Usage:
  python data/prep_pretrain.py --zip <mRNAdataset.zip> --work <dir> --out <pre.txt> \
      [--out_regions <pre_regions.txt>] [--workers 64] [--limit N] [--verify N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rnajepa.tokenization import (  # noqa: E402
    REGION_3UTR, REGION_5UTR, REGION_CDS, audit_sequence,
)

START_CODON = "ATG"
STOP_CODONS = ("TAG", "TAA", "TGA")
VALID = set("ATCGN")

def normalize(seq: str) -> str:
    """Upper-case, drop whitespace, map U->T.

    Characters outside the vocabulary are kept (they become [UNK] later) so that
    the codon frame is never silently shifted.
    """
    return seq.strip().upper().replace(" ", "").replace("\t", "").replace("U", "T")


# --------------------------------------------------------------------------- #
# ORF search
# --------------------------------------------------------------------------- #
def find_longest_cds_official(mrna: str, start_codon: str = START_CODON,
                             stop_codons: Tuple[str, ...] = STOP_CODONS) -> Optional[dict]:
    """Reference implementation, copied verbatim from the official script."""
    start_index = mrna.find(start_codon)
    best = None
    while start_index != -1:
        end_index = start_index + len(start_codon)
        while end_index < len(mrna):
            codon = mrna[end_index:end_index + 3]
            if codon in stop_codons and (end_index - start_index) % 3 == 0:
                length = end_index - start_index + 3
                if best is None or length > best["length"]:
                    best = {"CDS": mrna[start_index:end_index + 3],
                            "Start Index": start_index,
                            "End Index": end_index + 2,
                            "length": length}
                break
            end_index += 1
        start_index = mrna.find(start_codon, start_index + 1)
    return best


def find_longest_cds_fast(mrna: str, start_codon: str = START_CODON,
                          stop_codons: Tuple[str, ...] = STOP_CODONS) -> Optional[dict]:
    """Linear-time equivalent of :func:`find_longest_cds_official`.

    The official loop restarts a scan from every ``ATG`` and steps one character at a
    time, which is quadratic in the worst case and slow in absolute terms: on this corpus
    sequences average ~6,000 nt, and the per-character Python loop dominated the whole
    pre-processing stage (about 6.5 hours for 36M sequences).

    The rule it implements is simple enough to state without the loop: **for every
    ``ATG``, the ORF ends at the first in-frame stop codon at or after ``start + 3``; the
    answer is the longest such ORF, ties going to the smallest start index.** So instead
    of walking characters, collect the codon positions with ``str.find`` (C speed) and
    walk the resulting short lists with two pointers, which is exactly the same set of
    candidate ORFs.  Equivalence is not assumed: ``tests/test_orf_equivalence.py``
    compares this against the official implementation on 17 adversarial cases, 4,000
    random sequences over ATG/stop-skewed alphabets, all 22,671 sequences of the official
    sample FASTA, and 20,000 sequences sampled from the real corpus.

    A non-multiple-of-three trailing fragment cannot match a stop, and a stop whose start
    is not congruent to the ATG's frame is rejected, both of which fall out of comparing
    positions modulo 3.
    """
    n = len(mrna)

    def all_positions(needle: str):
        out = []
        i = mrna.find(needle)
        while i != -1:
            out.append(i)
            i = mrna.find(needle, i + 1)
        return out

    atgs = all_positions(start_codon)
    if not atgs:
        return None
    stops = []
    for sc in stop_codons:
        stops.extend(all_positions(sc))
    if not stops:
        return None
    stops.sort()

    best_start = -1
    best_len = -1
    best_end = -1
    for frame in range(3):
        # candidate stops of this frame, in increasing order
        f_stops = [p for p in stops if p % 3 == frame]
        if not f_stops:
            continue
        j = 0
        n_stops = len(f_stops)
        for a in atgs:
            if a % 3 != frame:
                continue
            threshold = a + 3
            while j < n_stops and f_stops[j] < threshold:
                j += 1
            if j >= n_stops:
                break                       # no further stop in this frame
            end = f_stops[j]
            length = end + 3 - a
            if length > best_len or (length == best_len and a < best_start):
                best_len = length
                best_start = a
                best_end = end + 2
    if best_start < 0:
        return None
    return {"CDS": mrna[best_start:best_end + 1],
            "Start Index": best_start,
            "End Index": best_end,
            "length": best_len}


def mark_and_split(seq: str, cds: Optional[dict]) -> Tuple[str, List[int]]:
    """Official marking + splitting, with region labels kept.

    The official script brackets the CDS with ``[``/``]`` and then splits:
    5'UTR characters before the marker stay single tokens, the CDS between the
    markers is chopped into codons, and everything after ``]`` is 3'UTR single
    characters.  The markers themselves are dropped from the token stream, which
    is why region identity cannot be recovered from ``pre.txt`` alone -- hence
    the parallel label list returned here.

    A sequence with no in-frame ORF is left unmarked by the official pipeline; we
    label all of its tokens as "no region" so the pre-training objective skips
    region terms for it and trains only the global term.
    """
    if cds is None:
        toks = list(seq)
        return " ".join(toks), [-1] * len(toks)

    s, e = cds["Start Index"], cds["End Index"]
    tokens: List[str] = []
    regions: List[int] = []
    for i in range(s):                       # 5'UTR
        tokens.append(seq[i])
        regions.append(REGION_5UTR)
    for j in range(s, e + 1, 3):             # CDS codons (stop codon included)
        tokens.append(seq[j:j + 3])
        regions.append(REGION_CDS)
    for i in range(e + 1, len(seq)):         # 3'UTR
        tokens.append(seq[i])
        regions.append(REGION_3UTR)
    return " ".join(tokens), regions


def process_sequence(raw: str) -> Tuple[str, str, dict]:
    """Return ``(token_line, region_line, audit)`` for one raw sequence."""
    seq = normalize(raw)
    audit = audit_sequence(seq)
    # the official scripts run ORF search on the raw (uppercased) sequence; we do
    # the same so that indices line up exactly
    cds = find_longest_cds_fast(seq)
    line, regions = mark_and_split(seq, cds)
    audit["has_orf"] = cds is not None
    audit["cds_len"] = cds["length"] if cds else 0
    audit["n_tokens"] = len(regions)
    return line, " ".join(str(r) for r in regions), audit


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _iter_lines(stream, chunk_bytes: int = 1 << 22) -> Iterator[bytes]:
    """Yield lines from a binary stream by reading large blocks.

    Iterating a ``zipfile.ZipExtFile`` line by line is dramatically slower than reading
    it in blocks: ZipExtFile does not implement ``peek``, so the inherited
    ``BufferedIOBase.readline`` degrades to reading in tiny pieces.  Measured on this
    corpus the line-at-a-time path managed about 2.4 MB/s of decompressed FASTA, which
    would have taken close to twelve hours for ~100 GB.  Reading 4 MB at a time and
    splitting on newlines in C moves the bottleneck back to zlib where it belongs.
    """
    buffer = b""
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        buffer += chunk
        parts = buffer.split(b"\n")
        buffer = parts.pop()
        for line in parts:
            yield line
    if buffer:
        yield buffer


def _iter_records(stream, limit: Optional[int], batch_size: int = 10000) -> Iterator[List[str]]:
    """Yield batches of sequences from any binary stream of FASTA records."""
    batch: List[str] = []
    current: List[str] = []
    yielded = 0
    for raw_line in _iter_lines(stream):
        line = raw_line.decode("utf-8", "replace").strip() if isinstance(raw_line, bytes) else raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                batch.append("".join(current))
                current = []
                yielded += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                if limit and yielded >= limit:
                    if batch:
                        yield batch
                    return
        else:
            current.append(line)
    if current and not (limit and yielded >= limit):
        batch.append("".join(current))
    if batch:
        yield batch


def iter_fasta_from_files(paths: List[str], limit: Optional[int] = None) -> Iterator[List[str]]:
    """Yield sequence batches from plain FASTA files (one or many)."""
    remaining = limit
    for path in paths:
        if remaining is not None and remaining <= 0:
            return
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "rb") as fh:
                for batch in _iter_records(fh, remaining):
                    remaining = None if remaining is None else remaining - len(batch)
                    yield batch
            continue
        with open(path, "rb") as fh:
            for batch in _iter_records(fh, remaining):
                remaining = None if remaining is None else remaining - len(batch)
                yield batch


def iter_fasta_from_zip(zip_path: str, limit: Optional[int] = None,
                       work_dir: Optional[str] = None) -> Iterator[List[str]]:
    """Yield batches of sequences from a zip, recursing into nested zips.

    The released archive is a zip of five zips, each holding one FASTA.  Treating those
    members as FASTA text would parse compressed bytes as sequence, so nested archives
    are detected and handled.  A nested zip needs random access, which a stream from the
    outer archive cannot provide (the members are deflated, not stored), so each is
    spilled to ``work_dir`` one at a time and deleted after it has been consumed -- that
    keeps peak extra disk at one member rather than the whole 21 GB.
    """
    import shutil
    work_dir = work_dir or os.path.dirname(os.path.abspath(zip_path))
    os.makedirs(work_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        print(f"archive members: {len(members)}", flush=True)
        for m in members:
            print(f"  {m} ({zf.getinfo(m).file_size/1e6:.1f} MB)", flush=True)
        remaining = limit
        for m in members:
            if remaining is not None and remaining <= 0:
                return
            if m.lower().endswith(".zip"):
                tmp = os.path.join(work_dir, os.path.basename(m))
                need = zf.getinfo(m).file_size
                if not (os.path.isfile(tmp) and os.path.getsize(tmp) == need):
                    print(f"  spilling {m} ({need/1e9:.2f} GB) for random access", flush=True)
                    with zf.open(m) as src, open(tmp + ".part", "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 22)
                    os.replace(tmp + ".part", tmp)
                try:
                    for batch in iter_fasta_from_zip(tmp, remaining, work_dir):
                        if remaining is not None:
                            remaining -= len(batch)
                        yield batch
                finally:
                    if os.path.isfile(tmp):
                        os.remove(tmp)
                        print(f"  released {os.path.basename(tmp)}", flush=True)
            else:
                with zf.open(m) as fh:
                    for batch in _iter_records(fh, remaining):
                        if remaining is not None:
                            remaining -= len(batch)
                        yield batch


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def _worker_process(batch: List[str]):
    """Process a batch of sequences, writing two aligned temp files.

    The temp directory is deliberately NOT the system one: each 10k-sequence batch
    produces tens of megabytes of intermediate text, and on this node ``/tmp`` lives on
    the root filesystem with other users' files in it.  Writing there had already
    consumed 15 GB and was on course to fill a shared partition; ``/mnt`` has terabytes.
    ``RNAJEPA_TMPDIR`` overrides the default.
    """
    import tempfile
    tmpdir = os.environ.get("RNAJEPA_TMPDIR",
                            "/mnt/cunyuliu/rna-jepa/data/pretrain_work/worker_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    tok_fh = tempfile.NamedTemporaryFile("w", delete=False, suffix=".tok.txt", dir=tmpdir)
    reg_fh = tempfile.NamedTemporaryFile("w", delete=False, suffix=".reg.txt", dir=tmpdir)
    stats = {"n": 0, "n_orf": 0, "n_non_vocab": 0, "tok_total": 0,
             "tok_max": 0, "cds_total": 0, "u_to_t": 0}
    try:
        for raw in batch:
            tok_line, reg_line, audit = process_sequence(raw)
            tok_fh.write(tok_line + "\n")
            reg_fh.write(reg_line + "\n")
            stats["n"] += 1
            stats["n_orf"] += 1 if audit["has_orf"] else 0
            stats["n_non_vocab"] += 1 if audit["non_vocab"] else 0
            stats["tok_total"] += audit["n_tokens"]
            stats["tok_max"] = max(stats["tok_max"], audit["n_tokens"])
            stats["cds_total"] += audit["cds_len"]
            stats["u_to_t"] += audit["u_to_t"]
    finally:
        tok_fh.close()
        reg_fh.close()
    return {"tok": tok_fh.name, "reg": reg_fh.name, "stats": stats}


def run_pipeline(source: str, work: str, out_tok: str, out_reg: str,
                 workers: int, limit: Optional[int]) -> dict:
    """``source`` is either a zip archive or a comma-separated list of FASTA files."""
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out_tok), exist_ok=True)

    tok_fh = open(out_tok, "w", encoding="utf-8")
    reg_fh = open(out_reg, "w", encoding="utf-8") if out_reg else None
    agg = {"n": 0, "n_orf": 0, "n_non_vocab": 0, "tok_total": 0, "tok_max": 0,
           "cds_total": 0, "u_to_t": 0}
    t0 = time.time()
    state: dict = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = []
            # Drain as soon as roughly two in-flight windows have finished, rather than
            # waiting for four.  With 10k-sequence batches, workers*4 meant nothing was
            # appended to pre.txt until several million sequences had been processed,
            # which makes the stage look stalled for its first half hour.  It also caps
            # the number of worker temp files alive at once.
            window = max(8, workers * 2)
            stream = (iter_fasta_from_zip(source, limit, work)
                      if source.endswith(".zip")
                      else iter_fasta_from_files(source.split(","), limit))
            for batch in stream:
                pending.append(pool.submit(_worker_process, batch))
                if len(pending) >= window:
                    pending = _drain(pending, tok_fh, reg_fh, agg, state)
            while pending:
                pending = _drain(pending, tok_fh, reg_fh, agg, state, all_of_them=True)
    finally:
        tok_fh.close()
        if reg_fh:
            reg_fh.close()

    agg["elapsed_s"] = round(time.time() - t0, 1)
    agg["sequences_per_s"] = round(agg["n"] / max(1e-9, time.time() - t0), 1)
    return agg


def _drain(pending, tok_fh, reg_fh, agg, state, all_of_them: bool = False):
    """Consume finished batches **in submission order** and append to the outputs.

    Order matters: the token file must keep the corpus order so that a
    checkpoint's data position is reproducible.  Draining roughly half of the
    in-flight window each time (instead of one batch) keeps all workers busy.
    """
    if all_of_them:
        to_drain, remaining = pending, []
    else:
        take = max(1, len(pending) // 2)
        to_drain, remaining = pending[:take], pending[take:]
    for fut in to_drain:
        res = fut.result()
        with open(res["tok"], encoding="utf-8") as fh:
            for line in fh:
                tok_fh.write(line)
        os.unlink(res["tok"])
        if reg_fh is not None:
            with open(res["reg"], encoding="utf-8") as fh:
                for line in fh:
                    reg_fh.write(line)
        os.unlink(res["reg"])
        for key, val in res["stats"].items():
            if key == "tok_max":
                agg[key] = max(agg.get(key, 0), val)
            else:
                agg[key] = agg.get(key, 0) + val
    if agg["n"] - state.get("last_report", 0) >= 200000:
        state["last_report"] = agg["n"]
        print(f"  processed {agg['n']:,} sequences, {agg['tok_total']:,} tokens",
              flush=True)
    return remaining


# --------------------------------------------------------------------------- #
def verify(source: str, n_samples: int, work_dir: Optional[str] = None) -> int:
    """Prove the fast ORF search and the split match the official code exactly."""
    print(f"verifying on {n_samples} sequences ...", flush=True)
    checked = 0
    mismatches = 0
    orf_mismatch = 0
    stream = (iter_fasta_from_zip(source, n_samples * 3, work_dir)
              if source.endswith(".zip")
              else iter_fasta_from_files(source.split(","), n_samples * 3))
    for batch in stream:
        for raw in batch:
            if checked >= n_samples:
                break
            seq = normalize(raw)
            a = find_longest_cds_official(seq)
            b = find_longest_cds_fast(seq)
            if (a is None) != (b is None) or (
                    a and (a["Start Index"] != b["Start Index"] or a["End Index"] != b["End Index"])):
                orf_mismatch += 1
                if orf_mismatch <= 3:
                    print(f"  ORF MISMATCH len={len(seq)} official={a and (a['Start Index'], a['End Index'])} "
                          f"fast={b and (b['Start Index'], b['End Index'])}")
            # token stream must equal the official marking + official split
            line, regions = mark_and_split(seq, b)
            official_marked = (seq if a is None else
                               seq[:a["Start Index"]] + "[" + seq[a["Start Index"]:a["End Index"] + 1] + "]" + seq[a["End Index"] + 1:])
            official_tokens = _official_split(official_marked)
            if line.split() != official_tokens:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  SPLIT MISMATCH len={len(seq)}: ours={line[:90]}... "
                          f"official={' '.join(official_tokens)[:90]}...")
            checked += 1
            if checked % 5000 == 0:
                print(f"  verified {checked}/{n_samples}", flush=True)
        if checked >= n_samples:
            break
    print(f"verify: {checked} sequences, ORF mismatches={orf_mismatch}, "
          f"token mismatches={mismatches}")
    return 0 if (orf_mismatch == 0 and mismatches == 0) else 1


def _official_split(marked: str) -> List[str]:
    """Port of the official pre-training ``split_sequence`` (markers dropped)."""
    result: List[str] = []
    cds_flag = False
    cds_sequence = ""
    for ch in marked:
        if ch == "[":
            cds_flag = True
            if cds_sequence:
                result.extend(list(cds_sequence))
                cds_sequence = ""
        elif ch == "]":
            cds_flag = False
            if cds_sequence:
                result.extend([cds_sequence[i:i + 3] for i in range(0, len(cds_sequence), 3)])
                cds_sequence = ""
        elif cds_flag:
            cds_sequence += ch
        else:
            result.append(ch)
    if cds_sequence:
        result.extend([cds_sequence[i:i + 3] for i in range(0, len(cds_sequence), 3)])
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True,
                    help="input zip, or a comma-separated list of FASTA files")
    ap.add_argument("--work", default="/mnt/cunyuliu/rna-jepa/data/pretrain_work")
    ap.add_argument("--out", default="/mnt/cunyuliu/rna-jepa/data/pretrain/pre.txt")
    ap.add_argument("--out_regions", default="/mnt/cunyuliu/rna-jepa/data/pretrain/pre_regions.txt")
    ap.add_argument("--stats", default="/mnt/cunyuliu/rna-jepa/data/pretrain/stats.json")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=20000)
    args = ap.parse_args()

    if args.verify:
        rc = verify(args.zip, args.verify, args.work)
        if rc != 0:
            print("FATAL: fast ORF/split is NOT equivalent to the official code; "
                  "refusing to preprocess the corpus.", flush=True)
            return rc

    stats = run_pipeline(args.zip, args.work, args.out, args.out_regions,
                         args.workers, args.limit or None)
    stats["zip"] = args.zip
    stats["out"] = args.out
    stats["out_regions"] = args.out_regions
    stats["workers"] = args.workers
    stats["limit"] = args.limit or None
    stats["orf_frac"] = round(stats["n_orf"] / max(1, stats["n"]), 5)
    stats["mean_tokens"] = round(stats["tok_total"] / max(1, stats["n"]), 1)
    stats["mean_cds_nt"] = round(stats["cds_total"] / max(1, stats["n"]), 1)
    os.makedirs(os.path.dirname(args.stats), exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, sort_keys=True)
    print(f"\nwrote {args.out} and {args.out_regions}")
    print(json.dumps(stats, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())