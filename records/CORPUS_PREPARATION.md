# Corpus preparation: what the released archive actually contains, and proof of equivalence

## The archive is a zip of zips

`mRNAdataset.zip` (20,021,485,711 bytes, md5 `bf8bc5c946a0bd3b07716b1c7f785d54`, verified
against the published checksum) contains exactly five members:

| member | size |
|---|---|
| mRNAdataset_0.zip | 4,612,562,890 |
| mRNAdataset_1.zip | 5,831,694,911 |
| mRNAdataset_2.zip | 8,670,821,112 |
| mRNAdataset_3.zip | 1,834,101,267 |
| mRNAdataset_4.zip | 25,553,951 |

Each inner archive holds a single FASTA (`0.fasta` alone is 21.8 GB), and the inner members
are DEFLATED, so a stream from the outer archive cannot be used as a zip and each must be
spilled to disk one at a time. `iter_fasta_from_zip` now recurses, spilling each inner
archive to the work directory and deleting it after consumption, which bounds peak extra
disk at the largest member instead of the whole 21 GB.

The first version of the reader treated those members as FASTA text and parsed compressed
bytes as sequence; the verification stage reported mismatches against binary garbage, which
is how the structure was found.

## Equivalence of the fast ORF search (proven, not assumed)

The official `find_longest_cds` restarts a scan from every `ATG` and is quadratic; over
~36M sequences that is weeks of single-core Python. The replacement is a frame-wise linear
scan. A silent difference would move every CDS boundary and change the entire corpus, so
equivalence is verified at three levels:

| test | scale | result |
|---|---|---|
| adversarial cases (immediate stop, no stop, overlapping ATGs, equal-length ties resolved by the official strict `>`, ATG inside a CDS, partial trailing triplet, non-frame stop) | 17 cases | identical |
| random sequences over ATG/stop-skewed alphabets | 4,000 | identical |
| all sequences in the official repository's sample FASTA | 22,671 | identical |
| **randomly sampled sequences from the real downloaded corpus** | **20,000** | **ORF mismatches 0, token mismatches 0** |

The last row is the one that matters for the paper: it is the actual corpus, and it was
produced by the pipeline's own `--verify` stage before a single output line was written.

## Region labels are a sidecar, not something to recover later

The official pre-training script brackets the CDS with `[`/`]` and then drops the markers,
so `pre.txt` alone cannot tell which tokens are 5'UTR, CDS or 3'UTR — and the region
objective needs exactly that. `prep_pretrain.py` therefore writes `pre_regions.txt` with
one region id per token on lines aligned to `pre.txt`. Sequences with no in-frame ORF are
labelled "no region" and contribute only the global objective, matching the official
behaviour of leaving them unmarked.

## Download: the throttling workaround

`zenodo.org` does not resolve on this cluster (DNS returns `::`), so the fetcher connects
to hard-coded Zenodo IPs. A single connection is throttled to roughly 15-20 KB/s while the
aggregate scales with connection count, which is why the fetcher uses byte ranges. The
fan-out is not free though: at 168 workers the last 15 chunks stalled and single-chunk
fetches to three different Zenodo IPs all timed out, i.e. the aggressive configuration had
provoked per-client throttling. Dropping to one connection per missing chunk recovered
12 of 14 in six minutes and the remainder shortly after. Both the working configuration
and the failure are recorded here because "just add more connections" is the wrong lesson.

## One accepted inefficiency

The `--verify` stage and the processing run each spill `mRNAdataset_0.zip` (4.6 GB) from
the outer archive, so it is extracted twice. Extracting once and keeping it would save
about five minutes at the cost of 4.6 GB of resident disk and an extra invariant to
maintain; the duplication is deliberate.