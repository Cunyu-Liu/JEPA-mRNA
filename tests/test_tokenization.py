"""Unit tests for :mod:`rnajepa.tokenization`.

The critical test is :func:`test_matches_official_implementation`: our
``split_sequence`` must be byte-identical to the official mRNABERT
``process_finetune_data.split_sequence`` on the real mRFP sequences that ship
with the official repository.  If this diverges, every downstream number is
built on a different tokenisation than the baseline, which invalidates the
head-to-head comparison.

Run:  python tests/test_tokenization.py
"""

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "mRNABERT", "data_process"))

from rnajepa.tokenization import (  # noqa: E402
    REGION_3UTR, REGION_5UTR, REGION_CDS, REGION_NONE,
    audit_sequence, find_longest_cds, mark_cds, normalize_bases, split_sequence,
)

import process_finetune_data as official  # noqa: E402

MRNA_DIR = os.path.join(_ROOT, "third_party", "mRNABERT", "data_process", "fine-tune", "mRFP")


def _load_mrfp_sequences():
    seqs = []
    for split in ("train", "dev", "test"):
        path = os.path.join(MRNA_DIR, f"{split}.csv")
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))[1:]
        seqs.extend(r[0] for r in rows if r and r[0])
    return seqs


def test_matches_official_implementation():
    seqs = _load_mrfp_sequences()
    assert len(seqs) == 1459, f"expected the 1459 official mRFP rows, got {len(seqs)}"
    for option in ("codon", "utr", "complete"):
        for s in seqs:
            mine = split_sequence(s, option)[0]
            theirs = official.split_sequence(s, option)
            assert mine == theirs, (
                f"split mismatch (option={option}) for sequence {s[:60]}...\n"
                f"  mine  : {mine[:120]}\n  theirs: {theirs[:120]}"
            )
    print(f"  ok  matches official split_sequence on {len(seqs)} mRFP rows x 3 options")


def test_region_labels_complete():
    marked = "AC[ATGGCATAA]GGT"          # 5'UTR 'AC', CDS, 3'UTR 'GGT'
    text, regions = split_sequence(marked, "complete")
    toks = text.split()
    assert toks == ["A", "C", "ATG", "GCA", "TAA", "G", "G", "T"], toks
    assert regions == [REGION_5UTR, REGION_5UTR, REGION_CDS, REGION_CDS, REGION_CDS,
                       REGION_3UTR, REGION_3UTR, REGION_3UTR], regions
    print("  ok  region labels follow the official boundary rule (stop codon -> CDS)")


def test_region_labels_no_marker():
    text, regions = split_sequence("ATGGCATAA", "codon")
    assert text.split() == ["ATG", "GCA", "TAA"]
    assert regions == [REGION_NONE] * 3
    print("  ok  codon/utr options carry no region identity (by construction)")


def test_unspaced_input_is_destroyed():
    """Documented root cause: un-spaced input tokenises to UNK soup.

    Uses the shipped vocabulary via a minimal WordPiece emulation of the
    released 74-token vocab: a match is only possible for tokens present
    verbatim, and continuation pieces would need a '##' entry, of which the
    vocabulary has none.
    """
    vocab = set()
    with open(os.path.join(_ROOT, "weights_ref", "mRNABERT", "vocab.txt")) as fh:
        for line in fh:
            tok = line.strip()
            if tok:
                vocab.add(tok)
    assert not any(t.startswith("##") for t in vocab), "vocabulary unexpectedly has ## pieces"

    raw = "ATGGCATCAGAAGACGTCATAAAAGAATTTATGCGATTC"
    spaced = split_sequence(raw, "codon")[0]
    assert all(t in vocab for t in spaced.split()), "spaced tokens must all be in-vocab"

    # greedy WordPiece without '##' pieces: first token can match, the rest dies
    i, matched = 0, []
    while i < len(raw):
        for j in range(min(3, len(raw) - i), 0, -1):
            piece = raw[i:i + j] if not matched else "##" + raw[i:i + j]
            if piece in vocab:
                matched.append(piece)
                i += j
                break
        else:
            matched.append("[UNK]")
            i += 1
    unk_ratio = matched.count("[UNK]") / len(matched)
    assert unk_ratio > 0.9, f"expected mostly UNK, got {unk_ratio:.2f}"
    print(f"  ok  un-spaced input yields {unk_ratio:.0%} UNK tokens (root cause reproduced)")


def test_normalize_and_audit():
    # U -> T, whitespace stripped, everything else kept (no silent dropping:
    # dropping characters would shift the codon frame)
    assert normalize_bases("augc n-x") == "ATGCN-X"
    assert normalize_bases("ATG GC A") == "ATGGCA"
    a = audit_sequence("AUGCX")
    assert a == {"length": 5, "non_vocab": 1, "u_to_t": 1}, a
    print("  ok  normalize/audit map U->T and flag non-vocabulary characters")


def test_brackets_removed_for_single_region_options():
    for option in ("utr", "codon"):
        text, _ = split_sequence("AC[ATG]GG", option)
        assert "[" not in text and "]" not in text, text
    print("  ok  utr/codon options drop CDS markers as the official code does")


def test_find_longest_cds_and_mark():
    seq = "AAA" + "ATG" + "GCA" * 3 + "TAA" + "TTT"
    info = find_longest_cds(seq)
    assert info is not None
    assert info["CDS"] == "ATG" + "GCA" * 3 + "TAA", info["CDS"]
    assert info["Start Index"] == 3
    marked = mark_cds(seq, info)
    assert marked == "AAA[ATG" + "GCA" * 3 + "TAA]TTT", marked
    toks, regions = split_sequence(marked, "complete")
    assert toks.split()[3] == "ATG"
    assert regions[3] == REGION_CDS
    assert regions[0] == REGION_5UTR and regions[-1] == REGION_3UTR
    print("  ok  longest-ORF search + marking agrees with the official semantics")


def test_no_orf_leaves_sequence_unmarked():
    seq = "AAACCCGGGTTT"  # no ATG
    assert find_longest_cds(seq) is None
    text, regions = split_sequence(seq, "complete")
    assert text.split() == list(seq)
    assert all(r == REGION_5UTR for r in regions)
    print("  ok  ORF-less sequence stays unmarked (region objectives are skipped later)")


def test_stop_codon_belongs_to_cds():
    """Regression guard for the boundary rule that the RWKV project got wrong."""
    marked = "AC[ATGTCCTAA]GG"
    toks, regions = split_sequence(marked, "complete")
    assert toks.split() == ["A", "C", "ATG", "TCC", "TAA", "G", "G"]
    assert regions[4] == REGION_CDS, "stop codon must stay inside the CDS"
    print("  ok  stop codon is attributed to the CDS, not the 3'UTR")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nall tokenization tests passed")