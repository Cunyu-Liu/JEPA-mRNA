"""Prove the linear-time ORF search is equivalent to the official quadratic one.

``prep_pretrain.py`` replaces the official ``find_longest_cds`` (which restarts a
scan from every ATG) with a frame-wise single pass, because the official version
would take weeks over ~36M sequences.  A silent difference would move CDS
boundaries and change the entire pre-training corpus, so equivalence is proven
here rather than assumed:

* hand-written adversarial cases (immediate stop, no stop, overlapping ATGs,
  equal-length ties resolved by the official strict-`>`, ATG inside a CDS, ...);
* random sequences over a skewed alphabet (many ATGs, many stops);
* all 22,671 real mRNA sequences shipped with the official repository.

Run:  python tests/test_orf_equivalence.py [path/to/pre_input.fasta]
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "data"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from prep_pretrain import (  # noqa: E402
    _official_split, find_longest_cds_fast, find_longest_cds_official,
    mark_and_split, normalize,
)

DEFAULT_FASTA = os.path.join(_ROOT, "third_party", "mRNABERT", "data_process",
                             "pre-train", "pre_input.fasta")


def _compare(seq: str, label: str) -> None:
    a = find_longest_cds_official(seq)
    b = find_longest_cds_fast(seq)

    if (a is None) != (b is None):
        raise AssertionError(f"{label}: ORF presence differs official={a} fast={b}")
    if a is not None:
        if (a["Start Index"], a["End Index"]) != (b["Start Index"], b["End Index"]):
            raise AssertionError(
                f"{label}: official=({a['Start Index']},{a['End Index']}) "
                f"fast=({b['Start Index']},{b['End Index']})")

    line, _regions = mark_and_split(seq, b)
    if a is None:
        marked = seq
    else:
        marked = (seq[:a["Start Index"]] + "[" +
                  seq[a["Start Index"]:a["End Index"] + 1] + "]" +
                  seq[a["End Index"] + 1:])
    official = _official_split(marked)
    if line.split() != official:
        raise AssertionError(f"{label}: token stream differs\n  ours={line[:120]}\n"
                             f"  official={' '.join(official)[:120]}")


def test_adversarial_cases():
    cases = {
        "empty": "",
        "no_atg": "CCCCCCCCCCCC",
        "atg_no_stop": "AAAATGGGGCCCTTT",
        "immediate_stop": "AAATGTAAAA",                      # ATG TAA -> 6 nt
        "two_frames": "ATGAAATAAGGGATGCCCTAG",
        "atg_inside_cds": "ATGATGGGGTAA",                    # second ATG is inside
        "stop_before_atg": "TAAATGCCCTAA",
        "tie_same_length": "ATGAAATAGATGCCCTAG",
        "partial_triplet_at_end": "ATGAAAT",
        "lowercase": "atgaaatag",
        "with_unknown": "ATGNNNTAG",
        "long_no_stop": "ATG" + "AAA" * 50,
        "non_frame_stop": "ATGAATAAGGGTAA",                  # TAA at a non-frame offset
        "multi": "CCCATGCCCTAGTTTATGAAATAGGGG",
        "whole_seq_is_cds": "ATGCCCTAG",
        "nested_atg_run": "ATGATGATGAAATAG",
        "stop_at_very_end": "CCCATGCCC" + "AAA" * 3 + "TGA",
    }
    for label, seq in cases.items():
        _compare(normalize(seq), label)
    print(f"  ok  {len(cases)} adversarial cases match the official implementation")


def test_random_sequences():
    rng = random.Random(1234)
    # skewed alphabets make ATG/stop codons frequent, which is where the two
    # algorithms could plausibly diverge
    alphabets = ["ATCG", "ATGC", "AT", "ATG", "ATGC N".replace(" ", "")]
    for i in range(4000):
        alpha = alphabets[i % len(alphabets)]
        n = rng.randint(1, 400)
        seq = "".join(rng.choice(alpha) for _ in range(n))
        _compare(seq, f"random[{i}] len={n} alpha={alpha}")
    print("  ok  4000 random sequences (skewed alphabets) match")


def test_real_mrna(fasta: str, limit: int = 0):
    if not os.path.isfile(fasta):
        print(f"  SKIP real sequences: {fasta} not found")
        return
    seqs = []
    cur = []
    with open(fasta, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    if limit:
        seqs = seqs[:limit]
    for idx, seq in enumerate(seqs):
        _compare(normalize(seq), f"real[{idx}] len={len(seq)}")
    n_orf = sum(1 for s in seqs if find_longest_cds_fast(normalize(s)) is not None)
    print(f"  ok  {len(seqs)} real mRNA sequences match "
          f"({n_orf} contain an in-frame ORF, {len(seqs)-n_orf} do not)")


def main() -> int:
    fasta = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FASTA
    failures = 0
    for name, fn in (("adversarial", test_adversarial_cases),
                     ("random", test_random_sequences)):
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    try:
        test_real_mrna(fasta)
    except AssertionError as exc:
        failures += 1
        print(f"  FAIL real: {exc}")
    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print("\nORF equivalence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())