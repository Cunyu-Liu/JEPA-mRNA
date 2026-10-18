"""mRNA tokenization, exactly as the official mRNABERT pipeline does it.

Why this module is load-bearing
-------------------------------
The released checkpoint ships a 74-token WordPiece vocabulary whose only
multi-character entries are the 64 codons plus the five single nucleotides
A/T/C/G/N (see ``vocab.txt``).  It contains **no ``##`` continuation tokens**.
Consequently the tokenizer only works when the input is already
whitespace-separated::

    "ATG GCA TCA ..."   ->  [CLS] ATG GCA TCA ... [SEP]      (correct)
    "ATGGCATCA..."      ->  [CLS] ATG [UNK] [UNK] ... [SEP]  (destroyed)

Feeding raw, un-spaced sequences therefore collapses every sequence to a
near-constant representation (one real token plus a run of padding UNKs),
which is why a previous attempt at reproduction measured Spearman ~ 0.005
against the paper's ~0.89.  Every entry point in this project must go through
:func:`split_sequence`.

The split options below are a faithful port of
``third_party/mRNABERT/data_process/process_finetune_data.py`` and
``.../process_pretrain_data.py``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Region ids. 0/1/2 for the three mRNA regions. REGION_NONE marks tokens that
# are not part of any region (structural tokens: [CLS]/[SEP]/[PAD]) or tasks
# whose split option does not preserve region identity (utr/codon).
REGION_5UTR = 0
REGION_CDS = 1
REGION_3UTR = 2
REGION_NONE = -1

REGION_NAMES = {REGION_5UTR: "5UTR", REGION_CDS: "CDS", REGION_3UTR: "3UTR"}

# Characters allowed by the released vocabulary.
VALID_BASES = set("ATCGN")

SPLIT_OPTIONS = ("utr", "codon", "complete")


def normalize_bases(sequence: str) -> str:
    """Upper-case, strip whitespace, and map ``U`` to ``T``.

    Characters that are not in the released vocabulary (``X``, ``-``, digits,
    ...) are **kept**, not dropped, because the official pipeline keeps them
    too: they become ``[UNK]`` tokens.  Dropping them instead would change the
    sequence length and therefore shift the codon frame, silently corrupting
    every CDS task.  ``U`` becomes ``T`` because ``vocab.txt`` is the DNA-style
    vocabulary (``T`` present, ``U`` absent).  CDS markers are preserved so
    that ``complete`` splitting can use them.
    """
    return sequence.strip().upper().replace(" ", "").replace("U", "T")


def audit_sequence(sequence: str) -> dict:
    """Report how a raw sequence maps onto the released vocabulary.

    ``non_vocab`` counts characters that will tokenise to ``[UNK]``.  Callers
    use this to decide whether a row is usable at all; nothing is dropped here.
    """
    seq = normalize_bases(sequence)
    non_vocab = sum(1 for ch in seq if ch not in VALID_BASES and ch not in "[]")
    raw = sequence.strip().upper().replace(" ", "")
    return {
        "length": len(seq),
        "non_vocab": non_vocab,
        "u_to_t": raw.count("U"),
    }


def _flush_codons(buffer: str) -> List[str]:
    """Split a CDS buffer into codon tokens (last token may be short)."""
    return [buffer[i:i + 3] for i in range(0, len(buffer), 3)]


def _split_complete(sequence: str) -> Tuple[List[str], List[int]]:
    """Mixed UTR/CDS split driven by the ``[`` / ``]`` CDS markers.

    State machine over the marked sequence:

    * before the first ``[``      -> single-character tokens, region 5'UTR
    * between ``[`` and ``]``     -> codon tokens, region CDS
    * after the matching ``]``    -> single-character tokens, region 3'UTR

    Markers are consumed, never emitted.  A missing closing ``]`` (the official
    pre-training script tolerates this when an ORF runs to the end of the
    sequence) flushes the remaining buffer as codons.
    """
    tokens: List[str] = []
    regions: List[int] = []
    in_cds = False
    seen_cds = False
    buffer = ""

    def flush_utr(region: int) -> None:
        nonlocal buffer
        for ch in buffer:
            tokens.append(ch)
            regions.append(region)
        buffer = ""

    def flush_cds() -> None:
        nonlocal buffer
        for tok in _flush_codons(buffer):
            tokens.append(tok)
            regions.append(REGION_CDS)
        buffer = ""

    for ch in sequence:
        if ch == "[":
            if in_cds:                 # malformed nesting: close the open CDS first
                flush_cds()
            else:
                flush_utr(REGION_5UTR if not seen_cds else REGION_3UTR)
            in_cds = True
            seen_cds = True
        elif ch == "]":
            if in_cds:
                flush_cds()
                in_cds = False
            # a stray ']' is ignored rather than shifting every later token
        elif in_cds:
            buffer += ch
        else:
            buffer += ch
            flush_utr(REGION_5UTR if not seen_cds else REGION_3UTR)

    if buffer:
        if in_cds:
            flush_cds()
        else:
            flush_utr(REGION_5UTR if not seen_cds else REGION_3UTR)

    return tokens, regions


def split_sequence(sequence: str, option: str) -> Tuple[str, List[int]]:
    """Return ``(space_separated_token_string, region_id_per_token)``.

    ``option``:
      * ``utr``      -- every character is one token (5'UTR / 3'UTR tasks)
      * ``codon``    -- every three characters are one token (CDS tasks)
      * ``complete`` -- mixed split driven by ``[``/``]`` CDS markers
                        (full-length mRNA tasks)

    Region labels are ``REGION_NONE`` for ``utr``/``codon`` because those split
    options discard the information needed to recover region boundaries; only
    ``complete`` (or a pre-training record with markers) knows its regions.
    """
    if option not in SPLIT_OPTIONS:
        raise ValueError(f"unknown split option {option!r}; expected one of {SPLIT_OPTIONS}")

    seq = normalize_bases(sequence)

    if option == "utr":
        chars = [c for c in seq if c not in "[]"]
        return " ".join(chars), [REGION_NONE] * len(chars)

    if option == "codon":
        chars = [c for c in seq if c not in "[]"]
        toks = _flush_codons("".join(chars))
        return " ".join(toks), [REGION_NONE] * len(toks)

    toks, regions = _split_complete(seq)
    return " ".join(toks), regions


def find_longest_cds(mrna: str, start_codon: str = "ATG",
                     stop_codons: Tuple[str, ...] = ("TAG", "TAA", "TGA")) -> Optional[dict]:
    """Pure-python longest-ORF search, ported from the official pre-training script.

    Returns ``None`` when no in-frame ORF exists, in which case the official
    pipeline leaves the sequence unmarked and we skip region objectives for it.
    """
    start_index = mrna.find(start_codon)
    best = None
    while start_index != -1:
        end_index = start_index + len(start_codon)
        while end_index < len(mrna):
            codon = mrna[end_index:end_index + 3]
            if codon in stop_codons and (end_index - start_index) % 3 == 0:
                length = end_index - start_index + 3
                if best is None or length > best["length"]:
                    best = {
                        "CDS": mrna[start_index:end_index + 3],
                        "Start Index": start_index,
                        "End Index": end_index + 2,
                        "length": length,
                    }
                break
            end_index += 1
        start_index = mrna.find(start_codon, start_index + 1)
    return best


def mark_cds(sequence: str, cds_info: Optional[dict]) -> str:
    """Insert ``[``/``]`` around the CDS, as the official script does."""
    if not cds_info:
        return sequence
    s, e = cds_info["Start Index"], cds_info["End Index"]
    return sequence[:s] + "[" + sequence[s:e + 1] + "]" + sequence[e + 1:]