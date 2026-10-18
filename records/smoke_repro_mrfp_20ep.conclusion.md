# cds_mrfp baseline reproduction (R3 gate, task 1/3)

**Status: PASS.** test Spearman = 0.8663 vs paper 0.89 -> |delta| = 0.024 <= 0.05.

## Setup

- model: released `YYLY66/mRNABERT` (86.6M, custom `bert_layers.BertForSequenceClassification`)
- data: official `CDS/mRFP` split, 1021/219/219 (identical to the copy shipped in the official repo)
- tokenisation: `codon` mode (3 nt per token, whitespace separated) — the official pipeline
- lr 1e-4, AdamW, wd 0.01, seed 42, 20 epochs, batch 16, max_len 256, dev-selected checkpoint
- device: 1x A100-40GB, peak 4.10 GiB, wall 259 s

## Decisive diagnosis: why an earlier attempt measured Spearman ~ 0.005

The released vocabulary has 74 entries and **no `##` continuation pieces**. Feeding a raw,
un-spaced sequence therefore tokenises to almost nothing:

    "ATGGCATCAGAAGACGTCATAAAAGAATTTATGCGATTC"  ->  [CLS] [UNK] [SEP]      (97% UNK)

every example collapses to the same single-UNK input, so the model can only emit a
constant and the correlation is zero. Whitespace separation (UTR: one nucleotide per
token; CDS: one codon per token) is not a stylistic choice, it is mandatory.

Evidence: `tests/test_tokenization.py::test_unspaced_input_is_destroyed` reproduces the
collapse, and `test_matches_official_implementation` proves our splitter is byte-identical
to the official `process_finetune_data.split_sequence` on all 1459 mRFP rows.

## Notes that matter downstream

- The released checkpoint contains `cls.predictions.*` (MLM head) and **no** pooler, so
  `bert.pooler.*` and `classifier.*` are freshly initialised — exactly as the official code
  path does. Reported numbers therefore include that random-init variance, which is why the
  main table uses 5 seeds.
- The Triton attention kernel does not compile against current Triton. The official README
  instructs `pip uninstall triton`, which selects the pure-PyTorch attention path; we
  reproduce that configuration via `weights/mRNABERT_attn_fallback` (a stubbed
  `flash_attn_triton.py`) so the pristine weights directory stays untouched and its md5 is
  recorded.