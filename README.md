# RNA-JEPA

Region-aware factorised latent prediction for mRNA language models.

A masked-language-model (MLM) objective learns to fill in tokens; it is not asked to
represent a whole 5'UTR, CDS or 3'UTR as a unit. RNA-JEPA adds a joint-embedding
predictive objective on top of the mRNABERT backbone: a student sees a masked sequence,
a momentum teacher sees the full sequence, and a predictor must reconstruct the teacher's
*region summaries* and/or its *hidden states at masked positions* — with the objective
explicitly factorised by mRNA region. The head-to-head comparison against the released
mRNABERT weights uses one code path, one data split and one hyper-parameter protocol.

## Status

| Stage | State |
|---|---|
| Environment, cluster plumbing, GPU monitor, ledger | done |
| Official weights smoke-load + tokenisation equivalence tests | done (see below) |
| Downstream data (Zenodo 17786045) + manifest audit | done (145 tasks, 0 row-count mismatches vs the protocol spreadsheet) |
| Pre-training corpus (Zenodo 12516160, 20 GB) | downloading / preprocessing |
| Baseline gate R3 | CDS mRFP **pass** (0.8663 vs 0.89); 3'UTR-RBP under investigation; ultra-long TE running |
| RNA-JEPA model + pre-training loop | implemented, GPU integration tests pass |
| Pre-training V1/V2, head-to-head, ablations | pending |

## The two mistakes this repository is built to prevent

1. **Un-spaced input destroys the tokenisation.** The released vocabulary is 74 WordPiece
   entries with **no `##` pieces**, so a raw sequence collapses to `[CLS] [UNK] [SEP]`.
   Feeding raw sequences makes every example identical and any correlation ~0. All entry
   points go through `rnajepa.tokenization.split_sequence`, which is verified byte-identical
   to the official splitter on all 1459 mRFP rows.
2. **The Triton attention kernel does not compile** against current Triton. The official
   README handles this with `pip uninstall triton` (selecting the PyTorch attention path);
   we achieve the same configuration with a stubbed `flash_attn_triton.py` in a *copy* of
   the weights, leaving the pristine directory and its md5 untouched.

## Layout

```
src/rnajepa/       tokenization.py  metrics.py  finetune.py  model.py  pretrain.py
data/              build_manifest.py  prep_pretrain.py
scripts/           fetch_zenodo_parallel.sh  fetch_all_zenodo.sh  gpu_util.sh
                   submit_finetune.sh (GPU pick + OOM retry + ledger)
                   monitor.py  install_cron.sh  sync_to_cluster.sh
tests/             test_tokenization.py  test_model_gpu.py  test_orf_equivalence.py
configs/           tasks.yaml (frozen task registry)
records/           experiment conclusions and investigations
third_party/       mRNABERT (official code, read-only reference)
```

Code lives in the `/home` project; all large artefacts (raw data, prepared corpora,
checkpoints, evaluation output) live under `/mnt/cunyuliu/rna-jepa`. Nothing large is
committed here.

## Environment

```bash
conda create -n mrnabert python=3.8 -y
conda activate mrnabert
pip install torch==2.1.2                       # then, per the official README:
pip uninstall -y triton                        #   use the PyTorch attention path
pip install "transformers==4.32.0" "accelerate==0.24.1" \
            datasets evaluate einops numpy pandas peft scikit_learn scipy tqdm sentencepiece
```

`einops` is required by the released custom model code. `flash_attn_triton.py` must exist
next to `bert_layers.py` for `trust_remote_code` loading, but its kernel is not used.

## Data

| What | Where | Notes |
|---|---|---|
| mRNABERT weights | `YYLY66/mRNABERT` | 86.6M, custom `bert_layers` code, 74-token vocab |
| Pre-training corpus | Zenodo 12516160, `mRNAdataset.zip`, 20.02 GB | md5 `bf8bc5c946a0bd3b07716b1c7f785d54` |
| Downstream tasks | Zenodo 17786045, 7 archives, 248 MB | official train/dev/test CSVs, **already whitespace-tokenised** |

`scripts/fetch_zenodo_parallel.sh` exists because `zenodo.org` does not resolve on this
cluster (DNS returns `::`). Connecting to a hard-coded Zenodo IP works, and although a
single connection is throttled to ~15-20 KB/s, throughput scales roughly linearly with the
number of concurrent connections (measured ~4 MB/s at 168 workers). Files are fetched as
byte ranges, reassembled in order and verified against the published md5.

## Pre-processing the corpus

```bash
python data/prep_pretrain.py --zip <mRNAdataset.zip> \
       --out <pre.txt> --out_regions <pre_regions.txt> --workers 64
```

Writes the official `pre.txt` (UTR as single tokens, CDS as codon tokens) **plus**
`pre_regions.txt`, one region id per token on aligned lines. The sidecar is not optional
for us: the official script drops the `[`/`]` CDS markers, so region identity cannot be
recovered from `pre.txt` afterwards. Sequences with no in-frame ORF are labelled "no
region" and contribute only the global objective.

The ORF search is a frame-wise linear-time scan instead of the official per-ATG restart
(quadratic, would take weeks over 36M sequences). Equivalence is *proven*, not assumed:
`tests/test_orf_equivalence.py` compares both implementations on 17 adversarial cases,
4000 random sequences with ATG/stop-skewed alphabets, and all 22 671 real mRNA sequences
shipped with the official repository — identical ORF boundaries and identical token
streams on every one.

## Fine-tuning / evaluation

```bash
PYTHONPATH=src python -m rnajepa.finetune --task cds_mrfp --task_kind regression \
    --model_path <weights> --data_dir <task_dir> --out_dir <out> \
    --seed 42 --max_len 256 --batch_size 16 --epochs 20 --lr 1e-4 --device 0
```

Mirrors the official `regression.py`/`classification.py`: `AutoModelForSequenceClassification`
with `trust_remote_code=True`, `num_labels=1` for regression and the train-set class count
otherwise, right-padding to the longest sequence in the batch, dev-selected checkpoint,
single test evaluation. Additions that do not change the science: CUDA is mandatory (the
device is chosen before `torch` is imported and asserted after the model is created, so a
silent CPU fallback is impossible), binary tasks report positive-class F1 *and* macro F1
*and* MCC, and every run writes `result.json`, `predictions.tsv`, `history.json` and
`run_meta.json`.

To reproduce the baseline without a scheduler:

```bash
bash scripts/submit_finetune.sh --task cds_mrfp --task_kind regression \
     --model_path <weights> --data_dir <task_dir> --out_dir <out> \
     --min_free 12000 --fp16 --save_model
```

which picks the emptiest non-MIG GPU, retries on another card after an OOM, halves the
batch as a last resort (recorded as a protocol deviation), and appends a row per attempt
to `ledger.jsonl`.

## Monitoring

`scripts/install_cron.sh` installs a 10-minute cron entry running `scripts/monitor.py`,
which (a) health-checks every job the ledger still considers running — stale log, NaN/Inf
loss, traceback, CUDA OOM — into `alerts.log`, (b) appends a GPU snapshot to
`gpu_history.jsonl`, and (c) dispatches queued jobs from `queue/pending/*.json` onto any
GPU that has enough free memory. Capacity is taken from live `nvidia-smi` output; MIG
parents are excluded because they are not schedulable even though they still report an
aggregate free-memory figure.

## Tests

```bash
python tests/test_tokenization.py            # no GPU; tokeniser equivalence + root-cause guard
python tests/test_orf_equivalence.py         # no GPU; ORF/split equivalence on real sequences
PYTHONPATH=src python tests/test_model_gpu.py --model_path <weights>   # GPU
```

The GPU test asserts, among other things, that the predictor stays memory-linear at
L = 4096 (3 GiB peak), that gradients reach every learned block while the teacher stays
frozen, that the EMA anneals 0.996 -> 0.9997, that the curriculum holds `w_region = 1.0`
for the first 60% of steps and ends at 0.2, and that the orthogonality penalty decreases.

## Citation

Baseline model and data: Xiong et al., *mRNABERT: advancing mRNA sequence design with a
universal language model and comprehensive dataset*, Nature Communications 16, 10371 (2025).