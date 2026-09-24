# RNAformer baseline — measured run log

Date: 2026-09-24. Env: `/home/cunyuliu/miniconda3/envs/lucaone/bin/python` (torch 2.5.1+cu121,
pandas 2.3.3, numpy 1.26.4), cwd `/home/cunyuliu/rna-jepa`, `PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval`,
`TMPDIR=/mnt/cunyuliu/tmp`, GPU `CUDA_VISIBLE_DEVICES=1` (fp32). PyPI-reachable installs only.

## What was needed to make the published code run

`evaluate_RNAformer.py` cannot be used as-is: it hard-codes `datasets/test_sets.plk` relative to its
own cwd and scores element-wise on the raw matrix. Adapter written instead:
`eval/ss/run_rnaformer.py` (imports their `RiboFormer` + `Config` + `insert_lora_layer`, reuses the
project's `_plk_frames` / `_install_pandas1_index_shim` / `_row_pairs` / `pairs_to_dotbracket`).

Minimal import graph (verified): the model path needs **torch, einops, rotary_embedding_torch**
(`axial_attention.py` imports it unconditionally) and `pkg_resources`; `loralib` for the LoRA ckpt.
**lightning / deepspeed are NOT needed** — they only appear under `RNAformer/pl_module/`, which the
model import never touches. flash-attn is absent and correctly falls back to `Attention2d`.

Installed into the shared PYTHONPATH dir `/mnt/cunyuliu/pylibs` (does not touch the conda env):
`einops==0.8.0 rotary_embedding_torch==0.5.3 beartype` (`--no-deps`).

## Commands that work

```
cd /home/cunyuliu/rna-jepa
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval TMPDIR=/mnt/cunyuliu/tmp
PY=/home/cunyuliu/miniconda3/envs/lucaone/bin/python
M=/mnt/cunyuliu/rna-jepa/refmodels/models
O=/mnt/cunyuliu/rna-jepa/eval_decision

# predict (emits dbn + prob npz + release-GT json)
$PY eval/ss/run_rnaformer.py \
  --state-dict $M/RNAformer_32M_state_dict_bprna.pth \
  --config     $M/RNAformer_32M_config_bprna.yml \
  --split ref_bprna_ts0 \
  --out-dbn    $O/rnaformer_ref_bprna_ts0.dbn \
  --out-npz    $O/rnaformer_probs_ref_bprna_ts0.npz \
  --out-release-json $O/rnaformer_release_gt_ref_bprna_ts0.json

# score with the PROJECT metric implementation (dbn must be aligned with --split)
$PY eval/ss/run_baselines.py --split ref_bprna_ts0 --baselines "" \
  --external-dbn $O/rnaformer_ref_bprna_ts0.dbn --external-name rnaformer \
  --out $O/baselines_rnaformer_ref_bprna_ts0.json
```
Long run launched detached: `setsid nohup env ... $PY eval/ss/run_rnaformer.py ... > logs/rnaformer_bprna_ts0.log 2>&1 < /dev/null &`

## Checkpoint loading (all three strict-loaded cleanly)

| checkpoint | keys | missing | unexpected | notes |
|---|---|---|---|---|
| `..._bprna.pth` | 121 | 0 | 0 | strict OK |
| `..._biophysical.pth` | 121 | 0 | 0 | strict OK |
| `..._inter_family_finetuned.pth` | 205 | 0 | 0 | strict OK **only after** LoRA insert (config `lora: true`) **and** enabling recycling (`cycling=6`) |

The inter-family config ships `cycling: false` but its state_dict carries `recycle_pair_norm.*`, so a
naive strict load fails with `Unexpected key(s): "recycle_pair_norm.weight", "recycle_pair_norm.bias"`.
The adapter auto-enables `cycling=6` when those keys are present.

## Results (bpRNA-32M checkpoint; fp32; greedy non-crossing decode at P>0.5)

micro F1 pools TP/FP/FN over the split; macro F1 = mean per-seq F1; INF = mean (all from `ss.metrics`,
the same code `evaluate_decision.py` uses). The scorer's output equals the adapter's internal
`metrics_project_gt` exactly.

| split | n | micro F1 | macro F1 | INF | GT convention |
|---|---|---|---|---|---|
| `ref_pdb_ts3` | 19 | **0.9410** | 0.9340 | 0.9354 | project GT (canonical nested) |
| `ref_pdb_ts3` | 19 | 0.7826 | 0.7520 | 0.7805 | release full pair set |
| `ref_pdb_ts2` | 39 | **0.8590** | 0.8550 | 0.8633 | project GT |
| `ref_pdb_ts2` | 39 | 0.7203 | 0.7179 | 0.7470 | release full pair set |
| `ref_bprna_ts0` | 1291 | **0.7578** | 0.7454 | 0.7512 | project GT |
| `ref_bprna_ts0` | 1291 | 0.7154 | 0.7093 | 0.7189 | release full pair set |

Other checkpoints on `ref_pdb_ts3` (project GT): biophysical 0.8265 / 0.7806 / 0.7829,
inter-family 0.8245 / 0.7731 / 0.7760 (see `eval_decision/baselines_rnaformer_*_ref_pdb_ts3.json`).

### GT convention — the two numbers are different things

The project corpora keep only canonical (AU/GC/GU) nested pairs (pseudoknots / non-canonical /
multiplets removed); the reference release keeps them all. Pairs removed (release → project):

| split | release pairs | project pairs | removed | as manifest reasons |
|---|---|---|---|---|
| `ref_pdb_ts3` | 644 | 417 | 227 (35.2%) | 111 pseudoknot, 93 non-canonical, 6 multiplet (+crossing/illegal) |
| `ref_pdb_ts2` | 904 | 655 | 249 (27.5%) | 104 pseudoknot, 124 non-canonical, 2 multiplet |
| `ref_bprna_ts0` | 40067 | 34948 | 5119 (12.8%) | 1206 pseudoknot, 4005 non-canonical |

The GT-restricted F1 is **higher** purely because the hardest pairs leave the label set — it must never
be quoted as the abstract headline without saying so.

## Published comparison

* `ref_bprna_ts0` (TS0): **F1 = 0.728**, MCC 0.733, 17.2% solved — RNAformer 32M + recycling,
  Table 1 of arXiv:2307.10073 (Franke/Runge/Hutter, ICML-W CB 2023). Primary source, verified.
  Our measured release-full F1 0.715 (nested decode) and project-GT F1 0.758 bracket it.
* `ref_pdb_ts1/ts2/ts3`: **unverified** — the 2024 biorxiv preprint (10.1101/2024.02.12.579881)
  tables could not be retrieved (fetch timed out repeatedly).

## Verification

* Sequences predicted == split size: 19/19 (`ts3`), 39/39 (`ts2`), 1291/1291 (`bprna_ts0`).
  The release `bprna_ts0` frame has **1305** rows; the project JSONL has 1291 — 14 rows rejected as
  `bad_alphabet` (`'N'`), per `ref_bprna_ts0.manifest.json`. Reader cross-check agrees (1305 vs 1291).
* `run_baselines.py` enforces dbn/split alignment (equal count + exact sequence equality) and passed.
* Probability matrices: every `.npz` verified `float64`, square `L x L`, symmetric (atol 1e-12), all
  values in [0,1], zero diagonal. Layout = `sequences` + `probs` object arrays (`load_teacher_shard`).
* Probability convention: **post-sigmoid** `sigmoid(logits[0,:,:,-1])`, symmetrised, float64. Entries
  outside the canonical mask are the model's own outputs (not zeroed); apply `valid_pair_mask(seq)`
  for a like-for-like ECE comparison against the ViennaRNA teacher shards.

## Output paths

Predictions/structures: `/mnt/cunyuliu/rna-jepa/eval_decision/rnaformer_<split>.dbn`,
`rnaformer_probs_<split>.npz`, `rnaformer_release_gt_<split>.json`.
Scored JSON: `/mnt/cunyuliu/rna-jepa/eval_decision/baselines_rnaformer_<split>.json`
(plus `_bio_`/`_interfam_` variants). Log: `/home/cunyuliu/rna-jepa/logs/rnaformer_bprna_ts0.log`.

## What did NOT work (verbatim)

1. `ModuleNotFoundError: No module named 'beartype'` (import of `rotary_embedding_torch`). Fixed by
   installing `beartype` into `/mnt/cunyuliu/pylibs`.
2. GPU 6 is a MIG `1g.5gb` slice: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate
   20.00 MiB. GPU 0 has a total capacity of 4.75 GiB of which 1.38 MiB is free.` — MIG slices are too
   small; used full GPUs 1/0.
3. Second footprint conflict: `torch.OutOfMemoryError ... GPU 0 has a total capacity of 39.49 GiB of
   which 8.06 MiB is free` for the biophysical `ts3` run while `bprna_ts0` filled GPU 1; retried on a
   freer GPU.
4. Without cycling, inter-family strict load raised
   `RuntimeError: Error(s) in loading state_dict for RiboFormer: Unexpected key(s) in state_dict:
   "recycle_pair_norm.weight", "recycle_pair_norm.bias".` — fixed by setting `cycling=6`.
5. `WebFetch` of the 2024 biorxiv preprint (`.full`, `.full.pdf`, `r.jina.ai`) timed out/empty.