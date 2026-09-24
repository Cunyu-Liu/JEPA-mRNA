# UFold baseline: adapter, numbers, and the "weights unavailable" refutation

**Verdict.** The project record "UFold weights unavailable" is **wrong**. The weights are
present and load *cleanly* into UFold's own network; UFold now produces real numbers and
probability matrices on the ref splits.

* UFold source: `/home/cunyuliu/rna_baselines_src/UFold-main/`
* Weights: `/home/cunyuliu/rna_baselines_src/UFold-main/models/ufold_train_alldata.pt` (34.6 MB)
* Adapter (new, the only code written): `eval/ss/run_ufold.py`

## Why the shipped scripts could not run

`ufold_test.py` needs `data/TS0.cPickle` and `data/ArchiveII.pickle`, but `data/` holds only
`Readme.md` and `input.txt`; it also expects `models/ufold_train.pt`, a different filename.
The adapter reuses UFold's own model/encoding/decode code and bypasses only that missing data
pipeline.

## What the adapter reuses (no guessing)

| piece | source |
| --- | --- |
| network | `from Network import U_Net` (built `U_Net(img_ch=17)`) |
| config | `ufold.config.process_config(ufold/config.json)` |
| encoding | `ufold.data_generator.{perm, get_cut_len, creatmat}` + `Dataset_Cut_concat_new.__getitem__` recipe (16 outer-product channels + 1 `creatmat` channel = 17) |
| decode/refine | `ufold.postprocess.postprocess_new(u, x, 0.01, 0.1, 100, 1.6, True, 1.5)`, threshold `> 0.5`, then `get_ct_dict_fast` argmax + `seq2dot` |

`munch` is not installed in the cluster env, so the adapter injects a 2-line in-process
`munch` shim (a dict with attribute access) to let UFold's own `process_config` run unmodified.
No file outside the adapter was touched.

## Checkpoint load report (`models/ufold_train_alldata.pt`)

```
load_state_dict(state, strict=True) -> <All keys matched successfully>
n_ckpt_keys = 156, n_model_keys = 156
missing_in_ckpt = []      unexpected_in_ckpt = []
```

It is a clean, exact 17-channel `U_Net` all-data checkpoint: **no remapping needed**.
(Contrast with the shipped `ufold_test.py`/`ufold_predict.py` filenames
`ufold_train.pt` / `ufold_train_pdbfinetune.pt`, neither of which exists here.)

## Commands actually run (on `A100`, cwd `/home/cunyuliu/rna-jepa`)

```bash
# per split (run detached):
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval TMPDIR=/mnt/cunyuliu/tmp \
  /home/cunyuliu/miniconda3/envs/lucaone/bin/python eval/ss/run_ufold.py --split <SPLIT>

# authoritative scoring (project's single metrics implementation):
PYTHONPATH=/mnt/cunyuliu/pylibs:src:eval TMPDIR=/mnt/cunyuliu/tmp \
  /home/cunyuliu/miniconda3/envs/lucaone/bin/python eval/ss/run_baselines.py \
  --split ref_bprna_ts0 --external-dbn .../ufold_ref_bprna_ts0.dbn --external-name ufold \
  --out /mnt/cunyuliu/rna-jepa/eval_decision/baselines_ufold_ref_bprna_ts0.json
```

## Results

`ref_bprna_ts0` (UFold's own "TS0"; published TS0 micro F1 is 0.6-0.7):

| row | micro_f1 | macro_f1 | INF |
| --- | --- | --- | --- |
| **ufold** | **0.6598** | **0.7221** | 0.7325 |
| vienna_mfe | 0.5056 | 0.5085 | 0.5222 |
| vienna_centroid | 0.5393 | 0.5288 | 0.5405 |
| vienna_mea | 0.5241 | 0.5217 | 0.5346 |
| nussinov_turner | 0.2126 | 0.2304 | 0.2363 |

Other splits (split F1 from the adapter == scored F1 through `run_baselines.py`):

| split | n | micro_f1 | macro_f1 | note |
| --- | --- | --- | --- | --- |
| ref_bprna_ts0 | 1291 | 0.6598 | 0.7221 | matches published TS0 range |
| ref_pdb_ts1 | 63 | 0.6455 | 0.6518 | |
| bprna_new | 5388 | 0.6106 | 0.6439 | |

Predicted count == split count for every split (1291/1291, 63/63, 5388/5388); the scorer's
hard alignment check passes.

## Artifacts (all under `/mnt/cunyuliu/rna-jepa/eval_decision/`)

* `ufold_<split>.dbn` -- scored dot-bracket, FASTA `>name / seq / dot`, split order.
* `ufold_<split>_probs.npz` -- `sequences` + `probs` object arrays; `probs[i]` is the `L x L`
  `sigmoid(network_output)` **strict upper triangle** (diagonal and below = 0), which is exactly
  the convention `src/rnajepa/distill.load_teacher_shard` reads (verified: loads through it).
* `ufold_<split>.json` -- adapter metrics + both audits.
* `ufold_<split>.raw_ufold_decode.dbn` -- UFold's *verbatim* `seq2dot` output (audit only).
* `baselines_ufold_<split>.json` -- the scorer's JSON.

## Probability-matrix audit (required check)

`probs = sigmoid(network_output)`; the pre-masking matrix was audited per split:

* values in `[0, 1]` (min 0.000, max 1.000) -- OK;
* **symmetric**: `max|P - P^T| = 0.0` across all sequences (the U-Net head is
  `transpose(d1)*d1`, symmetric by construction);
* diagonal is *not* zero in the raw matrix (up to 0.999) and is therefore **zeroed**
  in the saved matrix (`np.triu(P, k=1)`), matching ViennaTeacher;
* sum of the upper triangle relates sensibly to the number of pairs but **over-counts**:

  | split | mean sum(upper prob) | mean GT n_pairs | corr |
  | --- | --- | --- | --- |
  | ref_bprna_ts0 | 107.1 | 27.1 | 0.812 |
  | ref_pdb_ts1 | 59.3 | 21.3 | 0.761 |
  | bprna_new | 74.1 | 25.1 | 0.669 |

  i.e. UFold's raw sigmoid is **over-confident / uncalibrated** (about 4x the true pair
  count on average) while still ranking well. This is exactly what the downstream ECE/Brier
  comparison is meant to expose -- the probability *convention* (sigmoid of the logits the
  network was trained under `BCEWithLogitsLoss`) is correct; the scale is what it is.

## One real defect found in UFold's own decode

UFold's `seq2dot(get_ct_dict_fast(...))` picks a per-position `argmax` partner, which is **not
guaranteed symmetric**, so the emitted dot-bracket is **unbalanced** for 34/1291 (2.6%)
`ref_bprna_ts0` sequences (1/63 on ref_pdb_ts1). The project scorer's strict `parse_pairs`
rejects such strings. The adapter therefore keeps a decoded pair only when it is **mutual**
(`partner[j] == i`) -- UFold's own decision with the minimal repair that yields a valid
structure. Cost is negligible: 95,674 directed pairs -> 47,817 mutual pairs (each mutual pair is
two directed entries, so only ~40 stray entries drop) and TS0 micro F1 moves 0.6584 -> 0.6598.
The raw UFold string is retained as `*.raw_ufold_decode.dbn` for audit.

## Environment notes

* Ran on `CUDA_VISIBLE_DEVICES=6` (the least-loaded A100); no running job was killed or
  disturbed (only the adapter's own stale run was stopped with `pkill -f '[r]un_ufold.py'`).
* ~0.17 s/seq (CPU-postprocess dominated); TS0 = ~3.8 min, bprna_new = ~14 min.