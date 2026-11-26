# UFold official recovery — 2026-11-27

## Decision

**Blocked: author-distributed weights have not been independently obtained and hashed. No prediction was generated.** The official source and the bpRNA-new5388 replay contract are recovered, but `full-ready` is false. The only accepted weight source is the Google Drive folder linked by the official UFold README; no third-party mirror, existing uncertified cache, or fabricated output is permitted.

## Authenticated identity

- Base: `c183df8457b2e3d3695b6e9eb78bb6b56bff5d91`; branch: `ufold-official-recovery-20261127`.
- Official repository: <https://github.com/uci-cbcl/UFold>.
- Pinned commit: [`75bd9acc83826059682dfca9d3659df66b132cd1`](https://github.com/uci-cbcl/UFold/commit/75bd9acc83826059682dfca9d3659df66b132cd1), authored by Laiyi Fu. GitHub API reports the commit signature `verified=true`, `reason=valid`.
- Paper linked by the authors: <https://doi.org/10.1093/nar/gkab1074>.
- Source license: MIT, copyright 2021 Computational Biology and Computational Learning @ UCI; LICENSE SHA256 `5c93b3a2d31b6e7b964633ff5e81bd9e31ff9ee845996ab3b0d3d71b0d368f2e`.
- Reproducible `git archive | gzip -n` SHA256: `a233d3577cbb213712c06f8f0827b18624e7df99ad093c6bfd70782b8e86b76b`.
- Source and future weights live under `/teams/HoulabData_1788502323/liucunyu/foldnar/ufold-official-recovery-20261127`; neither is vendored into Git.

## Weight authentication blocker

The pinned README points to <https://drive.google.com/drive/folders/1Sq7MVgFOshGPlumRE_hpNXadvhJKaryi?usp=sharing>. Git history shows this exact author link was added by commit `1f3e5fad604ec56ffb89e1bcb264034d734bda67` in 2021 and remains in the pinned commit.

On the local host, `gdown`/`urllib` failed with connection refused. On `px2`, `urllib` returned `Errno 99` and IPv4 `curl` timed out. Direct HTTPS to the related Google content endpoint also timed out. Consequently the Drive file IDs, byte sizes, hashes, and any separate weight license remain unobserved. The repository itself publishes no author checksum or alternate weight URL.

To unblock, retrieve the files through this exact author URL on a Google-Drive-capable network, preserve Drive metadata, independently compute SHA256, and strictly load `ufold_train_alldata.pt` into `Network.U_Net(img_ch=17)` on CPU. Only then may the manifest and `EXPECTED_CHECKPOINT_SHA256` be updated and a separately authorized GPU replay be considered.

## Frozen bpRNA-new5388 contract

`tools/run_ufold_official_bprna_new.py` freezes `/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl` at SHA256 `271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9`: 5,388 unique names, lengths 33–489, and 135,184 reference pairs. Input pairs are zero-based inclusive `i<j`, canonical AU/CG/GU, `j-i>3`, with one partner per base. Output is input-order JSONL containing only `name` and zero-based `pairs`, written atomically after exactly 5,388 rows and accompanied by a provenance manifest.

The replay follows `ufold_predict.py`: `ufold_train_alldata.pt`, `U_Net(img_ch=17)`, 16 ordered AUCG outer-product channels plus the official `creatmat` channel, batch size one, official `get_cut_len(length,80)`, and `postprocess_new(...,0.01,0.1,100,1.6,True,1.5)` with threshold `>0.5`. Noncanonical output is disabled. Missing or mismatched hashes fail closed before model construction or inference.

Static resources are 8,641,377 parameters and 34,593,076 bytes (32.99 MiB) for FP32 state-dict tensors. The largest padded side in this dataset is 496. A future one-sequence-at-a-time GPU job should reserve 4 GiB free CUDA memory as a conservative planning value; this is not a measured minimum, and no runtime estimate is claimed.

## Training exposure and permitted use

The official `get_args` default training list contains `RNAStralign`, `ArchiveII`, `TR0_with_data_augmentation`, `TS0`, `PDB_train`, **`bpnew`**, `TS1`, `TS2`, and `TS3`; `ufold_train.py` consumes that list. The offline predictor selects `ufold_train_alldata.pt`. Thus the intended all-data path declares direct bpnew exposure. No official per-weight training ledger exists, so the exact historical samples inside the binary are not independently proven.

Accordingly, bpRNA-new5388 is an external Test **only for protocol audit**, not model selection, checkpoint choice, tuning, ranking, or scientific superiority claims. Raw predictions remain absent until authentication and separate authorization are complete.

## Verification boundary

Only source compilation, dataset/pair checks, official feature construction, state-dict architecture inspection, and synthetic CT conversion are permitted in this task. `CUDA_VISIBLE_DEVICES` is hidden for CPU tests. No GPU inference, training, automatic evaluation, or FoldNAR publication process is started or modified.
