# Contamination-aware evaluation of RNA sequence representations: pretraining exposure, homology, and limits of structure-assisted learning

**Document status:** evidence draft, not submission-ready. Numerical claims are tied to the artifacts and independent checks listed in Section 7. No new architecture accuracy claim is made.

## Abstract

Sequence-only evaluation of RNA representations can be difficult to interpret when the pretraining corpus shares sequences with downstream benchmarks. We therefore audited exact and approximate sequence exposure before evaluating structure-assisted learning. Exact matching and MMseqs2 searches at 80% identity with at least 80% bidirectional coverage were performed for 165,854 unique queries against RNAcentral 22.0, Rfam 14.9, and Ensembl releases 109/56. At least one known source yielded a qualifying match for 162,489 queries, including all 196 VL0 and all 5,388 bpRNA-new sequences. These results prohibit a clean out-of-distribution interpretation, although they do not provide a complete historical contamination denominator because the NCBI nt snapshot is not pinned. Under this contamination-aware framework, the tested dual-view JEPA did not outperform its structure-free host on model-dev macro-F1 (0.844264 versus 0.851460). BPfold inner/outer motif inputs also showed no gain over the corresponding host or shuffled control. A frozen H0 representation achieved micro-AUPRC 0.7715 on verified RNA–protein contacts, compared with 0.7216 for native RiNALMo and 0.6292 for within-chain label shuffling. However, H2 shared H0's frozen encoder, structure-assisted arms were memory-blocked, and short chains remained weak. The results support explicit exposure reporting and strict controls rather than a new accuracy or independent-generalization claim.

**Keywords:** RNA secondary structure, pretraining contamination, sequence homology, JEPA, RNA–protein interface, evaluation reliability

## 1. Introduction

RNA language models are commonly evaluated by withholding exact sequences, sequence clusters, or Rfam families from downstream training. Such splits address different notions of independence and cannot, by themselves, establish that a pretrained representation has not encountered a test sequence. Exact pretraining overlap, approximate homology, and family identity must therefore be reported separately.

This work evaluates the evidentiary basis of three proposed improvements to RNA representation learning: dual-view sequence–structure prediction, BPfold-inspired motif inputs, and transfer to RNA–protein interface prediction. The primary question was whether training-time structural information could improve single-checkpoint sequence-only inference. The analysis does not treat an ensemble as a new architecture, canonical-nested pairs as three-dimensional structures, or a downstream probe as a substitute for the primary accuracy test.

The contributions are: (i) a versioned, hash-bound audit of exact sequence and approximate-homology exposure across three public source families; (ii) an official-metadata family mapping that preserves missing and ambiguous labels; (iii) a separation of engineering correctness, model-dev evidence, and external conclusions for structure-assisted and motif models; and (iv) independently checked interface baselines with verified contact labels and explicit pretraining-exposure limits. The outcome is an evidence framework and a set of negative and qualified results, not a claim of improved accuracy.

## 2. Source identity and evaluation scope

Table 1 identifies the public sources used for exposure analysis. These sources were used only for auditing, not for additional model training. RNAcentral and Rfam exact matches were computed from versioned FASTA files. Ensembl 109 and Ensembl Bacteria 56 comprised 31,046 ncRNA files, 701,138,004 compressed bytes, and 4,040,439 records, of which 4,032,030 passed the original 16–8192 nt length filter. Independent checks covered file identity, SHA256, gzip integrity, nonempty FASTA records, and counts.

**Table 1. Public sources and roles**

| Source | Version | Role |
|---|---|---|
| RNAcentral | 22.0 | Exact and approximate-homology exposure |
| Rfam | 14.9 FASTA, `full_region`, `family` | Exposure and official family mapping |
| Ensembl | 109 and Bacteria 56 | Exact and approximate-homology exposure |
| NCBI nt | Historical snapshot not pinned | Unresolved source; no current-file substitution |

RiNALMo preprocessing was rechecked against the paper: sequences shorter than 16 or longer than 8192 nt were removed, and inputs longer than 1022 nt were randomly cropped during training. Thus, a full-length source match does not establish that every nucleotide of a long sequence appeared in every training crop. No such uncertainty is used to dismiss direct matches.

The historical xens2 ensemble remains a baseline, not a new single-model architecture. Mol-JEPA was used only as a design reference at commit `27f0065b69c73d18eec7a32ecc413b6dc3f33d90`, under CC BY-NC 4.0. The implementation did not add torch-geometric. The original ResNet2D, Turner parameters, and exact noncrossing dynamic-programming decoder were preserved in the relevant model comparisons.

## 3. Methods

### 3.1 Exact exposure and query identity

Sequences were uppercased and T was normalized to U; ambiguous nucleotide symbols were otherwise preserved. Exact overlap was computed after applying the 16–8192 nt source-length filter. Duplicate rows were distinguished from unique sequences. Query names were not assumed to be unique identifiers: after finding same-name/different-sequence conflicts, candidate sequences were checked directly against the original FASTA sources. Each source's unique and row-level counts had to agree with its initial audit before union counts were accepted.

The three exact-source unions were deduplicated by normalized full sequence rather than added. Homology searches used MMseqs2 17.b804f `easy-search`, `--search-type 3`, `--min-seq-id 0.8`, `-c 0.8`, `--cov-mode 0`, and `--max-seqs 300`. The latter was not treated as a cap on final reported target rows. Every reported hit was independently required to satisfy `alnlen >= 0.8*qlen` and `alnlen >= 0.8*tlen`. Nine RNAcentral alignments failed this target-coverage check, but no query was lost from the three-source union: 162,489 queries were detected both before and after the postfilter. A query without an accepted match is marked “not detected,” not “nonhomologous.” Searches were run with a 50 GiB memory split. RNAcentral initially failed once with an estimated 185 GiB prefilter requirement and again during competition with the Ensembl search. It was subsequently run serially with the same threshold and complete query set, preserving both failure directories.

### 3.2 Official family mapping

Rfam families were propagated only through exact full-sequence matches linking a query to a FASTA accession/coordinate region and then to `full_region.rfam_acc` and `family.rfam_acc`. Multiple families for the same sequence were marked ambiguous rather than resolved arbitrarily. FASTA regions lacking official family metadata were reported separately from sequences absent from Rfam 14.9. Rfam 14.9 does not cover later releases, and TR1c includes mixed sources without universal Rfam labels. Consequently, mapped-family overlap is a partial analysis and is not interchangeable with sequence homology.

### 3.3 Structure-assisted and motif evaluations

H0–H6 used the same initialization, data, and primary loss. H0 was the no-JEPA host; H1 used dual-view MSE; H2 added target SIGReg; H3 shuffled structure labels; H4 used a fixed random structure encoder; H5 was a parameter-matched MLP; and H6 used a global mean structure vector. Pilot selection used single-seed model-dev macro-F1, not the external test sets. Independent pair counting checked micro/macro-F1, precision, and recall. A separate native-initialization comparison was explicitly distinguished from Plan-A warm-start and from optimizer continuation.

The BPfold experiment used the official MIT `key.energy` file and its inner/outer definitions. It compared no motif, inner/outer motif input, shuffled motif, Turner=0, and energy-only input. It was not presented as a complete BPfold reproduction or as a Turner/RIPE experiment.

### 3.4 RNA–protein contact probes

Contacts were recomputed from original mmCIF polymer and atom records. An observed RNA residue was labeled positive when a heavy atom of a polypeptide chain was within 5 Å. Unobserved residues were not treated as negatives. Complete polymer/reference-sequence correspondence and modified-parent mappings were hash-checked. The final train/holdout sets contained 804/202 chains and 1,453,871/231,155 observed residues, with no PDB, 80%-cluster, or exact-sequence overlap across the split. These downstream exclusions do not remove pretraining exposure.

Six frozen-representation or composition controls were run with seeds 42, 2026, and 3407: native RiNALMo, H0, local handcrafted features, within-chain label shuffling, length, and GC content. All 18 runs stopped at an internal-validation plateau. Thresholds were chosen only using PDB- and sequence-cluster-isolated training-internal validation. Micro-AUPRC and binary macro-F1 were recomputed independently with scikit-learn; distinct score thresholds made AUPRC independent of tied-row ordering. H2 was excluded as an independent JEPA arm because its encoder was tensor-identical to H0 at the selected checkpoint. Structure-assisted arms remained blocked by the original ResNet2D's full-pair memory requirement at 7,224 nt.

## 4. Results

### 4.1 Exact and approximate-homology exposure

**Table 2. Exact overlap and detected 80% homology after within-source deduplication**

| Set | Exact union / total | Homology union / queries |
|---|---:|---:|
| TR1c | 37,870 / 42,564 | 40,283 / 42,564 |
| VL0 | 193 / 196 | 196 / 196 |
| bpRNA-new | 5,381 / 5,388 | 5,388 / 5,388 |
| TS0 | 1,253 / 1,288 | 1,281 / 1,288 |
| Rfam | 10,327 / 10,790 | 10,711 / 10,790 |
| ncRNA test | 24,912 / 25,038 | 25,012 / 25,038 |
| Interface holdout | 86 / 202 | 111 / 198 |

Across all queries, 162,489/165,854 had a qualifying match in at least one known source; 3,365 did not. The 198 interface queries are the length-filtered subset of 202 chains. The exact and homology columns answer different questions and are not expected to be identical. No-hit records cannot be treated as clean because of search heuristics, the target cap, and the unresolved nt snapshot.

A read-only reanalysis of the verified original xens2 replay on bpRNA-new gave macro-F1 0.542209. The exact-match-detected subset had macro-F1 0.542424 (n=5,381), compared with 0.377253 for seven sequences without an exact match. All 5,388 sequences had at least one qualifying homology match, leaving no homology-not-detected comparison arm. The seven-sequence subset is not a clean control and is insufficient for significance testing or a causal exposure-effect claim. An older alternative result with 385 different full records was not mixed with the P0-verified historical/replay identity.

### 4.2 Family mapping is not equivalent to pretraining independence

The independent mapping check covered 58,742 rows. Rfam evaluation contained 8,002 uniquely mapped rows, 110 ambiguous rows, 2,111 rows with FASTA matches but no family metadata, and 567 without an exact Rfam 14.9 match. Corresponding counts for bpRNA-new were 4,938, 0, 442, and 8, and for TR1c were 24,392, 35, 4,487, and 13,650. The uniquely mapped family sets contained 1,597, 477, and 1,732 families, respectively.

Rfam evaluation and TR1c shared 120 mapped families. The bpRNA-new intersection was zero only for the mapped downstream labels. This zero supports the intended downstream family split; it does not negate the exact and approximate pretraining exposure in Table 2.

### 4.3 Dual-view pilot and initialization confounding

**Table 3. Model-dev macro-F1 for the dual-view pilot**

| Arm | Macro-F1 |
|---|---:|
| H0: no JEPA | 0.851460 |
| H1: dual-view MSE | 0.851186 |
| H2: MSE + target SIGReg | 0.844264 |
| H3: shuffled structure | 0.839411 |
| H4: fixed random structure encoder | 0.848010 |
| H5: parameter-matched MLP | 0.847129 |
| H6: global mean structure vector | 0.847233 |

H2 trailed H0 by 0.0071958679 and did not exceed the shuffle, random, MLP, or global-mean controls. The preregistered decision was `STOP_PILOT_NEGATIVE_DEV_ONLY`; three-seed formal training and frozen external evaluation were not started. A small micro-F1 increase was not substituted for the primary macro-F1 decision.

Native-initialization H0/H2 values were 0.653980 and 0.658859, respectively, a difference of +0.004879. Both stopped at step 500 without convergence, and optimizer continuation was unavailable. This difference is not evidence for structural benefit. Moreover, H0 and H2 had identical encoder tensors at selected step-100 checkpoints because the encoder warm-up lasted 1,600 steps. The interface H2 representation was therefore an H0 alias, not an independent estimate of JEPA transfer.

The full-host engineering measurement, not a universal mechanism result, gave 199,494,385,321 FLOPs for forward/backward profiling and a shared-parameter gradient cosine of −0.052939 for one batch. Exact DP and Python/C decoding were outside that profiler estimate.

### 4.4 BPfold motif control

Inner/outer motif input produced macro-F1 0.8514596293, identical to P0. Shuffled motif reached 0.8515088595, exceeding inner/outer by 0.0000492302. Turner=0 gave 0.8476000946, and energy-only input equaled P0. The decision was `STOP_BPFOLD_DEV_NEGATIVE`; external testing was not started. These results reject a benefit of this specific input experiment, not the original BPfold method generally.

### 4.5 Interface probes

**Table 4. Frozen-representation interface results, mean ± SD over three seeds**

| Representation/control | micro-AUPRC | Binary macro-F1 |
|---|---:|---:|
| RiNALMo | 0.7216 ± 0.0059 | 0.6448 ± 0.0076 |
| H0 host | 0.7715 ± 0.0053 | 0.6854 ± 0.0041 |
| Handcrafted local sequence features | 0.5438 ± 0.0070 | 0.4827 ± 0.0099 |
| Within-chain label shuffle | 0.6292 ± 0.0035 | 0.5429 ± 0.0025 |
| Length | 0.5522 ± 0.0132 | 0.3996 ± 0.0905 |
| GC content | 0.5524 ± 0.0465 | 0.3466 ± 0.0243 |

All 18 runs reached internal-validation plateaus, and independent metric differences were at most 2.22×10⁻16. H0 outperformed the other sequence-only controls on micro-AUPRC, but the within-chain shuffle reached 0.6292, indicating information in inter-chain positive-rate or composition differences. H0 AUPRC was 0.6233 on 100–499 nt chains (n=23) versus 0.7725 on chains at least 1,500 nt long (n=88). These results do not establish complex-environment understanding, independent JEPA transfer, or improved secondary-structure prediction. Complete F2 did not pass because the structure-assisted arm was not computable with the original full-pair ResNet2D.

## 5. Discussion

The audits demonstrate why downstream splitting and pretraining exposure must be distinguished. Exact overlap is already sufficient to reject a clean-OOD statement, while approximate-homology analysis exposes additional matches that exact filtering cannot address. The family-level zero for bpRNA-new and the 120 shared families for Rfam show that label design and sequence exposure provide different evidence.

The model results also separate engineering success from scientific support. The dual-view implementation passed mask, gradient, deployment, and CUDA tests, yet H2 did not outperform its host or controls. The motif implementation used verified official energy assets, yet produced no improvement over P0 or shuffling. H0's interface result is a qualified representation result limited by composition effects, length strata, direct pretraining exposure, and missing structure-assisted comparisons. None of these outcomes justifies reviving stopped augmentation, RIPE, local Pair-JEPA, ensemble voting, or a 19,000 nt beam to create a success narrative.

## 6. Threats to validity

- **Incomplete historical corpus.** The NCBI nt URL has no historical release, date, or content hash. The official RiNALMo repository's `remote_data.json` points to live resources; commit timestamps cannot identify the training snapshot. Current nt was not downloaded as a substitute. Complete historical exposure denominators remain unavailable.
- **Search and coverage limits.** MMseqs2 is heuristic, and `--max-seqs 300` is not a final output cap. `cov-mode=0` documentation alone was not accepted as proof of bidirectional coverage; independent postfiltering was applied. Short or low-complexity exact-positive sequences may not be recovered by MMseqs2, so exact and homology evidence are reported separately. No-hit is not proof of absence. Queries outside the 16–8192 nt filter are reported rather than counted as negatives.
- **Family-label coverage.** Rfam 14.9 mapping covers only exact, uniquely mapped sequences. Ambiguity, missing metadata, later releases, and mixed TR1c sources prevent a complete family-independence claim.
- **Protocol limits.** The dual-view pilot used one model-dev seed and a warm-start recipe. Native-initialization arms were not converged, and optimizer continuation was blocked. H2's long encoder warm-up prevents an independent interface JEPA estimate.
- **Computational limits.** The original full-pair ResNet2D ran out of memory at 7,224 nt. Encoder-only long-sequence execution does not establish long-sequence accuracy or complete structure-feature support.
- **Label scope.** Canonical-nested labels do not model complete three-dimensional structures or complex environments. TR1c mixes sources and is not an experimental-quality hierarchy. RNA–protein contacts are not RNA–RNA pairs, and unobserved residues were not used as negative labels.
- **Downstream confounding.** Inter-chain composition can support nontrivial label-shuffle performance. Interface holdout had 86/202 exact matches with RNAcentral, so downstream PDB/cluster exclusion alone cannot establish unseen-complex generalization.
- **Calibration and pair dependence.** Pair-level ECE is not used as evidence of calibrated structure posteriors, and no structure-level posterior calibration is offered. Correlated pairs limit the interpretation of marginal calibration; binary interface F1 and AUPRC do not address that limitation.

## 7. Reproducibility and evidence availability

The main code workspace is `/home/rna_jev_xens2_dualview_round56`. Large evidence files are under `/teams/HoulabData_1788502323/liucunyu/rinalmo_pretrain_audit` and `/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2`. The contract's Section 13 contains full commands and the rule that previous result directories must be archived rather than overwritten.

Key artifacts include `exact_overlap.json` (SHA256 `6cd88fc5…5877`), `exact_union_summary.json` (`d4412393…baa4`), `homology_union_summary.json` (`69a0b13b…7131`), `rfam14_9_exact_sequence_family_mapping_summary.json` (`bdef1812…8a99`), `converged_baseline_summary.json` (`2b465cd3…4870`), `nt_snapshot_assessment.json` (`5f4282bd…6232`), `xens2_exposure_stratified.json` (`c1c39477…07ec`), and `historical_bidirectional_coverage.json` (`8688a3fa…62e5`). The final evidence presentation contains 29 slides and has SHA256 `bc64bc3a8864aa90d638fd5ef4bfec23d377d227b798bebfa548cd7dc1c83d23`.

Relevant source revisions include `181b4d0` (tied-score AUPRC correction), `77018f6` (H0/H2 encoder identity), `33805b9` (bounded-memory homology processing), `610ee8e` (family mapping), `9f586b6` (homology-union evidence page), `96aa207` (nt provenance limitation), `e6dbffd` (frozen-prediction exposure stratification), and `8e00db0` (independent bidirectional-coverage review). Earlier revisions are retained for model and engineering provenance. No credentials or `.env` files are part of the artifacts.

## 8. Conclusion

Exact and approximate-homology analyses found extensive overlap between the evaluated sequences and three public sources associated with RiNALMo pretraining. The tested structure-assisted and motif variants did not improve model-dev secondary-structure macro-F1, while the H0 interface representation showed qualified transfer relative to sequence and composition controls. These findings support contamination-aware evaluation, explicit initialization and label audits, and clear separation of engineering and scientific claims. They do not support a new architecture accuracy breakthrough or an independent-generalization claim. A future test of the original structure-assisted hypothesis requires a distinct, converged protocol and a defensible account of historical pretraining exposure, not test-set-driven adjustment.

## Appendix A. Scope-specific responses to anticipated objections

This appendix records landing points for the project's standing reviewer questions. “Not evaluated” is a limitation, not evidence that the corresponding claim holds.

| # | Landing point and scope-specific answer |
|---|---|
| **Q1** | Section 1 and Section 4.3. This is not a new CONTRAfold or calibrated decision-head claim. The historical model lineage is retained, and no System-1 calibration or matched-FLOPs advantage is claimed. The dual-view comparison is limited to H0–H6 on model-dev; no head-to-head evidence for a new decision paradigm is offered. |
| **Q2** | Sections 3.3 and 6. No exact-marginal calibration, RLCD objective, or equality of pair probabilities with calibrated posteriors is claimed. These were not evaluated in the present v2 protocol. |
| **Q3** | Section 6 contains a dedicated calibration-scope limitation. Pair-level ECE is not used as evidence, and no structure-level posterior calibration is offered. Pair correlation and the absence of calibration evidence are explicit limitations. |
| **Q4** | Sections 3.3 and 4.4. The control is “Nussinov + Turner 堆叠能”, not ViennaRNA. Turner=0 gives 0.8476000946, and no claim is made that the baseline or motif input reproduces full ViennaRNA thermodynamics. |
| **Q5** | Sections 3.4 and 6. The memory limitation is measured rather than hidden: the original full-pair ResNet2D ran out of memory at 7,224 nt. No banded approximation, 19,000 nt beam, or full-length effectiveness claim is used. |
| **Q6** | Sections 3.3 and 4.5. RiNALMo itself is the backbone and a comparator, not a dismissed alternative. H0 and native RiNALMo are evaluated separately, but H2 is not an independent JEPA estimate, and no new backbone champion is claimed. |
| **Q7** | Sections 1 and 6. Canonical-nested pairs are the explicit scope. General pseudoknots, noncanonical interactions, and full three-dimensional or complex-environment modeling are not claimed, and no H-type accuracy result is invented for v2. |
| **Q8** | Decision ordering is reported as an ablation; we make no contribution claim about it. No new ordering experiment is presented in v2. |
| **Q9** | Sections 3.1, 4.1, 4.2, and 6. Exact, 80%-homology, and official-family analyses are reported separately. Downstream cleaning does not substitute for pretraining decontamination. No formal raw/own-clean/matched-clean accuracy comparison is claimed because the primary pilot did not pass. |
| **Q10** | Sections 2 and 3.3. Exact noncrossing DP is a preserved baseline, not a new contribution. No illegal-structure-rate or legalization-cost claim is made. |
| **Q11** | Section 4.3 and Section 7. No 50× claim, McCaskill comparison, LinearPartition comparison, or new raw-sequence speed benchmark is made. The 199,494,385,321-FLOP figure is a one-batch profiler measurement, not a latency or speedup result. |
| **Q12** | Section 6 and the project's `limitations.md` L1/L2. Jev's unreviewed status and undisclosed RLCD algorithm are not used as evidence. No RLCD reproduction or vendor performance claim is made; model claims are tied to the listed artifacts and official software sources. |

## References

[1] *RiNALMo: RNA language model*. Nature Communications, 2025. DOI: `10.1038/s41467-025-60872-5`.

[2] *Mol-JEPA: A multimodal molecular world model based on Joint Embedding Predictive Architecture*. arXiv:2608.22642v4. Official repository commit `27f0065b69c73d18eec7a32ecc413b6dc3f33d90`; CC BY-NC 4.0.

[3] *BPfold project*. Nature Communications, 2025. Official `key.energy`, MIT license; SHA256 prefix `b657c553…ce8b`.

[4] Danaee et al. *bpRNA*. Nucleic Acids Research, 2018. DOI: `10.1093/nar/gky285`.

[5] *RNAcentral 22.0*. Official sequence archive: `https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/22.0/sequences/`.

[6] *Rfam 14.9*. Official FASTA and database tables: `https://ftp.ebi.ac.uk/pub/databases/Rfam/14.9/`.

[7] *Ensembl release 109 and Ensembl Bacteria release 56*. Official FTP archives: `https://ftp.ensembl.org/pub/release-109/fasta/` and `https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-56/fasta/`.

[8] *MMseqs2*, version 17.b804f, installed from the Bioconda channel. `https://github.com/soedinglab/MMseqs2`.
