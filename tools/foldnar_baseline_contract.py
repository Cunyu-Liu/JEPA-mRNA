#!/usr/bin/env python3
"""Baseline identity contract for FoldNAR Phase 0.

Rows distinguish model families, released/project checkpoints, decoding variants and
CHANRG predictor heads. Unknown provenance is explicit and is never encoded as zero.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ART = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")
EXT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
SOURCE = REPO / "source_data"
RECORDS = REPO / "records/BASELINE_RESULTS.md"
SOTA = REPO / "tables/sota_vs_ours.md"
CHANRG_URL = "https://arxiv.org/html/2603.22330"
MULTIMOLECULE_URL = "https://pypi.org/project/multimolecule/"

HISTORICAL_25 = {
    "vienna_mfe": "ViennaRNA MFE", "vienna_centroid": "ViennaRNA centroid",
    "vienna_mea": "ViennaRNA MEA", "rnastructure": "RNAstructure",
    "contrafold": "CONTRAfold", "eternafold": "EternaFold", "mxfold2": "MXfold2",
    "bpfold": "BPfold", "ufold": "UFold", "efold": "eFold", "spot_rna": "SPOT-RNA",
    "rnaformer": "RNAformer", "kirigami": "Kirigami", "contextfold": "ContextFold",
    "e2efold": "E2Efold", "debfold": "DEBFold", "rinalmo": "RiNALMo",
    "ernie_rna": "ERNIE-RNA", "rna_fm": "RNA-FM", "nucleicbert": "NucleicBERT",
    "structrfm": "structRFM", "xens2": "xens2", "best_plan_a": "best Plan-A",
    "r2d": "R2D", "h0": "H0",
}
MAIN_BOARD_13 = {
    "vienna_mfe": "vienna_mfe", "vienna_centroid": "vienna_centroid",
    "vienna_mea": "vienna_mea", "nussinov_turner": "nussinov_turner",
    "mxfold2": "MXfold2", "eternafold": "EternaFold", "ufold": "UFold",
    "rnaformer_bprna": "RNAformer (bprna ckpt)",
    "rnaformer_interfamily": "RNAformer (inter-family ckpt)",
    "rnaformer_biophysical": "RNAformer (biophysical ckpt)",
    "rinalmo_ft": "RiNALMo-ft (Zenodo ckpt)",
    "nucleicbert_frozen_project_head": "NucleicBERT (frozen enc + our head)",
    "efold_0_1_2": "eFold 0.1.2 (Sci Adv 26 wheel weights, zero-shot sequence mode)",
}
CHANRG_BACKBONES = {
    "rinalmo_giga": "RiNALMo-Giga", "aido_rna": "AIDO.RNA",
    "rinalmo_mega": "RiNALMo-Mega", "ernie_rna": "ERNIE-RNA",
    "rinalmo_micro": "RiNALMo-Micro", "rna_fm": "RNA-FM",
}
CHANRG_HEADS = (("linear", "Linear"), ("resnet", "ResNet"), ("unet", "U-Net"))
CHANRG_29 = {
    "eternafold": "EternaFold", "contrafold": "CONTRAfold",
    "contrafold_nc": "CONTRAfold-nc", "linearfold_vienna": "LinearFold (ViennaRNA)",
    "vienna_mfe": "RNAfold", "linearfold_contrafold": "LinearFold (ContraFold)",
    "rnastructure_fold": "RNAstructure", "mxfold2": "MXFold2", "bpfold": "bpFold",
    "spot_rna": "SPOT-RNA", "ufold": "UFold",
}
for _backbone, _label in CHANRG_BACKBONES.items():
    for _head, _head_label in CHANRG_HEADS:
        CHANRG_29[f"{_backbone}_{_head}"] = f"{_label}-{_head_label}"
COHORTS = {"HISTORICAL_25": HISTORICAL_25, "MAIN_BOARD_13": MAIN_BOARD_13, "CHANRG_29": CHANRG_29}

PREDICTION_MODEL_TO_BASELINE = {
    "xens2": "xens2", "rinalmo_2d_s0": "r2d", "best_plan_a": "best_plan_a", "h0": "h0",
    "bpfold": "bpfold", "ufold": "ufold", "mxfold2": "mxfold2", "rinalmo_ft": "rinalmo_ft",
    "eternafold": "eternafold", "chanrg_ens3": "chanrg_ens3", "efold_0_1_3": "efold_0_1_3",
    "contrafold": "contrafold", "nucleicbert_frozen_official": "nucleicbert_frozen_official",
    "nucleicbert_ft": "nucleicbert_ft", "rnaformer_bprna": "rnaformer_bprna",
    "rnaformer_interfamily": "rnaformer_interfamily", "rnaformer_intrafamily": "rnaformer_intrafamily",
    "rnaformer_biophysical": "rnaformer_biophysical", "kirigami": "kirigami",
    "kirigami_fast": "kirigami_fast", "spot_rna": "spot_rna", "vienna_mfe": "vienna_mfe",
    "vienna_centroid": "vienna_centroid", "vienna_mea": "vienna_mea",
    "nussinov_turner": "nussinov_turner", "rnastructure_fold": "rnastructure_fold",
    "rnastructure_maxexpect": "rnastructure_maxexpect",
}

LEGACY_DISPOSITION = {
    "xens2": {"baseline_ids": ["xens2"], "reason": "exact frozen ensemble identity retained"},
    "rinalmo_2d_s0": {"baseline_ids": ["r2d"], "reason": "R2D seed-0 component alias"},
    "best_plan_a": {"baseline_ids": ["best_plan_a"], "reason": "VL0-selected Plan-A seed 0"},
    "h0": {"baseline_ids": ["h0"], "reason": "scientific ablation retained; no frozen E0 predictions"},
    "bpfold": {"baseline_ids": ["bpfold"], "reason": "raw main/Rfam predictions retained"},
    "ufold": {"baseline_ids": ["ufold"], "reason": "converted-weight identity and clean blocked status retained"},
    "mxfold2": {"baseline_ids": ["mxfold2"], "reason": "official package predictor retained"},
    "rinalmo_ft": {"baseline_ids": ["rinalmo_ft"], "reason": "Zenodo fine-tuned checkpoint; distinct from CHANRG heads"},
    "eternafold": {"baseline_ids": ["eternafold"], "reason": "EternaFoldParams.v1 project replay retained"},
    "efold": {"baseline_ids": ["efold_0_1_3"], "reason": "old row described 0.1.3; main-board 0.1.2 is separate"},
    "spot_rna": {"baseline_ids": ["spot_rna"], "reason": "converted checkpoint; bpRNA-new raw prediction located"},
    "vienna": {"baseline_ids": ["vienna_mfe", "vienna_centroid", "vienna_mea"], "reason": "ambiguous generic row expanded to three inference identities"},
    "rnastructure": {"baseline_ids": ["rnastructure", "rnastructure_fold", "rnastructure_maxexpect"], "reason": "family plus Fold/MaxExpect predictors separated"},
    "contrafold": {"baseline_ids": ["contrafold"], "reason": "default CONTRAfold separated from CONTRAfold-nc"},
    "kirigami": {"baseline_ids": ["kirigami", "kirigami_fast"], "reason": "same weights but distinct decoding implementations"},
    "linearfold": {"baseline_ids": ["linearfold_vienna", "linearfold_contrafold"], "reason": "ambiguous parameterization expanded; no generic prediction joined"},
    "nucleicbert_frozen": {"baseline_ids": ["nucleicbert_frozen_project_head", "nucleicbert_frozen_official"], "reason": "ambiguous old row not auto-joined: project head and official SSP checkpoint differ"},
    "nucleicbert_ft": {"baseline_ids": ["nucleicbert_ft"], "reason": "fine-tuned project checkpoint retained"},
}

EXCLUSIONS = [
    {"candidate": "H2", "reason": "not one of the requested baseline cohorts and has no frozen E0 prediction; never substituted for H0"},
    {"candidate": "generic LinearFold prediction", "reason": "excluded as ambiguous; ViennaRNA and ContraFold parameterizations are separate predictors"},
    {"candidate": "generic NucleicBERT frozen prediction", "reason": "excluded as ambiguous; official released SSP and project-trained head are not interchangeable"},
    {"candidate": "ncRNA classification and RNA-protein interface systems", "reason": "outside the FoldNAR Phase 0-2 secondary-structure contract"},
]


def _specs() -> list[dict]:
    """Return row metadata; cohort membership is attached independently below."""
    rows = []
    def add(baseline_id, model, category, version, evidence, status, reason, row_type="PREDICTOR"):
        rows.append({"baseline_id": baseline_id, "model": model, "category": category,
                     "version_protocol": version, "primary_evidence": evidence, "status": status,
                     "inclusion_reason": reason, "row_type": row_type})
    # Historical 25: families and explicitly required project/system identities.
    add("vienna_mfe", "ViennaRNA MFE", "STRUCTURED_DECODER", "ViennaRNA 2.7.2; MFE", RECORDS, "LEGACY_TABLE_ONLY", "historical 25 and 13-system board; raw summary has metrics but no per-sequence list")
    add("vienna_centroid", "ViennaRNA centroid", "STRUCTURED_DECODER", "ViennaRNA 2.7.2; centroid/BPP", RECORDS, "LEGACY_TABLE_ONLY", "historical 25 and 13-system board; summary only")
    add("vienna_mea", "ViennaRNA MEA", "STRUCTURED_DECODER", "ViennaRNA 2.7.2; MEA/BPP", RECORDS, "LEGACY_TABLE_ONLY", "historical 25 and 13-system board; summary only")
    add("rnastructure", "RNAstructure", "STRUCTURED_DECODER", "family identity; version not frozen in historical contract", RECORDS, "FAMILY_PROVENANCE_ONLY", "historical model family; concrete Fold/MaxExpect predictors are separate", "MODEL_FAMILY")
    add("contrafold", "CONTRAfold", "STRUCTURED_DECODER", "official CONTRAfold default parameters; project raw replay", SOURCE / "appendix_leakage/round39_contrafold_raw_summary.json", "RAW_COMPLETE_MAIN", "historical 25 and CHANRG structured decoder")
    add("eternafold", "EternaFold", "STRUCTURED_DECODER", "EternaFoldParams.v1; serial CPU project replay", SOURCE / "appendix_leakage/round39_eternafold_raw_summary.json", "RAW_COMPLETE", "historical 25, main board and CHANRG")
    add("mxfold2", "MXfold2", "STRUCTURED_DECODER", "official package TrainSetAB.pth; python -m mxfold2 predict", SOURCE / "appendix_leakage/round39_mxfold2_raw_summary.json", "RAW_COMPLETE_CLEAN_PENDING", "historical 25, main board and CHANRG")
    add("bpfold", "BPfold", "DIRECT_NEURAL", "BPfold raw full-length sequence inference", EXT / "bpfold/raw_summary.json", "RAW_COMPLETE_MAIN_AND_RFAM", "historical 25 and CHANRG direct neural predictor")
    add("ufold", "UFold", "DIRECT_NEURAL", "recovered ufold_train_alldata.pt; MultiMolecule conversion", SOURCE / "appendix_leakage/round39_ufold_raw_summary.json", "RAW_COMPLETE_CLEAN_BLOCKED", "historical 25, main board and CHANRG; training union remains blocked")
    add("efold", "eFold", "DIRECT_NEURAL", "family identity; released versions separated below", SOTA, "FAMILY_PROVENANCE_ONLY", "historical model family; do not mix 0.1.2 and 0.1.3", "MODEL_FAMILY")
    add("spot_rna", "SPOT-RNA", "DIRECT_NEURAL", "MultiMolecule converted checkpoint; not original-author distribution", SOURCE / "appendix_leakage/round39_spotrna_raw_summary.json", "RAW_BPRNA_NEW_ONLY", "historical 25 and CHANRG direct neural predictor")
    add("rnaformer", "RNAformer", "DIRECT_NEURAL", "32M family; four checkpoints separated", SOTA, "FAMILY_PROVENANCE_ONLY", "historical model family; checkpoint-specific rows carry predictions", "MODEL_FAMILY")
    add("kirigami", "Kirigami", "DIRECT_NEURAL", "official recovered weights; standard decoder", RECORDS, "RAW_TS0_AND_BPRNA_NEW_ONLY", "historical 25; two raw splits located")
    add("contextfold", "ContextFold", "STRUCTURED_DECODER", "1.00; official readme/resources recovered; implementation and trained parameters blocked", SOURCE / "appendix_leakage/round39_contextfold_asset_recheck.json", "BLOCKED_OFFICIAL_IMPLEMENTATION_AND_PARAMETERS", "required historical 25 coverage; official protocol evidence recovered but no executable/model/prediction")
    add("e2efold", "E2Efold", "DIRECT_NEURAL", "repository commit 2aab9e78326ddaa27d14c49eb34001c0003a1960; official weights not recovered", SOURCE / "fig05_speed/round38_baseline_assets.json", "BLOCKED_PROVENANCE", "required historical 25 coverage; old bpRNA-new note is not a reproduced result")
    add("debfold", "DEBFold", "DIRECT_NEURAL", "BLOCKED_PROVENANCE", RECORDS, "BLOCKED_PROVENANCE", "required historical 25 coverage; no verified current checkpoint/prediction")
    add("rinalmo", "RiNALMo", "FOUNDATION_MODEL", "family identity; micro/mega/giga and task heads separated", RECORDS, "FAMILY_PROVENANCE_ONLY", "historical foundation-model family", "MODEL_FAMILY")
    add("ernie_rna", "ERNIE-RNA", "FOUNDATION_MODEL", "family identity; CHANRG heads separated", CHANRG_URL, "FAMILY_PROVENANCE_ONLY", "historical foundation-model family", "MODEL_FAMILY")
    add("rna_fm", "RNA-FM", "FOUNDATION_MODEL", "family identity; CHANRG heads separated", CHANRG_URL, "FAMILY_PROVENANCE_ONLY", "historical foundation-model family", "MODEL_FAMILY")
    add("nucleicbert", "NucleicBERT", "FOUNDATION_MODEL", "family identity; official SSP/project head/fine-tuned rows separated", SOTA, "FAMILY_PROVENANCE_ONLY", "historical foundation-model family", "MODEL_FAMILY")
    add("structrfm", "structRFM", "FOUNDATION_MODEL", "BLOCKED_PROVENANCE", RECORDS, "BLOCKED_PROVENANCE", "required historical 25 coverage; no verified current checkpoint/prediction")
    add("xens2", "xens2", "PROJECT", "Plan-A s0/s1 + RiNALMo-2D s0; w_plana=0.7; plana_s1_weight=0.5", ART / "artifacts/p0/xens2_bprna_new_replay/run_manifest.json", "COMPLETE_RAW", "frozen project ensemble identity")
    add("best_plan_a", "best Plan-A", "PROJECT", "VL0-selected Plan-A seed 0; raw-complete E0 predictions blocked", EXT.parent / "round39_xens2/qa/xens2_plana_distill_s0_vl0/qa_summary.json", "LIMITED_CLEAN_SUBSET_RAW_E0_BLOCKED", "required single-model project comparator")
    add("r2d", "R2D / RiNALMo-2D seed 0", "PROJECT", "frozen RiNALMo-giga embeddings + project ResNet2D head; step 20000", ART / "manifests/p0/xens2_r2d_identity.json", "ARCHIVEII_FULL_BLOCKED", "required project component; distinct from RiNALMo-ft and CHANRG heads")
    add("h0", "H0", "PROJECT", "H0 scientific ablation; same data/decoding required", RECORDS, "BLOCKED_NO_FROZEN_E0_PREDICTIONS", "required project ablation; no frozen E0 prediction")
    # Main-board predictor identities not already present above.
    add("nussinov_turner", "Nussinov+Turner", "STRUCTURED_DECODER", "project MLP_T=0; max-product decode", SOURCE / "appendix_leakage/round39_vienna_raw_summary.json", "LEGACY_TABLE_ONLY", "13-system board internal physical reference; summary only")
    for baseline_id, label, digest in (
        ("rnaformer_bprna", "RNAformer bpRNA checkpoint", "b2775613e9addafe30d17f7152dc985e8381826842d5d5056030006adfad6769"),
        ("rnaformer_interfamily", "RNAformer inter-family checkpoint", "bd99f8d72084af52a368a527e6394626a98e6dd132e8f7b37d6bd1b258c34a0b"),
        ("rnaformer_biophysical", "RNAformer biophysical checkpoint", "bd3c90c97893bc47597305f20400adf87a4a352907598d12556000a9a6b6cc8f"),
    ):
        evidence = SOURCE / f"appendix_leakage/round39_{baseline_id.replace('_interfamily', '_interfam').replace('_biophysical', '_biophysical_raw_clean').replace('_bprna', '_bprna')}_raw_summary.json"
        if baseline_id == "rnaformer_biophysical":
            evidence = SOURCE / "appendix_leakage/round39_rnaformer_biophysical_raw_clean_summary.json"
        add(baseline_id, label, "DIRECT_NEURAL", f"official RNAformer 32M checkpoint; weights_sha256={digest}; raw B=1 full-length", evidence, "RAW_COMPLETE_MAIN", "13-system main-board checkpoint identity")
    add("rinalmo_ft", "RiNALMo-ft", "FOUNDATION_MODEL", "Zenodo giga_ss_bprna_ft.pt; strict-pair project re-score", SOURCE / "appendix_leakage/round39_rinalmo_ft_raw_summary.json", "RAW_COMPLETE_CLEAN_PENDING", "13-system board; not interchangeable with CHANRG U-Net")
    add("nucleicbert_frozen_project_head", "NucleicBERT frozen encoder + project head", "FOUNDATION_MODEL", "official frozen MLM encoder; project SecStruct2DPredictionHead trained on bprna_tr1c", SOTA, "LEGACY_TABLE_ONLY", "13-system board identity; not the released official SSP checkpoint")
    add("efold_0_1_2", "eFold 0.1.2", "DIRECT_NEURAL", "0.1.2 wheel + bundled efold_weights.pt; zero-shot sequence; native postprocess_new_nc", SOTA, "LEGACY_TABLE_ONLY", "13-system board; distinct weight hash/protocol from 0.1.3")
    # CHANRG predictors not represented by a project/historical row.
    add("contrafold_nc", "CONTRAfold-nc", "STRUCTURED_DECODER", "CHANRG noncanonical-capable variant; exact version/parameters BLOCKED_PROVENANCE", CHANRG_URL, "LEGACY_TABLE_ONLY", "CHANRG 29; exact canonical name read from main Figure 2")
    add("linearfold_vienna", "LinearFold (ViennaRNA)", "STRUCTURED_DECODER", "CHANRG predictor; exact version BLOCKED_PROVENANCE", CHANRG_URL, "LEGACY_TABLE_ONLY", "CHANRG 29; ViennaRNA parameterization")
    add("linearfold_contrafold", "LinearFold (ContraFold)", "STRUCTURED_DECODER", "CHANRG predictor; exact version BLOCKED_PROVENANCE", CHANRG_URL, "LEGACY_TABLE_ONLY", "CHANRG 29; ContraFold parameterization")
    add("rnastructure_fold", "RNAstructure Fold", "STRUCTURED_DECODER", "project raw Fold binary; CHANRG version not assumed identical", SOURCE / "appendix_leakage/round39_rnastructure_raw_summary.json", "RAW_COMPLETE_MAIN", "CHANRG RNAstructure maps at method level to Fold; project raw predictor retained")
    for backbone, label in CHANRG_BACKBONES.items():
        for head, head_label in CHANRG_HEADS:
            baseline_id = f"{backbone}_{head}"
            add(baseline_id, f"{label}-{head_label}", "FOUNDATION_MODEL",
                f"CHANRG MultiMolecule {head_label} contact head; checkpoint/version/training config BLOCKED_PROVENANCE",
                CHANRG_URL, "LEGACY_TABLE_ONLY", "CHANRG 29 reconstructed from 6 backbones x 3 registered contact heads")
    # Contract additions and supplemental measured/internal identities.
    add("linearpartition", "LinearPartition-class method", "STRUCTURED_DECODER", "long-sequence contract candidate; exact variant/version BLOCKED_PROVENANCE", RECORDS, "BLOCKED_PROVENANCE", "required long-sequence method class; not silently merged with LinearFold")
    add("eco_rna", "Eco-RNA", "FOUNDATION_MODEL", "official weights and inference code unpublished at contract freeze", "https://arxiv.org/html/2610.13106", "BLOCKED_OFFICIAL_ASSETS_UNPUBLISHED", "contract requires deferred reproduction, not paper-number substitution")
    add("efold_0_1_3", "eFold 0.1.3", "DIRECT_NEURAL", "0.1.3; commit 4e146fe92c574104b2518d7350e0fa4a39f32030; raw zero-shot sequence", SOURCE / "fig05_speed/round38_baseline_assets.json", "RAW_COMPLETE_MAIN", "legacy 18-row registry identity and eight newly registered raw predictions")
    add("nucleicbert_frozen_official", "NucleicBERT official SSP", "FOUNDATION_MODEL", "official secstr.ckpt epoch99/global-step258000; B=2 first-598-nt protocol", SOURCE / "appendix_leakage/round39_nucleicbert_frozen_raw_summary.json", "RAW_COMPLETE_MAIN", "eight raw predictions located; distinct from main-board project head")
    add("nucleicbert_ft", "NucleicBERT fine-tuned", "FOUNDATION_MODEL", "project nb_finetuned.pt; full fine-tuning protocol", RECORDS, "RAW_COMPLETE_MAIN", "legacy registry identity and eight raw predictions")
    add("rnaformer_intrafamily", "RNAformer intra-family checkpoint", "DIRECT_NEURAL", "official RNAformer 32M intra-family; weights_sha256=d357c43e991cb44617e2f2733cbb090abfcaf56f63ef7b2142013232e07bf9c3", SOURCE / "appendix_leakage/round39_rnaformer_intrafam_raw_summary.json", "RAW_COMPLETE_MAIN", "supplemental released checkpoint; not substituted for biophysical")
    add("kirigami_fast", "Kirigami fast decoder", "DIRECT_NEURAL", "same Kirigami weights as standard row; fast decoder is a separate inference identity", RECORDS, "RAW_COMPLETE_MAIN", "eight raw predictions located; decoder identity cannot be merged with standard Kirigami")
    add("rnastructure_maxexpect", "RNAstructure MaxExpect", "STRUCTURED_DECODER", "project raw MaxExpect binary; distinct from Fold", SOURCE / "appendix_leakage/round39_rnastructure_raw_summary.json", "RAW_COMPLETE_MAIN", "supplemental decoding variant with eight raw predictions")
    add("chanrg_ens3", "CHANRG internal three-member ensemble", "PROJECT", "project internal CHANRG ensemble; not a CHANRG published predictor", RECORDS, "FROZEN_PROJECT_INTERNAL", "required for prediction-registry bidirectional trace; excluded from published 29 count")
    return rows


def _public_evidence(url: str) -> dict:
    return {"source_type": "public_url", "url": url, "path": None, "exists": True,
            "retrieval_status": "RETRIEVED_DURING_PHASE0_REMEDIATION"}


def _source_memberships(baseline_id: str) -> list[dict]:
    return [{"cohort": cohort, "cohort_label": roster[baseline_id]}
            for cohort, roster in COHORTS.items() if baseline_id in roster]


def _default_training(baseline_id: str) -> dict:
    if baseline_id in {"vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner",
                       "rnastructure", "rnastructure_fold", "rnastructure_maxexpect",
                       "linearfold_vienna", "linearpartition"}:
        return {"status": "NOT_APPLICABLE_NOT_SEQUENCE_FITTED", "project_training_dataset_ids": [],
                "external_training_corpora": [], "reason": "thermodynamic/prior parameter inference; no learned sequence-fit training set"}
    if baseline_id in {"xens2", "best_plan_a", "r2d", "h0", "nucleicbert_frozen_project_head", "nucleicbert_ft"}:
        return {"status": "PROJECT_TR1c_PROTOCOL", "project_training_dataset_ids": ["bprna_tr1c"],
                "development_dataset_ids": ["bprna_vl0"], "external_training_corpora": [],
                "clean_scope": "project-specific; never applied to external baselines"}
    if baseline_id in CHANRG_29 and baseline_id.rsplit("_", 1)[-1] in {"linear", "resnet", "unet"}:
        return {"status": "PUBLISHED_CHANRG_PROTOCOL_CHECKPOINT_BLOCKED",
                "project_training_dataset_ids": ["chanrg_train_train"],
                "development_dataset_ids": ["chanrg_train_validation"],
                "current_file_coverage": "FROZEN_SUBSET_OR_VERSION_MISMATCH; full official counts not available",
                "checkpoint_config": "BLOCKED_PROVENANCE"}
    external = {
        "contextfold": (["Andronescu RNA 2010 S-Full-Train for paper prediction models; S-Full initialization for family-specialized models"], "OFFICIAL_README_RESOURCES_RECOVERED; exact released Train defaults and implementation/model bytes blocked"),
        "debfold": (["DEBFold official TrainingValidationSet.zip: 8782 structures/8384 unique sequences; SHA256=20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2"], "OFFICIAL_TRAIN_VALIDATION_ARCHIVE_RECOVERED; 25-fold assignments and checkpoint-specific training union not published"),
        "mxfold2": (["bpRNA-1m TrainSetA+TrainSetB"], "OFFICIAL_PACKAGE_TRAINING_IDENTITY; homology overlap unresolved"),
        "rinalmo_ft": (["bpRNA TR0"], "ZENODO_CHECKPOINT_TRAINING_CORPUS; TestSetB overlap known"),
        "ufold": ([], "BLOCKED_PROVENANCE: paper/code-default/historical audit do not define one identical training union"),
        "efold_0_1_2": (["bpRNA subset", "RNAstralign", "Ribonanza"], "PUBLISHED_COMPOSITION_PARTIAL; ArchiveII reported held out"),
        "efold_0_1_3": (["bpRNA subset", "RNAstralign", "Ribonanza"], "PUBLISHED_COMPOSITION_PARTIAL; 0.1.3 weight-specific audit not complete"),
        "nucleicbert_frozen_official": (["RNAstralign TR0 (from raw artifact training_corpus)"], "RAW_ARTIFACT_RECORD; full pretraining union not inferred"),
        "rnaformer_bprna": (["official bprna_data.plk"], "OFFICIAL_DATASET_IDENTITY; exact clean overlap outside current baseline audit"),
        "rnaformer_interfamily": (["official inter-family training data"], "OFFICIAL_DATASET_IDENTITY; exact clean overlap outside current baseline audit"),
        "rnaformer_intrafamily": (["official intra-family training data"], "OFFICIAL_DATASET_IDENTITY; exact clean overlap outside current baseline audit"),
        "rnaformer_biophysical": (["official biophysical_model_data.plk train=410408"], "OFFICIAL_DATASET_IDENTITY; Rfam synthetic RNAfold-distilled experiment, not PDB experimental labels"),
    }
    datasets, status = external.get(baseline_id, ([], "BLOCKED_PROVENANCE: exact training union not verified"))
    result = {"status": status, "project_training_dataset_ids": [], "external_training_corpora": datasets}
    if baseline_id == "debfold":
        result.update(source_commit="ff693cdf4cb177e50d8a6a15c7a6292af802b75e",
                      archive_sha256="20d4404ffa2c1d2b973f230de859bbfadd47cf3de46c3164e7b8c550b891c6a2")
    return result


def build_baseline_rows(predictions: list[dict], file_info: Callable) -> list[dict]:
    by_baseline = {}
    for pred in predictions:
        baseline_id = PREDICTION_MODEL_TO_BASELINE.get(pred["model"])
        if baseline_id:
            by_baseline.setdefault(baseline_id, []).append(pred)
    rows = []
    for spec in _specs():
        baseline_id = spec["baseline_id"]
        preds = by_baseline.get(baseline_id, [])
        frozen = [p for p in preds if p.get("status") == "FROZEN"]
        summaries = [p for p in preds if p.get("status") == "SUMMARY_ONLY"]
        blocked = [p for p in preds if p.get("status") == "BLOCKED"]
        memberships = _source_memberships(baseline_id)
        evidence_value = spec["primary_evidence"]
        evidence = _public_evidence(evidence_value) if isinstance(evidence_value, str) and evidence_value.startswith("http") else {
            "source_type": "local_file", **file_info(Path(evidence_value))}
        additional = []
        if baseline_id in CHANRG_29:
            additional.append(_public_evidence(MULTIMOLECULE_URL))
        metadata = next((p.get("model_metadata", {}) for p in frozen if p.get("model_metadata")), {})
        weights_path = metadata.get("weights") or metadata.get("checkpoint")
        if isinstance(weights_path, dict):
            weights_path = None
        weights_sha = metadata.get("weights_sha256")
        if baseline_id in {"vienna_mfe", "vienna_centroid", "vienna_mea", "nussinov_turner",
                           "rnastructure", "rnastructure_fold", "rnastructure_maxexpect",
                           "linearfold_vienna", "linearpartition"}:
            official_weights = {"status": "NOT_APPLICABLE_NO_LEARNED_WEIGHTS", "path": None,
                                "parameters_or_binary": metadata.get("parameters") or metadata.get("binary"),
                                "sha256": metadata.get("parameters_sha256") or metadata.get("binary_sha256")}
        elif weights_path or weights_sha:
            distribution = metadata.get("distribution") or "LOCAL_FROZEN_ASSET"
            official_weights = {"status": "LOCAL_FROZEN_ASSET", "path": weights_path,
                                "sha256": weights_sha, "distribution": distribution,
                                "official_distribution_verified": baseline_id not in {"ufold", "spot_rna"}}
        else:
            official_weights = {"status": "BLOCKED_PROVENANCE", "path": None, "sha256": None,
                                "reason": "official/current checkpoint asset or identity not verified"}
        training = _default_training(baseline_id)
        license_rec = {"status": "BLOCKED_PROVENANCE", "spdx": None,
                       "reason": "model/weight license not verified; dataset or framework license is not a substitute"}
        if baseline_id in CHANRG_29 and baseline_id.rsplit("_", 1)[-1] in {"linear", "resnet", "unet"}:
            license_rec["framework_code_license"] = "MultiMolecule AGPL-3.0-or-later; not a predictor-weight license"
        if baseline_id in {"xens2", "best_plan_a", "r2d", "h0", "chanrg_ens3"}:
            license_rec = {"status": "PROJECT_INTERNAL_REDISTRIBUTION_REVIEW_PENDING", "spdx": None,
                           "reason": "project asset; no external license inferred"}
        inference = {"status": "FROZEN_RAW_PROTOCOL" if frozen else "BLOCKED_PROVENANCE",
                     "version_protocol": spec["version_protocol"],
                     "decoder": metadata.get("decoder"), "threshold": metadata.get("threshold"),
                     "evaluation_scope": metadata.get("evaluation_scope"),
                     "efold_version": metadata.get("efold_version")}
        if summaries:
            inference["status"] = "SUMMARY_ONLY_NO_PER_SEQUENCE_PREDICTION"
        if frozen:
            comparison = "PROJECT_RAW_GT_ONLY; raw is not clean and is not automatically CHANRG-protocol comparable"
        elif summaries or spec["status"] == "LEGACY_TABLE_ONLY":
            comparison = "LEGACY_TABLE_ONLY; no frozen per-sequence join; external protocol numbers must not be ranked as project raw predictions"
        else:
            comparison = "NOT_COMPARABLE_UNTIL_IDENTITY_WEIGHTS_AND_PREDICTIONS_ARE_RESOLVED"
        provenance_blocked = any(str(rec.get("status", "")).startswith("BLOCKED_PROVENANCE")
                                 for rec in (official_weights, training, license_rec))
        row = {
            "baseline_id": baseline_id, "model": spec["model"], "category": spec["category"],
            "row_type": spec["row_type"], "role": "baseline",
            "source_cohorts": [m["cohort"] for m in memberships], "cohort_memberships": memberships,
            "version_protocol": spec["version_protocol"], "official_weights": official_weights,
            "training_data": training, "license": license_rec, "inference": inference,
            "comparability": {"status": comparison, "clean_subset_policy": "baseline-specific; no shared TR1c-clean assumed",
                              "e0_expanded_71_used": any(p.get("e0_scope") == "E0_EXPANDED_71" for p in frozen)},
            "provenance_status": "BLOCKED_PROVENANCE" if provenance_blocked else "PARTIAL_OR_VERIFIED",
            "evidence": evidence, "evidence_additional": additional,
            "status": spec["status"], "inclusion_reason": spec["inclusion_reason"],
            "exclusion_reason": None,
            "prediction_registry_scope": "FROZEN_RAW" if frozen else "SUMMARY_ONLY" if summaries else "BLOCKED" if blocked else "NO_CURRENT_PREDICTION",
            "prediction_ids": [p["prediction_id"] for p in preds],
            "n_frozen_prediction_rows": len(frozen), "n_summary_only_rows": len(summaries),
            "n_blocked_prediction_rows": len(blocked),
            "unknown_fields_are_not_zero": True,
        }
        if baseline_id in CHANRG_29:
            head = baseline_id.rsplit("_", 1)[-1]
            row["legacy_result"] = {"status": "LEGACY_TABLE_ONLY", "numeric_values_copied": False,
                                    "result_access": "CANONICAL_MAIN_FIGURE_ONLY" if head == "unet" or head not in {"linear", "resnet"} else "EXTENDED_DATA_NOT_RETRIEVED",
                                    "identity_evidence": "RECONSTRUCTED_FROM_MAIN_AND_FRAMEWORK" if head in {"linear", "resnet", "unet"} else "VERIFIED_MAIN_FIGURE"}
        rows.append(row)
    return rows


def link_registries(data: list[dict], predictions: list[dict], baselines: list[dict]) -> list[str]:
    """Add forward/backward links and return every unresolved relation."""
    errors = []
    baseline_by_id = {row["baseline_id"]: row for row in baselines}
    data_by_id = {row["dataset_id"]: row for row in data}
    for row in data:
        row["prediction_ids"] = []
    for pred in predictions:
        baseline_id = PREDICTION_MODEL_TO_BASELINE.get(pred["model"])
        pred["baseline_id"] = baseline_id
        pred["dataset_id"] = pred["split"]
        if baseline_id not in baseline_by_id:
            errors.append(f"prediction {pred['prediction_id']} has no baseline row")
        if pred["split"] not in data_by_id:
            errors.append(f"prediction {pred['prediction_id']} has no dataset row")
        else:
            data_by_id[pred["split"]]["prediction_ids"].append(pred["prediction_id"])
    for baseline in baselines:
        expected = [p["prediction_id"] for p in predictions if p.get("baseline_id") == baseline["baseline_id"]]
        if baseline["prediction_ids"] != expected:
            errors.append(f"baseline {baseline['baseline_id']} prediction_ids mismatch")
        for dataset_id in baseline["training_data"].get("project_training_dataset_ids", []):
            if dataset_id not in data_by_id:
                errors.append(f"baseline {baseline['baseline_id']} training dataset {dataset_id} missing")
            elif baseline["baseline_id"] not in data_by_id[dataset_id].setdefault("training_baseline_ids", []):
                data_by_id[dataset_id]["training_baseline_ids"].append(baseline["baseline_id"])
    return errors


def baseline_metadata(rows: list[dict]) -> dict:
    core = set().union(*(set(roster) for roster in COHORTS.values()))
    core_rows = {row["baseline_id"] for row in rows if row["source_cohorts"]}
    return {
        "identity_granularity": "model family + checkpoint + decoding/head variant; method-level CHANRG aliases never imply version equality",
        "cohort_sizes": {name: len(roster) for name, roster in COHORTS.items()},
        "source_membership_total": sum(len(roster) for roster in COHORTS.values()),
        "core_union_ids": sorted(core),
        "core_union_count": len(core),
        "supplemental_row_count": len(rows) - len(core_rows),
        "four_category_counts": {category: sum(row["category"] == category for row in rows)
                                 for category in ("STRUCTURED_DECODER", "DIRECT_NEURAL", "FOUNDATION_MODEL", "PROJECT")},
        "chanrg_composition": {
            "structured_decoders": 8, "direct_neural": 3, "foundation_backbones": 6,
            "heads_per_backbone": ["Linear", "ResNet", "U-Net"], "foundation_predictors": 18,
            "total_predictors": 29, "canonical_main_figure_models": 17,
            "identity_evidence": "RECONSTRUCTED_FROM_MAIN_AND_MULTIMOLECULE_0_2_3_SOURCE",
            "extended_data_results": "BLOCKED_NOT_RETRIEVED",
            "checkpoint_availability": "BLOCKED_PROVENANCE_FOR_INDIVIDUAL_HEADS",
            "warning": "framework head support is not evidence that every checkpoint is published",
        },
        "legacy_18_disposition": LEGACY_DISPOSITION,
        "exclusions": EXCLUSIONS,
        "sota_protocol": "one frozen model/config per protocol; raw/clean and project/CHANRG panels separate; LEGACY_TABLE_ONLY is never a frozen prediction or SOTA claim",
        "scientific_decision_effect": "REGISTRY_COMPLETION_ONLY; E0/E1 decision unchanged",
    }
