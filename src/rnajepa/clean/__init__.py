"""Auditable data cleaning pipeline (C1-C6) for RNA secondary-structure data.

See ``rna-jepa/spec/analysis_plan.md`` and ``rna-jepa/spec/constraints.md`` for
the frozen plan this package implements, and spec §4 for the stage definitions.

Public entry point: :func:`run_pipeline`.
"""

from .c1_sequence import (
    LOW_COMPLEXITY_IS_STUB,
    clean_sequence,
    clean_sequences,
    normalize_sequence,
    shannon_low_complexity,
)
from .c2_redundancy import (
    DRY_RUN_IS_APPROXIMATION,
    ToolUnavailableError,
    deduplicate,
    detect_contamination,
    family_group_split,
    family_overlap,
)
from .c3_structure import clean_structures, parse_pairs, validate_structure
from .c4_labels import (
    LabelConfidence,
    assign_confidence_tier,
    multi_source_agreement,
    normalize_2_8,
    normalize_boxplot,
    trim_outliers,
)
from .c5_audit import audit_report, build_manifest, distribution_report, split_distances
from .c6_splits import (
    FrozenSplit,
    family_ood_split,
    freeze_archiveii,
    freeze_bprna_new,
    freeze_bprna_ts_tr_tm,
    freeze_family_split,
    freeze_pdb_ts,
    freeze_rnastralign,
    gc_bucket_ood,
    length_bucket_ood,
)
from .data_card import render_data_card_from_result, render_data_card_template
from .pipeline import CleanConfig, CleanResult, run_pipeline
from .records import AttritionTable, ReasonCode, Record, Removal

__all__ = [
    "AttritionTable",
    "CleanConfig",
    "CleanResult",
    "DRY_RUN_IS_APPROXIMATION",
    "FrozenSplit",
    "LOW_COMPLEXITY_IS_STUB",
    "LabelConfidence",
    "ReasonCode",
    "Record",
    "Removal",
    "ToolUnavailableError",
    "assign_confidence_tier",
    "audit_report",
    "build_manifest",
    "clean_sequence",
    "clean_sequences",
    "clean_structures",
    "deduplicate",
    "detect_contamination",
    "distribution_report",
    "family_group_split",
    "family_ood_split",
    "family_overlap",
    "freeze_archiveii",
    "freeze_bprna_new",
    "freeze_bprna_ts_tr_tm",
    "freeze_family_split",
    "freeze_pdb_ts",
    "freeze_rnastralign",
    "gc_bucket_ood",
    "length_bucket_ood",
    "multi_source_agreement",
    "normalize_2_8",
    "normalize_boxplot",
    "normalize_sequence",
    "parse_pairs",
    "render_data_card_from_result",
    "render_data_card_template",
    "run_pipeline",
    "shannon_low_complexity",
    "split_distances",
    "trim_outliers",
    "validate_structure",
]
