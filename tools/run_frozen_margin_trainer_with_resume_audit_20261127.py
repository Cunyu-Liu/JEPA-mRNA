#!/usr/bin/env python3
"""Execute the frozen trainer while recording resume validation control flow."""
from __future__ import annotations

import sys
from pathlib import Path

SOURCE_REPO = Path("/home/rna_jev_margin_candidate")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(SOURCE_REPO / "src"))

from tools import train_plan_a_gec_convergence_v1 as trainer


class ValidationAuditList(list):
    def __init__(self, values):
        super().__init__(values)
        self.truth_checks = 0

    def __bool__(self):
        self.truth_checks += 1
        result = list.__len__(self) > 0
        if self.truth_checks == 1 or self.truth_checks % 500 == 0:
            print(f"[resume-audit] validation_truth_check={self.truth_checks} "
                  f"records={list.__len__(self)} result={result}", flush=True)
        return result


def load_with_validation_audit(config, split, record_limit=0, verify_data_hash=True):
    records = trainer._load_registered_records_without_audit(
        config, split, record_limit, verify_data_hash,
    )
    if split == "validation":
        records = ValidationAuditList(records)
        print(f"[resume-audit] loaded_validation_records={len(records)}", flush=True)
    return records


def validation_record_with_audit(*args, **kwargs):
    print(f"[resume-audit] validation_record_step={args[0]}", flush=True)
    return trainer._build_validation_log_record_without_audit(*args, **kwargs)


def main() -> None:
    trainer._load_registered_records_without_audit = trainer.load_registered_records
    trainer._build_validation_log_record_without_audit = trainer.build_validation_log_record
    trainer.load_registered_records = load_with_validation_audit
    trainer.build_validation_log_record = validation_record_with_audit
    sys.argv[0] = str(SOURCE_REPO / "tools/train_plan_a_gec_convergence_v1.py")
    trainer.main()


if __name__ == "__main__":
    main()
