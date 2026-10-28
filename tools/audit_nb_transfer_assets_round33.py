#!/usr/bin/env python3
"""Verify recovered JEV-on-NB results and document the full-FT evidence boundary."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

ROOT = Path(os.environ.get("NB_TRANSFER_ROOT", "/private/tmp/rnajev_audit/round33/nb_transfer"))
JEV_ROOT = Path(os.environ.get("NB_JEV_ROOT", ROOT / "nb_jev_r2dtr1c_s0_step20000" if (ROOT / "nb_jev_r2dtr1c_s0_step20000").exists() else ROOT))
BASELINE_ROOT = Path(os.environ.get("NB_BASELINE_ROOT", ROOT))
SOURCE = Path(os.environ.get("SOURCE_DATA_JSON", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03/figures/source_data.json"))
OUT = Path(os.environ.get("NB_TRANSFER_MANIFEST_OUT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261028_01/figures/nb_transfer_recovery_manifest.json"))
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3", "ref_pdb_ts_hard", "testsetb", "archiveii_embok_clean"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))["nb_transfer"]
    jev_path = JEV_ROOT
    records = []
    for index, split in enumerate(SPLITS):
        result_split = "archiveii_embok_clean_trunc598" if split == "archiveii_embok_clean" else split
        path = jev_path / result_split / "result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        micro = data["pair_level"]["micro"]
        recomputed = 2 * micro["tp"] / (2 * micro["tp"] + micro["fp"] + micro["fn"])
        assert np.isclose(recomputed, micro["f1"], rtol=0, atol=1e-12)
        assert round(recomputed, 4) == round(source["jev_on_nb"][index], 4)
        assert data["device"] == "cuda" and data["n_sequences"] > 0
        baseline_path = BASELINE_ROOT / f"baselines_nucleicbert_{split}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_micro = baseline["baselines"]["nucleicbert"]["micro_f1"]
        baseline_recomputed = 2 * baseline["baselines"]["nucleicbert"]["tp"] / (
            2 * baseline["baselines"]["nucleicbert"]["tp"] + baseline["baselines"]["nucleicbert"]["fp"] + baseline["baselines"]["nucleicbert"]["fn"])
        assert np.isclose(baseline_micro, baseline_recomputed, rtol=0, atol=1e-12)
        records.append({"split": split, "jev_result": str(path), "jev_result_sha256": sha256(path),
                        "jev_micro_f1": recomputed, "n_sequences": data["n_sequences"],
                        "frozen_official_probe_result": str(baseline_path),
                        "frozen_official_probe_sha256": sha256(baseline_path),
                        "frozen_official_probe_micro_f1": baseline_micro,
                        "frozen_probe_is_not_full_ft": True})
    payload = {"schema_version": 1, "status": "PASS", "n_splits": len(records), "records": records,
               "jev_protocol": "recovered original nb_jev_r2dtr1c_s0_step20000 result.json files",
               "full_ft_evidence": {
                   "status": "full-FT result JSON not found in the audited shutdown backup",
                   "current_use": "retain the previously verified source_data.nb_transfer.nb_ft values and existing PPT panel; do not substitute frozen-probe values",
                   "nb_ft_values": source["nb_ft"]},
               "scientific_boundary": "Frozen official-probe and full fine-tuned NucleicBERT are distinct comparators; this audit does not create a new full-FT result"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checked = json.loads(temporary.read_text(encoding="utf-8"))
    assert checked["n_splits"] == 8 and checked["status"] == "PASS"
    temporary.replace(OUT)
    print("NB_TRANSFER_RECOVERY_AUDIT_PASS", json.dumps({
        "n_splits": len(records), "full_ft_json_found": False,
        "output": str(OUT), "sha256": sha256(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
