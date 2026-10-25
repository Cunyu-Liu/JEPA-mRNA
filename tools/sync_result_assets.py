#!/usr/bin/env python3
"""从验收后的统计 JSON 同步六臂图源数据，禁止手工转录。"""
from pathlib import Path
import json
import os

ROOT = Path(os.environ.get("RNAJEV_DECK_ROOT", "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/temp_20261025_03"))
FIG = ROOT / "figures"
stats = json.loads((FIG / "matched_30k_statistics.json").read_text(encoding="utf-8"))
source_path = FIG / "source_data.json"
source = json.loads(source_path.read_text(encoding="utf-8"))
assert stats["status"] == "PASS" and stats["n_arms"] == 6 and stats["n_cells"] == 24
order = ["chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf"]
current = source["matched_30k_current"]
for arm, record in stats["arms"].items():
    short = arm.replace("chanrg_clean_30k_", "clean_") if arm.startswith("chanrg_clean") else arm.replace("mixed_30k_", "mixed_")
    current[short] = [record["metrics"][split] for split in order]
current["statistics"] = {
    "family_statistics": stats["family_statistics"],
    "verdicts": stats["verdicts"],
    "paired_clean_vs_mixed": stats["paired_clean_vs_mixed"],
}
current["note"] = "All six arms are complete and independently verified; three-seed statistics and preregistered verdicts are included."
source_path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
checked = json.loads(source_path.read_text(encoding="utf-8"))
for family in ("clean", "mixed"):
    for seed in range(3):
        key = f"{family}_s{seed}"
        assert key in checked["matched_30k_current"]
        values = checked["matched_30k_current"][key]
        print(key, "OOD_mean", sum(values[1:]) / 3)
print("SYNC_RESULT_ASSETS_PASS")
