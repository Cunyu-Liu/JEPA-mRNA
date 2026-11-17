import json
from pathlib import Path

import numpy as np

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
summary = {"status": "PASS_INTERFACE_BASELINE_EVALUATION", "scientific_scope": "lightweight interface baseline; not dual-view JEPA or full RNA structure prediction", "modes": {}}
for mode in ("sequence_only", "structure_aware", "length", "gc", "random"):
    rows = []
    for seed in (42, 2026, 3407):
        result = json.loads((ROOT / f"{mode}_seed{seed}/result.json").read_text())
        assert result["train_holdout_group_overlap"] == 0
        assert result["train_holdout_pdb_overlap"] == 0
        assert not result["holdout_structure_labels_used_for_inference"]
        rows.append(result)
    auprc = [row["auprc"] for row in rows]
    macro = [row["macro_f1"] for row in rows]
    summary["modes"][mode] = {
        "auprc_mean": float(np.mean(auprc)), "auprc_sd": float(np.std(auprc)),
        "macro_f1_mean": float(np.mean(macro)), "macro_f1_sd": float(np.std(macro)),
        "seeds": [{"seed": row["seed"], "auprc": row["auprc"], "macro_f1": row["macro_f1"],
                   "best_step": row["best_step"], "converged": row["converged"],
                   "checkpoint_sha256": row["checkpoint_sha256"]} for row in rows],
    }
out = ROOT / "baseline_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
