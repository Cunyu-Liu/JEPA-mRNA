import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    h0 = torch.load(ROOT / "artifacts/p2/pilot_v1/H0/best.pt", map_location="cpu", weights_only=False, mmap=True)
    h2 = torch.load(ROOT / "artifacts/p2/pilot_v1/H2/best.pt", map_location="cpu", weights_only=False, mmap=True)
    differences = {name: float((h0["model"]["encoder"][name] - h2["model"]["encoder"][name]).abs().max())
                   for name in h0["model"]["encoder"]}
    assert h0["step"] == h2["step"] == 100
    assert h0["meta"]["schedule"]["warmup_head_steps"] == h2["meta"]["schedule"]["warmup_head_steps"] == 1600
    assert all(value == 0 for value in differences.values())
    feature_evidence = {}
    for split in ("train", "holdout"):
        hashes = {}
        for mode in ("h0", "h2"):
            path = next((ROOT / "artifacts/v2/interface/features" / mode / split).glob("*.npz"))
            with np.load(path, allow_pickle=False) as saved:
                hashes[mode] = hashlib.sha256(saved["embedding"].tobytes()).hexdigest()
        assert hashes["h0"] == hashes["h2"]
        feature_evidence[split] = hashes
    result = {"status": "ALIASED_H0_SHARED_FROZEN_ENCODER_NOT_INDEPENDENT_JEPA_TRANSFER",
              "scientific_scope": "encoder identity and warmup fact; not a new accuracy result or a proof that JEPA cannot work",
              "h0_step": h0["step"], "h2_step": h2["step"], "warmup_head_steps": 1600,
              "encoder_exact_equal": True, "max_abs_difference": 0.0,
              "h0_checkpoint_sha256": sha256(ROOT / "artifacts/p2/pilot_v1/H0/best.pt"),
              "h2_checkpoint_sha256": sha256(ROOT / "artifacts/p2/pilot_v1/H2/best.pt"),
              "feature_evidence": feature_evidence,
              "h2_seed42_probe_retained_as_alias": True,
              "remaining_duplicate_h2_runs_stopped": True,
              "source_sha256": sha256(__file__)}
    output = ROOT / "artifacts/v2/interface/h2_encoder_identity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
