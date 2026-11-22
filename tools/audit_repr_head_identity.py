#!/usr/bin/env python
"""Identity audit: instantiate each arm's head and verify architecture knobs.

Checks proj width (d_z), scorer stem width (hidden), block count, output head,
total head params, and seed-determinism of the fresh initialization. CPU-only;
no training. This is the pre-pilot identity gate for the representation/head
capacity matrix.
"""
import json, sys
import torch
sys.path.insert(0, "/home/rna_jev_xens2_dualview_round56/src")
from rnajepa.decision_head import FlatDecisionHead

D_MODEL = 1280
ARMS = {
    "control_matched": {"d_z": 128, "resnet_blocks": 4, "resnet_hidden": 64},
    "repr_wide":       {"d_z": 256, "resnet_blocks": 4, "resnet_hidden": 64},
    "head_deep":       {"d_z": 128, "resnet_blocks": 8, "resnet_hidden": 128},
    "repr_head":       {"d_z": 256, "resnet_blocks": 8, "resnet_hidden": 128},
}

def build(cfg, seed=0):
    torch.manual_seed(seed)
    return FlatDecisionHead(d_model=D_MODEL, scorer="resnet2d", chunk_size=0, **cfg)

def main():
    report = {"d_model": D_MODEL, "arms": {}, "all_pass": True}
    for name, cfg in ARMS.items():
        h = build(cfg); sd = h.state_dict()
        proj = list(sd["pair_repr.proj.weight"].shape)
        stem = list(sd["resnet2d.stem.0.weight"].shape)
        headw = list(sd["resnet2d.head.0.weight"].shape)
        blk_idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("resnet2d.blocks.")})
        nparams = sum(p.numel() for p in h.parameters())
        h2 = build(cfg); sd2 = h2.state_dict()
        deterministic = all(torch.equal(sd[k], sd2[k]) for k in sd)
        checks = {
            "proj_width_is_d_z": proj[0] == cfg["d_z"] and proj[1] == 3 * D_MODEL,
            "stem_hidden_ok": stem[0] == cfg["resnet_hidden"] and stem[1] == cfg["d_z"] + 1,
            "output_head_ok": headw == [1, cfg["resnet_hidden"], 1, 1],
            "block_count_ok": blk_idx == list(range(cfg["resnet_blocks"])),
            "fresh_init_deterministic": deterministic,
        }
        ok = all(checks.values())
        report["all_pass"] &= ok
        report["arms"][name] = {"config": cfg, "proj_shape": proj, "stem_shape": stem,
                                "output_head_shape": headw, "n_blocks": len(blk_idx),
                                "head_params": nparams, "checks": checks, "pass": ok}
        print(f"{name}: proj={proj} stem={stem} blocks={len(blk_idx)} params={nparams} "
              f"det={deterministic} PASS={ok}")
    print("ALL_PASS=", report["all_pass"])
    out = "/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/representation_head_mechanism_4arm_v1/identity_audit.json"
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print("wrote", out)
    sys.exit(0 if report["all_pass"] else 1)

if __name__ == "__main__":
    main()