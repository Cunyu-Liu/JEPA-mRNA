#!/usr/bin/env python3
"""Round-28 T-A112: three-seed head ensemble eval for the CHANRG GT arms.

Loads chanrg_r2d_{s0,s1,s2}_gt (identical config, different seeds), wraps
all heads into one averaged-score model, then runs the full
evaluate_decision.evaluate() pipeline (DP decode, Platt calibration on
chanrg_train_validation, legality, latency) with the ensemble in place of a
single model. Single variable vs the two-head eval: head count (2 -> 3).
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Optional

import torch
import torch.nn as nn

sys.path.insert(0, "/home/rna_jev/src")
sys.path.insert(0, "/home/rna_jev/eval/ss")

import evaluate_decision as ed
from rnajepa.train_decision import TrainConfig, build_decision_model


class _HeadShim:
    """Non-Module attribute holder: evaluator reads prior_weight/calibration
    off model.head; forward goes through the model itself, not the shim, so
    no module-cycle recursion."""

    def __init__(self, ens):
        self._ens = ens

    @property
    def prior_weight(self):
        return self._ens._prior_weight

    @property
    def calibration(self):
        return self._ens._calibration


class MultiHeadEnsemble(nn.Module):
    """Averages N heads' scores; keeps the head-only contract the evaluator
    expects (model.head, model.head_only, forward(h, ids, lengths) -> scores,
    plus the prior_weight / calibration attributes the decode path reads)."""

    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.head_only = True
        w = [getattr(h, "prior_weight", None) for h in heads]
        w = [x for x in w if isinstance(x, (int, float))]
        self._prior_weight = sum(w) / len(w) if w else None
        self._calibration = getattr(heads[0], "calibration", None)
        object.__setattr__(self, "head", _HeadShim(self))

    def forward(self, h, ids=None, lengths=None):
        outs = [hh(h, ids, lengths=lengths) for hh in self.heads]
        ts = [o.scores if hasattr(o, "scores") else o for o in outs]
        avg = sum(ts) / len(ts)
        if hasattr(outs[0], "scores"):
            outs[0].scores = avg
            return outs[0]
        return avg

    def load_state_dict(self, *a, **k):  # weights already in place
        return [], []  # (missing, unexpected) contract of nn.Module

    def to(self, *a, **k):
        super().to(*a, **k)
        return self


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="two or more head checkpoints to average")
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    device = "cuda"
    for i, a in enumerate(rest):
        if a == "--device" and i + 1 < len(rest):
            device = rest[i + 1]

    heads = []
    checkpoint_meta = []
    for path in args.ckpts:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        checkpoint_meta.append({"path": path, "sha256": digest.hexdigest()})
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                                if k in TrainConfig.__dataclass_fields__})
        config = TrainConfig(**{**asdict(config), "encoder_size": "35M",
                                "device": device, "allow_cpu": False})
        m = build_decision_model(config)
        missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
        missing_head = [key for key in missing if key.startswith("head.")]
        if unexpected or missing_head:
            raise RuntimeError(f"incompatible head checkpoint {path}: "
                               f"missing_head={missing_head}, unexpected={list(unexpected)}")
        m.eval()
        heads.append(m.head)

    ens = MultiHeadEnsemble(heads).to(device).eval()

    orig_build = ed.build_decision_model

    def fake_build(cfg):
        if getattr(cfg, "head_chunk_size", 0):
            for h in ens.heads:
                if hasattr(h, "chunk_size"):
                    h.chunk_size = int(cfg.head_chunk_size)
        return ens

    ed.build_decision_model = fake_build

    ed_argv = ["--checkpoint", args.ckpts[0]] + rest
    try:
        ns = ed.main(ed_argv)  # writes result.json under ns.out via evaluate()
        result_path = Path(ns.out) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["ensemble"] = {"n_heads": len(heads), "checkpoints": checkpoint_meta,
                              "aggregation": "arithmetic_mean_of_head_scores"}
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        return ns
    finally:
        ed.build_decision_model = orig_build


if __name__ == "__main__":
    raise SystemExit(main())
