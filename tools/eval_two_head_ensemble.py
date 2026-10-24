#!/usr/bin/env python3
"""Round-28 T-A111: two-seed head ensemble eval for the CHANRG corpus arms.

Loads chanrg_r2d_s0_gt + chanrg_r2d_s1_gt (identical config, different seeds),
wraps both heads into one averaged-score model, then runs the full
evaluate_decision.evaluate() pipeline (DP decode, Platt calibration on
chanrg_train_validation, legality, latency) with the ensemble in place of a
single model. Single variable vs the single-head evals: head count (1 -> 2).
"""
import argparse
import sys
from dataclasses import asdict
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


class TwoHeadEnsemble(nn.Module):
    """Averages both heads' scores; keeps the head-only contract the evaluator
    expects (model.head, model.head_only, forward(h, ids, lengths) -> scores,
    plus the prior_weight / calibration attributes the decode path reads)."""

    def __init__(self, head_a, head_b, w_a=0.5):
        super().__init__()
        self.head_a = head_a
        self.head_b = head_b
        self.w_a = w_a
        self.head_only = True
        # evaluator reads these off model.head (reweight_turner_prior, L767):
        # average the trained prior weight, delegate calibration to head_a
        # (both heads share the calibration structure; s0/s1 same config)
        try:
            self._prior_weight = 0.5 * (head_a.prior_weight + head_b.prior_weight)
        except (AttributeError, TypeError):
            self._prior_weight = getattr(head_a, "prior_weight", None)
        self._calibration = getattr(head_a, "calibration", None)
        # plain-object shim: attribute reads work, no nn.Module cycle
        object.__setattr__(self, "head", _HeadShim(self))

    def forward(self, h, ids=None, lengths=None):
        out_a = self.head_a(h, ids, lengths=lengths)
        out_b = self.head_b(h, ids, lengths=lengths)
        ta = out_a.scores if hasattr(out_a, "scores") else out_a
        tb = out_b.scores if hasattr(out_b, "scores") else out_b
        avg = self.w_a * ta + (1.0 - self.w_a) * tb
        if hasattr(out_a, "scores"):
            out_a.scores = avg
            return out_a
        return avg

    def load_state_dict(self, *a, **k):  # weights already in place
        return [], []  # (missing, unexpected) contract of nn.Module

    def to(self, *a, **k):
        super().to(*a, **k)
        return self


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--w-a", type=float, default=0.5)
    args, rest = ap.parse_known_args(argv)
    # strip a leading bare "--" if the caller passed one
    if rest and rest[0] == "--":
        rest = rest[1:]
    device = "cuda"
    for i, a in enumerate(rest):
        if a == "--device" and i + 1 < len(rest):
            device = rest[i + 1]

    heads = []
    for path in [args.ckpt_a, args.ckpt_b]:
        ckpt = torch.load(path, map_location="cpu")
        config = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                                if k in TrainConfig.__dataclass_fields__})
        config = TrainConfig(**{**asdict(config), "encoder_size": "35M",
                                "device": device, "allow_cpu": False})
        m = build_decision_model(config)
        m.load_state_dict(ckpt["model"], strict=False)
        m.eval()
        heads.append(m.head)

    ens = TwoHeadEnsemble(heads[0], heads[1], w_a=args.w_a).to(device).eval()

    # swap the model constructor inside evaluate_decision for ours
    orig_build = ed.build_decision_model

    def fake_build(cfg):
        # keep evaluator's head-chunk override visible to our heads
        if getattr(cfg, "head_chunk_size", 0):
            for h in (ens.head_a, ens.head_b):
                if hasattr(h, "chunk_size"):
                    h.chunk_size = int(cfg.head_chunk_size)
        return ens

    ed.build_decision_model = fake_build

    ed_argv = ["--checkpoint", args.ckpt_a] + rest
    ns = ed.main(ed_argv)  # writes result.json under ns.out via evaluate()

    ed.build_decision_model = orig_build
    return ns


if __name__ == "__main__":
    raise SystemExit(main())
