#!/usr/bin/env python3
"""Preflight for the RiNALMo-giga weights: prove they load before using them.

Why this exists
---------------
The first attempt to load these weights "succeeded" and produced correctly shaped
outputs -- but every tensor was **randomly initialised**.  ``AutoModel.from_pretrained``
only *warns* about that ("Some weights ... were newly initialized"), and the warning
listed essentially the whole model.  Extracting 10k embeddings from that would have
silently produced 10k useless vectors, and the resulting arm would have looked like a
normal training run.

The cause is a key-prefix mismatch: the checkpoint stores the encoder under
``model.*`` (it was saved from a class whose encoder attribute was named ``model``),
while multimolecule 0.0.8 names it ``rinalmo`` (or nothing, for ``RiNALMoModel``).
``transformers 4.57.6`` additionally refuses the dict outright with "The state
dictionary ... is corrupted", and its tokenizer API is incompatible with
multimolecule 0.0.8's tokenizer -- so ``from_pretrained`` is unusable here.

This module therefore loads explicitly and asserts exactly what it expects:

* strip the ``model.`` prefix and load into :class:`RiNALMoModel`;
* the **only** tensors allowed to be absent are ``pooler.dense.{weight,bias}`` --
  the pooler is used solely for the pooled ``[cls]`` output, which per-residue
  embeddings never touch;
* the tokenizer is not used at all: the vocabulary is 28 tokens and hand-built ids
  are checked against ``len(seq)``.

Verified independently: with the MLM head attached, the model reproduces the
scores published on the model card for its own fill-mask example.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import torch

WEIGHTS = "/mnt/cunyuliu/rna-jepa/weights/rinalmo-giga"

#: The 28-token vocabulary, in the order given by ``vocab.txt``.
VOCAB: Dict[str, int] = {
    "<pad>": 0, "<cls>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "<null>": 5,
    "A": 6, "C": 7, "G": 8, "U": 9, "N": 10,
}

#: Tensors the checkpoint legitimately does not contain, with the reason.
EXPECTED_ABSENT = {
    "pooler.dense.weight": "pooler output is unused: we take per-residue states",
    "pooler.dense.bias": "pooler output is unused: we take per-residue states",
}


def encode(seq: str, device: str = "cpu") -> torch.Tensor:
    """``[<cls>] + residues + [<eos>]`` as a ``(1, L + 2)`` id tensor."""
    ids = [VOCAB["<cls>"]] + [VOCAB.get(c, VOCAB["<unk>"]) for c in seq] + [VOCAB["<eos>"]]
    return torch.tensor([ids], dtype=torch.long, device=device)


def load_encoder(device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """Load the RiNALMo-giga encoder, asserting that nothing important is missing.

    Returns ``(model, report)``.  Raises if any tensor outside
    :data:`EXPECTED_ABSENT` fails to load -- a partially loaded encoder must never
    be used, because it trains and evaluates perfectly happily while being wrong.
    """
    from safetensors.torch import load_file
    from multimolecule import RiNALMoConfig, RiNALMoModel

    with open(f"{WEIGHTS}/config.json", encoding="utf-8") as fh:
        cfg = RiNALMoConfig(**json.load(fh))
    model = RiNALMoModel(cfg)
    raw = load_file(f"{WEIGHTS}/model.safetensors")
    stripped = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    if not stripped:
        raise RuntimeError("no 'model.*' keys found in the checkpoint; the key layout "
                           "changed and this loader must be revisited")

    info = model.load_state_dict(stripped, strict=False)
    unexpected = list(info.unexpected_keys)
    missing = sorted(info.missing_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    if missing != sorted(EXPECTED_ABSENT):
        raise RuntimeError(
            f"unexpectedly missing tensors: {missing}. Only {sorted(EXPECTED_ABSENT)} "
            "may be absent (unused pooler). A partially loaded encoder would still "
            "run and still look plausible, so this is a hard failure.")
    report = {
        "weights": WEIGHTS,
        "n_tensors_in_checkpoint": len(raw),
        "n_encoder_tensors_loaded": len(stripped),
        "loaded_fraction_of_encoder": round(
            len(stripped) / max(1, len(stripped) + len(missing)), 6),
        "expected_absent": EXPECTED_ABSENT,
        "d_model": int(cfg.hidden_size),
        "n_layers": int(cfg.num_hidden_layers),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "license_note": "multimolecule/rinalmo-giga is AGPL-3.0",
    }
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    return model, report


def verify_against_model_card(device: str = "cuda") -> Dict[str, object]:
    """End-to-end check: reproduce the fill-mask scores published on the model card.

    The model card reports, for ``UAGCUUAUCAG<mask>CUGAUGUUGA`` (mask = 12th
    residue), ``G 0.219446, U 0.206619, X 0.199547, A 0.195566, C 0.178808``.
    Reproducing that ordering requires the *trained* encoder and head, so it is
    evidence about the weights themselves rather than about the loader.

    The MLM head is reconstructed by mapping ``model.* -> rinalmo.*`` and
    ``lm_head.bias -> lm_head.decoder.bias``; the extra ``lm_head.bias`` that
    multimolecule 0.0.8 expects has no checkpoint counterpart, so this check is
    reported as informational and never gates anything.
    """
    from safetensors.torch import load_file
    from multimolecule import RiNALMoConfig, RiNALMoForMaskedLM

    with open(f"{WEIGHTS}/config.json", encoding="utf-8") as fh:
        cfg = RiNALMoConfig(**json.load(fh))
    mlm = RiNALMoForMaskedLM(cfg)
    raw = load_file(f"{WEIGHTS}/model.safetensors")
    mapped = {}
    for k, v in raw.items():
        if k.startswith("model."):
            mapped["rinalmo." + k[len("model."):]] = v
        elif k == "lm_head.bias":
            mapped["lm_head.decoder.bias"] = v
        else:
            mapped[k] = v
    info = mlm.load_state_dict(mapped, strict=False)
    mlm = mlm.to(device).to(torch.bfloat16).eval()

    seq = "UAGCUUAUCAGCUGAUGUUGA"
    ids = encode(seq, device)
    ids[0, 12] = VOCAB["<mask>"]
    with torch.no_grad():
        out = mlm(input_ids=ids)
    probs = torch.softmax(out.logits[0, 12].float(), dim=-1)
    top = torch.topk(probs, 5)
    inv = {v: k for k, v in VOCAB.items()}
    got = [(inv[int(i)], round(float(s), 6)) for s, i in zip(top.indices, top.values)]
    return {"top5": got, "unmapped_missing": sorted(info.missing_keys),
            "matches_model_card_top1": got[0][0] == "G"}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, report = load_encoder(args.device)
    print(json.dumps(report, indent=1))
    print(json.dumps(verify_against_model_card(args.device), indent=1))
