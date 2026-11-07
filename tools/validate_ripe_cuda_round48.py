import argparse
import hashlib
import json
from pathlib import Path

import torch

from ripe_structrfm import enable_ripe_structrfm
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.ripe import RipeConfig, RipePositionalBias
from structrfm_round48 import load_structrfm_encoder


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(pretrained, lengths):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    encoder, tokenizer, load_report = load_structrfm_encoder(pretrained, device="cuda")
    vocab = {base: tokenizer.convert_tokens_to_ids(base) for base in "ACGUT"}
    assert len(set(vocab.values())) == 5
    encoder = enable_ripe_structrfm(encoder, "r3", vocab=vocab)
    encoder.train()
    for module in encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    ripe_parameters = []
    for layer in encoder.encoder.layer:
        ripe_parameters.extend(layer.attention.self.ripe.parameters())
    for parameter in ripe_parameters:
        parameter.requires_grad_(True)
    encoder_rows = []
    ids = torch.tensor([vocab[base] for base in "ACGU"], device="cuda")
    for length in lengths:
        input_ids = ids.repeat((length + 3) // 4)[:length].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if length == min(lengths):
                output = encoder(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        assert hidden.shape == (1, length, 768)
        first_attention = encoder.encoder.layer[0].attention.self
        assert first_attention._ripe_attention_score_dtype == torch.bfloat16
        assert first_attention._ripe_softmax_dtype == torch.float32
        assert torch.isfinite(hidden).all()
        row = {"length": length, "finite": True, "hidden_dtype": str(hidden.dtype),
               "attention_score_dtype": str(first_attention._ripe_attention_score_dtype),
               "softmax_dtype": str(first_attention._ripe_softmax_dtype)}
        if length == min(lengths):
            hidden.float().square().mean().backward()
            gradients = [p.grad for p in ripe_parameters]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            row["ripe_gradient_tensors"] = len(gradients)
        encoder_rows.append(row)
        encoder.zero_grad(set_to_none=True)
        del output, hidden, input_ids, attention_mask
        torch.cuda.empty_cache()
    encoder = encoder.cpu()
    torch.cuda.empty_cache()
    head = FlatDecisionHead(d_model=768, d_z=128, hidden=64, scorer="resnet2d",
                            chunk_size=0, jepa_use_ema=False, ripe_mode="r3").cuda()
    head.train()
    head_rows = []
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    for length in lengths:
        token_ids = torch.arange(length, device="cuda").remainder(4).unsqueeze(0)
        hidden = torch.randn(1, length, 768, device="cuda", generator=generator)
        chunks = [head.pair_repr.cross(hidden, hidden[:, start:start + 128])
                  for start in range(0, length, 128)]
        z = torch.cat(chunks, dim=2)
        if length == min(lengths):
            scores = head.score_pair_matrix(z, token_ids)
        else:
            with torch.no_grad():
                scores = head.score_pair_matrix(z, token_ids)
        assert scores.shape == (1, length, length) and torch.isfinite(scores).all()
        row = {"length": length, "finite": True}
        if length == min(lengths):
            scores.square().mean().backward()
            for name, parameter in head.ripe.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            row["gradient_all_ripe_parameters"] = True
        head_rows.append(row)
        head.zero_grad(set_to_none=True)
        del hidden, z, scores
        torch.cuda.empty_cache()
    controls = {}
    for mode in ("r5", "r6"):
        first = RipePositionalBias(RipeConfig(mode=mode, random_period_seed=0,
                                              random_complement_seed=0))
        second = RipePositionalBias(RipeConfig(mode=mode, random_period_seed=0,
                                               random_complement_seed=0))
        assert all(torch.equal(a, b) for a, b in zip(first.buffers(), second.buffers()))
        controls[mode] = {name: value.tolist() for name, value in first.named_buffers()
                          if name in {"random_periods", "random_complement"}}
    return {"status": "PASS_RIPE_CUDA_SYNTHETIC", "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0), "lengths": lengths,
            "encoder_load": load_report, "encoder": encoder_rows, "head": head_rows,
            "controls": controls,
            "boundary": "synthetic capability gate only; not a benchmark or long-sequence superiority claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[515, 1024])
    args = parser.parse_args()
    result = validate(args.pretrained, args.lengths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
