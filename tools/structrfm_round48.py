import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_structrfm_encoder(pretrained_path, device="cpu", dtype=torch.float32):
    path = Path(pretrained_path).resolve()
    model = AutoModel.from_pretrained(str(path), torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    weights = load_file(str(path / "model.safetensors"))
    model_keys = set(model.state_dict())
    checkpoint_keys = {key[5:] for key in weights if key.startswith("bert.")}
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    allowed_missing = {"embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"structRFM strict identity failed: missing={missing}, unexpected={unexpected}")
    model.to(device)
    report = {
        "backbone": "structRFM",
        "source": str(path),
        "weights_sha256": file_sha256(path / "model.safetensors"),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "d_model": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "checkpoint_bert_tensors": len(checkpoint_keys),
        "model_tensors": len(model_keys),
        "allowed_missing": sorted(set(missing) & allowed_missing),
        "unexpected": unexpected,
        "strict_identity": True,
        "resolved_device": str(next(model.parameters()).device),
    }
    if device != "cpu" and not report["resolved_device"].startswith("cuda"):
        raise RuntimeError("structRFM CPU fallback detected")
    return model, tokenizer, report


def install_structrfm_training(base, xens2_eval, pretrained_path):
    _model, tokenizer, _report = load_structrfm_encoder(pretrained_path)
    vocab = {
        "<cls>": tokenizer.cls_token_id,
        "<eos>": tokenizer.sep_token_id,
        "<unk>": tokenizer.unk_token_id,
    }
    for letter in "ACGUTN":
        token = "U" if letter == "T" else letter
        vocab[letter] = tokenizer.convert_tokens_to_ids(token)
    base.VOCAB = vocab
    base.ENCODER_SPECIAL_TOKENS = False
    base.load_encoder = lambda device, dtype: tuple(
        load_structrfm_encoder(pretrained_path, device=device, dtype=dtype)[::2])

    def scores(encoder, head, sequence, device):
        if len(sequence) > encoder.config.max_position_embeddings:
            raise ValueError(f"sequence length {len(sequence)} exceeds strict structRFM boundary")
        inputs = tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state.float().half().float()
        head_ids = torch.tensor(
            [[base.BASE_TO_INDEX.get(char, base.UNK_INDEX) for char in sequence]],
            device=device)
        scores_ = head.pair_scores_and_types(hidden)[0][0]
        if head.use_turner_prior:
            prior = base.turner_phys_scores(head_ids)[0].to(device)
            scores_ = scores_ + head.prior_weight.to(scores_.dtype) * prior
        lengths = torch.tensor([len(sequence)], device=device)
        scores_ = head.calibration(scores_.unsqueeze(0), lengths)[0]
        mask = base.pair_mask_from_ids(head_ids, head.min_loop).to(device)[0]
        return torch.where(mask, scores_, torch.full_like(scores_, float("-inf")))

    xens2_eval.plana_scores = scores
    return tokenizer


def preflight(pretrained_path, output_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for structRFM preflight")
    model, tokenizer, report = load_structrfm_encoder(pretrained_path, device="cuda")
    sequence = "GUCCCAACUCUUGCGGGGAGGGAU"
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    model.train()
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = output.last_hidden_state
    if hidden.shape != (1, len(sequence), 768):
        raise RuntimeError(f"unexpected hidden shape {hidden.shape}")
    hidden.square().mean().backward()
    grad_tensors = sum(parameter.grad is not None for parameter in model.parameters())
    if grad_tensors == 0:
        raise RuntimeError("no structRFM gradients")
    try:
        too_long = tokenizer("A" * (model.config.max_position_embeddings + 1), return_tensors="pt")
        model(input_ids=too_long["input_ids"].cuda(),
              attention_mask=too_long["attention_mask"].cuda())
    except (IndexError, RuntimeError, ValueError) as exc:
        boundary = {"status": "PASS_EXPLICIT_REJECT", "error": str(exc)[:500]}
    else:
        raise RuntimeError("silent long-sequence processing is forbidden")
    report.update({
        "status": "PASS_CUDA_PREFLIGHT",
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "input_shape": list(input_ids.shape),
        "hidden_shape": list(hidden.shape),
        "gradient_tensors": grad_tensors,
        "long_sequence_boundary": boundary,
    })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preflight(args.pretrained, args.output)
