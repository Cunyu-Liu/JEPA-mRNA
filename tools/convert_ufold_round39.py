#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.official_repo))
    from Network import U_Net

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    original_model = U_Net(img_ch=17)
    original_model.load_state_dict(source, strict=True)
    from transformers.activations import ACT2FN
    ACT2FN["elu"] = torch.nn.ELU
    from multimolecule import UfoldConfig, UfoldModel
    model = UfoldModel(UfoldConfig())
    target = model.state_dict()
    assert len(source) == len(target) == 156
    converted = {}
    for source_key, source_tensor in source.items():
        parts = source_key.split(".")
        if parts[0].startswith("Conv") and parts[0] != "Conv_1x1":
            index = int(parts[0][4:]) - 1
            suffix = {"0": "conv1", "1": "batch_norm1", "3": "conv2", "4": "batch_norm2"}[parts[2]]
            target_key = f"encoder.down_blocks.{index}.{suffix}.{parts[3]}"
        elif parts[0].startswith("Up_conv"):
            index = 5 - int(parts[0][7:])
            suffix = {"0": "conv1", "1": "batch_norm1", "3": "conv2", "4": "batch_norm2"}[parts[2]]
            target_key = f"encoder.decoder_blocks.{index}.{suffix}.{parts[3]}"
        elif parts[0].startswith("Up"):
            index = 5 - int(parts[0][2:])
            suffix = {"1": "conv", "2": "batch_norm"}[parts[2]]
            target_key = f"encoder.up_blocks.{index}.{suffix}.{parts[3]}"
        elif parts[0] == "Conv_1x1":
            target_key = f"encoder.prediction.{parts[1]}"
        else:
            raise ValueError(source_key)
        assert target_key in target, (source_key, target_key)
        assert source_tensor.shape == target[target_key].shape, (source_key, target_key, source_tensor.shape, target[target_key].shape)
        assert torch.isfinite(source_tensor).all(), source_key
        tensor = source_tensor.detach().clone()
        if source_key == "Conv1.conv.0.weight":
            source_bases, target_bases = "AUCG", "ACGU"
            permutation = [source_bases.index(a) * 4 + source_bases.index(b)
                           for a in target_bases for b in target_bases]
            tensor = torch.cat((tensor[:, permutation], tensor[:, 16:]), dim=1)
        converted[target_key] = tensor
    assert set(converted) == set(target)
    model.load_state_dict(converted, strict=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file(converted, args.out)
    else:
        torch.save(converted, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"UFOLD_CONVERSION_PASS tensors={len(converted)} sha256={digest} out={args.out}")


if __name__ == "__main__":
    main()
