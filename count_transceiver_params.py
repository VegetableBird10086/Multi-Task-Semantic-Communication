#!/usr/bin/env python3
"""Count parameters in the project's Transceiver model."""

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from models.transceiver import Transceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 Transceiver 的总参数量。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="可选：训练保存的完整 .pth 模型，或包含 state_dict 的检查点。",
    )
    parser.add_argument("--train-snr", type=int, default=3)
    return parser.parse_args()


def model_args(args: argparse.Namespace) -> Namespace:
    """Build the subset of training arguments required by Transceiver.__init__."""
    return Namespace(
        train_snr=args.train_snr,
    )


def build_model(args: argparse.Namespace) -> nn.Module:
    return Transceiver(model_args(args))


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("检查点既不是 nn.Module，也不是字典。")

    for key in ("state_dict", "model_state_dict", "transceiver_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise TypeError("检查点中未找到 state_dict。")


def remove_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(name.startswith("module.") for name in state_dict):
        return {name[len("module.") :]: value for name, value in state_dict.items()}
    return state_dict


def load_model(args: argparse.Namespace) -> nn.Module:
    if args.checkpoint is None:
        return build_model(args)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.module if hasattr(checkpoint, "module") else checkpoint

    if isinstance(checkpoint, dict):
        for key in ("model", "transceiver"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, nn.Module):
                return candidate.module if hasattr(candidate, "module") else candidate

    model = build_model(args)
    state_dict = remove_ddp_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return model


def main() -> None:
    args = parse_args()
    model = load_model(args)
    encoder_total = sum(
        parameter.numel() for parameter in model.semantic_encoder.parameters()
    )
    decoder_total = sum(
        parameter.numel() for parameter in model.semantic_decoder.parameters()
    )
    channel_encoder_total = sum(
        parameter.numel() for parameter in model.channel_encoder.parameters()
    )
    channel_decoder_total = sum(
        parameter.numel() for parameter in model.channel_decoder.parameters()
    )
    total = sum(parameter.numel() for parameter in model.parameters())

    print(f"{'组件':<24} {'总参数量':>14}")
    print("-" * 40)
    print(f"{'semantic_encoder':<24} {encoder_total / 1_000_000:>11.2f} M")
    print(f"{'channel_encoder':<24} {channel_encoder_total / 1_000_000:>11.2f} M")
    print(f"{'channel_decoder':<24} {channel_decoder_total / 1_000_000:>11.2f} M")
    print(f"{'semantic_decoder':<24} {decoder_total / 1_000_000:>11.2f} M")
    print("-" * 40)
    print(f"{'Transceiver':<24} {total / 1_000_000:>11.2f} M")


if __name__ == "__main__":
    main()
