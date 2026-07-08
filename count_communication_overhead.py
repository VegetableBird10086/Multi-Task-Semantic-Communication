#!/usr/bin/env python3
"""Calculate the communication cost of the VQAv2 semantic representation."""

import argparse

from models.channel.encoder import ChannelEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="统计语义特征的通信开销。")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="一次传输的样本数，默认：1。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    channel_encoder = ChannelEncoder(d_model=768)
    feature_dim = channel_encoder.dense1.out_features
    bytes_per_float32 = 4
    bytes_per_sample = feature_dim * bytes_per_float32
    batch_bytes = bytes_per_sample * args.batch_size

    print(f"传输张量形状：[{args.batch_size}, {feature_dim}]")
    print(f"数据类型：float32（{bytes_per_float32} Byte/特征值）")
    print(
        f"每样本通信开销：{feature_dim} × {bytes_per_float32} "
        f"= {bytes_per_sample:,} Byte "
        f"= {bytes_per_sample / 1000:.3f} KB "
    )
    if args.batch_size > 1:
        print(
            f"整个 batch 通信开销：{batch_bytes:,} Byte "
            f"= {batch_bytes / 1024:.3f} KiB "
        )


if __name__ == "__main__":
    main()
