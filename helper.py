"""Command-line options shared by the VQAv2 entry points."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", default=5, type=int)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["VQAv2"],
        default=["VQAv2"],
        help="当前版本仅支持 VQAv2（VQAv2）",
    )
    parser.add_argument("--device_id", default=0, type=int, help="GPU id")
    parser.add_argument("--lrs", default=[1e-4], type=float, nargs="+")
    parser.add_argument("--optims", default=["AdamW"], type=str, nargs="+")
    parser.add_argument("--bs", default=[32], type=int, nargs="+")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--save_path", default="./checkpoint", type=str)
    parser.add_argument("--load_from", type=str, help="checkpoint path")
    parser.add_argument("--train_snr", default=15, type=int)
    parser.add_argument("--pretrained_lr_scale", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--channel_seed", default=0, type=int)
    parser.add_argument("--test_interval", default=1, type=int)
    return parser.parse_args()
