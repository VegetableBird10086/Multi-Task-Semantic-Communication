"""NNI-based VQAv2 hyperparameter search entry point."""

import os

import nni
import torch
from torch.utils.data import DataLoader

from datasets import VQAv2
from eval import eval_step
from helper import parse_args
from main import (
    VQA_TASK,
    build_optimizer,
    set_seed,
)
from models.transceiver import Transceiver
from train import train_step


def main():
    set_seed()
    args = parse_args()
    nni_config = nni.get_next_parameter()
    args.lrs = [nni_config.get("trans_lr", args.lrs[0])]
    args.optims = [nni_config.get("optim", args.optims[0])]

    device = f"cuda:{args.device_id}"
    os.makedirs(args.save_path, exist_ok=True)
    train_set = VQAv2(["train", "nominival"])
    test_set = VQAv2(["minival"])
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs[0],
        shuffle=True,
        drop_last=True,
        num_workers=args.nthreads,
        pin_memory=True,
    )
    test_loader = DataLoader(test_set, batch_size=args.bs[0], shuffle=False)

    transceiver = Transceiver(args).to(device)
    optimizer = build_optimizer(args, transceiver)
    best_metric = 0.0

    for epoch in range(args.epochs):
        transceiver.train()
        for data in train_loader:
            data = {key: value.to(device) for key, value in data.items()}
            data.pop("task_id")
            data["task_type"] = VQA_TASK
            loss, _, _ = train_step(data, transceiver, optimizer, device)
            nni.report_intermediate_result({"training_loss": loss})

        if epoch % args.test_interval == 0:
            transceiver.eval()
            metric = eval_step(transceiver, test_loader, device)
            best_metric = max(best_metric, metric)
            nni.report_intermediate_result(metric)

    nni.report_final_result(best_metric)


if __name__ == "__main__":
    main()
