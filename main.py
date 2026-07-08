"""Single-GPU VQAv2 training entry point."""

import os
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VQAv2
from helper import parse_args
from models.transceiver import Transceiver
from optimizer import BertAdam
from train import train_step


VQA_TASK = "VQAv2"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_optimizer(args, model):
    optimizer_name = args.optims[0]
    learning_rate = args.lrs[0]
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is not None:
        pretrained_parameters = []
        task_parameters = []
        pretrained_prefixes = (
            "semantic_encoder.text_encoder.",
            "semantic_encoder.shared_block.",
        )
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(pretrained_prefixes):
                pretrained_parameters.append(parameter)
            else:
                task_parameters.append(parameter)

        parameter_groups = [
            {
                "params": pretrained_parameters,
                "lr": learning_rate * args.pretrained_lr_scale,
            },
            {"params": task_parameters, "lr": learning_rate},
        ]
        optimizer_kwargs = {"lr": learning_rate}
        if optimizer_name == "AdamW":
            optimizer_kwargs["weight_decay"] = args.weight_decay
        return optimizer_class(parameter_groups, **optimizer_kwargs)
    return BertAdam(
        model.parameters(),
        lr=learning_rate,
        warmup=0.1,
        t_total=1944 * args.epochs,
    )


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def main():
    warnings.filterwarnings("ignore")
    set_seed()
    args = parse_args()
    print(args)

    device = f"cuda:{args.device_id}"
    os.makedirs(args.save_path, exist_ok=True)
    train_set = VQAv2(["train",])
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs[0],
        shuffle=True,
        drop_last=True,
        num_workers=args.nthreads,
        pin_memory=True,
    )

    transceiver = Transceiver(args).to(device)
    optimizer = build_optimizer(args, transceiver)

    for epoch in range(args.epochs):
        start_time = time.time()
        transceiver.train()
        tqdm.write(f"Epoch {epoch} training starts")

        for step, data in enumerate(train_loader):
            data = {key: value.to(device) for key, value in data.items()}
            data.pop("task_id")
            data["task_type"] = VQA_TASK
            loss, metric, metric_value = train_step(data, transceiver, optimizer, device)

            if step % 10 == 0:
                elapsed = time.time() - start_time
                eta = (len(train_loader) - step) / (step + 1) * elapsed
                tqdm.write(
                    f"Epoch {epoch}: [{step}/{len(train_loader)}], task:VQAv2, "
                    f"loss:{loss}, {metric}:{metric_value}, "
                    f"ela:{format_time(elapsed)}, eta:{format_time(eta)}"
                )

            if step % 200 == 0:
                torch.save(transceiver, f"{args.save_path}/epoch_{epoch}.pth")

        tqdm.write(f"Epoch {epoch} training ends")
        torch.save(transceiver, f"{args.save_path}/epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
