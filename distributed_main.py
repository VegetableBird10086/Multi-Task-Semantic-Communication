"""Distributed VQAv2 training entry point."""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets import VQAv2
from helper import parse_args
from main import (
    VQA_TASK,
    build_optimizer,
    format_time,
    set_seed,
)
from models.transceiver import Transceiver
from train import train_step


def main():
    set_seed()
    args = parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    train_set = VQAv2(["train", "nominival"])
    sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs[0],
        sampler=sampler,
        drop_last=True,
        num_workers=args.nthreads,
        pin_memory=True,
    )

    transceiver = Transceiver(args).to(device)
    optimizer = build_optimizer(args, transceiver)
    transceiver = DistributedDataParallel(
        transceiver, device_ids=[local_rank], find_unused_parameters=False
    )

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print(args)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        transceiver.train()
        start_time = time.time()

        for step, data in enumerate(train_loader):
            data = {key: value.to(device) for key, value in data.items()}
            data.pop("task_id")
            data["task_type"] = VQA_TASK
            loss, metric, metric_value = train_step(data, transceiver, optimizer, device)

            if rank == 0 and step % 10 == 0:
                elapsed = time.time() - start_time
                eta = (len(train_loader) - step) / (step + 1) * elapsed
                tqdm.write(
                    f"Epoch {epoch}: [{step}/{len(train_loader)}], task:VQAv2, "
                    f"loss:{loss}, {metric}:{metric_value}, "
                    f"ela:{format_time(elapsed)}, eta:{format_time(eta)}"
                )

        if rank == 0:
            torch.save(transceiver.module, f"{args.save_path}/epoch_{epoch}.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
