"""Evaluate a checkpoint on the VQAv2 minival split."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VQAv2
from helper import parse_args


VQA_TASK = "VQAv2"
EVAL_SNR = 15


def eval_step(transceiver, test_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = {key: value.to(device) for key, value in data.items()}
            data.pop("task_id")
            data["task_type"] = VQA_TASK

            predictions = transceiver(data)
            predicted_answers = torch.argmax(predictions, dim=1)
            predicted_label_scores = data["label"].gather(
                1, predicted_answers.unsqueeze(1)
            )
            correct += (predicted_label_scores > 0).sum().item()
            total += predicted_answers.numel()

    accuracy = correct / total
    tqdm.write(f"Task: VQAv2 accuracy: {accuracy:.4f}")
    return accuracy


def main():
    args = parse_args()
    dataset = VQAv2(["minival"])
    test_loader = DataLoader(dataset, batch_size=args.bs[0], shuffle=False)
    device = f"cuda:{args.device_id}"

    transceiver = torch.load(args.load_from, map_location="cpu")
    transceiver = transceiver.to(device)
    transceiver.update_std(EVAL_SNR)
    transceiver.eval()
    eval_step(transceiver, test_loader, device)


if __name__ == "__main__":
    main()
