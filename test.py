"""Generate VQAv2 answer predictions from a checkpoint."""

import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VQAv2Test
from helper import parse_args


VQA_TASK = "VQAv2"


def load_label_map():
    path = "data/vqav2/data/trainval_label2ans.json"
    with open(path, "r") as file:
        return json.load(file)


def test_step(transceiver, test_loader, device, snr):
    label2ans = load_label_map()
    results = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = {key: value.to(device) for key, value in data.items()}
            data.pop("task_id")
            data["task_type"] = VQA_TASK

            predictions = transceiver(data)
            predicted_answers = torch.argmax(predictions, dim=1).cpu().tolist()
            question_ids = data["qid"].cpu().tolist()
            for answer_index, question_id in zip(predicted_answers, question_ids):
                answer = (
                    label2ans[str(answer_index)]
                    if isinstance(label2ans, dict)
                    else label2ans[answer_index]
                )
                results.append(
                    {
                        "answer": answer,
                        "question_id": int(question_id),
                    }
                )

    os.makedirs("result", exist_ok=True)
    output_path = f"result/VQAv2_snr_{snr}.json"
    with open(output_path, "w") as file:
        json.dump(results, file)
    tqdm.write(f"VQAv2 predictions written to {output_path}")


def main():
    args = parse_args()
    dataset = VQAv2Test(["nominival"])
    test_loader = DataLoader(dataset, batch_size=args.bs[0], shuffle=False)
    device = f"cuda:{args.device_id}"

    transceiver = torch.load(args.load_from, map_location="cpu")
    transceiver = transceiver.to(device)
    transceiver.eval()

    for snr in range(-6, 21, 3):
        transceiver.update_std(snr)
        test_step(transceiver, test_loader, device, snr)


if __name__ == "__main__":
    main()
