"""VQAv2 dataset backed by pre-extracted Faster R-CNN region features."""

import base64
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import BertTokenizer


DATA_DIR = "data/vqav2/data"
IMAGE_DIR = os.path.join(DATA_DIR, "img")
VQA_TASK_ID = 10
FILE_MAPPINGS = {
    "train": "train2014",
    "valid": "val2014",
    "minival": "val2014",
    "nominival": "val2014",
    "test": "test2015",
}


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


class VQAv2(Dataset):
    def __init__(self, file_list, require_labels=True):
        self.tokenizer = BertTokenizer(20, 20)
        self.data_infos = []
        for split in file_list:
            self.data_infos.extend(load_json(os.path.join(DATA_DIR, f"{split}.json")))

        self.ans2label = load_json(os.path.join(DATA_DIR, "trainval_ans2label.json"))
        label2ans = load_json(os.path.join(DATA_DIR, "trainval_label2ans.json"))
        if len(self.ans2label) != len(label2ans):
            raise ValueError("VQAv2 answer mappings have different lengths")

        self.offset = {}
        for split in file_list:
            offset_path = os.path.join(
                IMAGE_DIR, f"{FILE_MAPPINGS[split]}_offset.txt"
            )
            with open(offset_path, "r") as file:
                for line in file:
                    image_id, offset = line.rstrip().split("\t")
                    self.offset[image_id] = int(offset)

        self.data = [
            data_info
            for data_info in self.data_infos
            if data_info["img_id"] in self.offset
            and (not require_labels or bool(data_info.get("label")))
        ]

    def _read_image_features(self, image_id):
        image_split = image_id[5:7]
        if image_split == "tr":
            feature_split = "train"
        elif image_split == "va":
            feature_split = "valid"
        else:
            feature_split = "test"

        feature_path = os.path.join(
            IMAGE_DIR, f"{FILE_MAPPINGS[feature_split]}_obj36.tsv"
        )
        with open(feature_path, "r") as file:
            file.seek(self.offset[image_id])
            fields = file.readline().rstrip("\n").split("\t")

        if fields[0] != image_id:
            raise ValueError(f"TSV offset mismatch: expected {image_id}, got {fields[0]}")
        features = np.frombuffer(base64.b64decode(fields[-1]), dtype=np.float32)
        return torch.from_numpy(features.reshape(36, -1).copy())

    def __getitem__(self, index):
        data_info = self.data[index]
        label = torch.zeros(len(self.ans2label), dtype=torch.float32)
        for answer, score in data_info["label"].items():
            label[self.ans2label[answer]] = score

        data = {
            "image": self._read_image_features(data_info["img_id"]),
            "task_id": torch.tensor([VQA_TASK_ID]),
            "label": label,
        }
        data.update(self.tokenizer(data_info["sent"]))
        return data

    def __len__(self):
        return len(self.data)


class VQAv2Test(VQAv2):
    def __init__(self, file_list):
        super().__init__(file_list, require_labels=False)

    def __getitem__(self, index):
        data_info = self.data[index]
        data = {
            "image": self._read_image_features(data_info["img_id"]),
            "task_id": torch.tensor([VQA_TASK_ID]),
            "qid": data_info["question_id"],
        }
        data.update(self.tokenizer(data_info["sent"]))
        return data
