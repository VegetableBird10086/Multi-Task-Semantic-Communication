"""VQAv2 semantic encoder.

The project previously constructed image, speech, video, and text-task backbones
unconditionally.  VQAv2 only needs a BERT question encoder and the visual-BERT
fusion block, because region features are read from the Faster R-CNN TSV files.
"""

from pathlib import Path

import torch
from torch import nn
from transformers.models.auto.configuration_auto import AutoConfig

from .shared_block2 import VBFeatureExtraction
from .text_encoder import BertModel


VQA_TASK = "VQAv2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BERT_MODEL_DIR = PROJECT_ROOT / "pretrained" / "bert-base-uncased"
REQUIRED_BERT_FILES = ("config.json", "pytorch_model.bin")
TEXT_HIDDEN_LAYERS = 2


class SemanticEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        missing_files = [
            name for name in REQUIRED_BERT_FILES if not (BERT_MODEL_DIR / name).is_file()
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"本地 BERT 权重不完整：{BERT_MODEL_DIR} 缺少 {missing}。"
                "请按 README 的预训练权重下载步骤准备文件。"
            )

        model_dir = str(BERT_MODEL_DIR)
        bert_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        bert_config.num_hidden_layers = TEXT_HIDDEN_LAYERS
        self.text_encoder = BertModel.from_pretrained(
            model_dir,
            config=bert_config,
            add_pooling_layer=False,
            local_files_only=True,
        )
        self.token_embeddings = nn.Linear(883, 1024)
        self.shared_block = VBFeatureExtraction.from_pretrained(model_dir)

    def forward(self, data):
        task_type = data.get("task_type")
        if task_type != VQA_TASK:
            raise ValueError(f"当前版本只支持 VQAv2（task_type={VQA_TASK!r}），收到：{task_type!r}")

        text = data["text"]
        token_type_ids = data["segment_ids"]
        attention_mask = data["text_mask"]
        position_ids = torch.arange(text.size(1), dtype=torch.long, device=text.device)
        text_embedding = self.text_encoder.embeddings(
            input_ids=text,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e5
        head_mask = [None] * self.text_encoder.config.num_hidden_layers
        encoder_outputs = self.text_encoder.encoder(
            text_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )[0]

        batch_size, visual_tokens = data["image"].shape[:2]
        visual_segment_ids = torch.ones(
            batch_size, visual_tokens, dtype=torch.long, device=text.device
        )
        visual_attention_mask = torch.ones_like(visual_segment_ids)
        data["memories"] = self.shared_block(
            input_ids=encoder_outputs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_feats=data["image"],
            visual_token_type_ids=visual_segment_ids,
            visual_attention_mask=visual_attention_mask,
        )
        return data
