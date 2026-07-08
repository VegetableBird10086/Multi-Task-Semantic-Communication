"""VQAv2 answer classifier."""

from torch import nn


VQA_TASK = "VQAv2"
VQA_ANSWER_COUNT = 3129


class SemanticDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Use the task name as the classifier key in newly trained checkpoints.
        self.classifiers = nn.ModuleDict(
            {
                VQA_TASK: nn.Sequential(
                    nn.Linear(768, 768 * 2),
                    nn.ReLU(),
                    nn.Linear(768 * 2, 768),
                    nn.LayerNorm(768),
                    nn.Flatten(1),
                    nn.Linear(768, VQA_ANSWER_COUNT),
                )
            }
        )

    def forward(self, data):
        task_type = data.get("task_type")
        if task_type != VQA_TASK:
            raise ValueError(f"当前版本只支持 VQAv2（task_type={VQA_TASK!r}），收到：{task_type!r}")
        return self.classifiers[VQA_TASK](data["memories"])
