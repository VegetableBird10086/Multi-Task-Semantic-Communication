"""VQAv2 training step."""

import torch
from torch import nn

from metric_meter import MetricMeter


VQA_TASK = "VQAv2"
loss_meter = MetricMeter()
accuracy_meter = MetricMeter()


def multi_positive_loss(logits, labels):
    """Maximize probability assigned to any annotated answer."""
    valid_answers = labels > 0
    if not torch.all(valid_answers.any(dim=1)):
        raise ValueError("Every training sample must have at least one valid answer")

    valid_logits = logits.masked_fill(~valid_answers, float("-inf"))
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(valid_logits, dim=1)
    ).mean()


def train_step(data, transceiver, optimizer, device):
    task_type = data.get("task_type")
    if task_type != VQA_TASK:
        raise ValueError(f"当前版本只支持 VQAv2，收到 task_type={task_type!r}")

    optimizer.zero_grad()
    predictions = transceiver(data)
    loss = multi_positive_loss(predictions, data["label"])
    loss.backward()
    nn.utils.clip_grad_norm_(transceiver.parameters(), 5.0)
    optimizer.step()

    predicted_answers = torch.argmax(predictions, dim=1)
    predicted_label_scores = data["label"].gather(
        1, predicted_answers.unsqueeze(1)
    )
    accuracy = (predicted_label_scores > 0).float().mean().item()

    loss_meter.update(loss.item())
    accuracy_meter.update(accuracy)
    return loss_meter.get_ema_metric(), "Accuracy", accuracy_meter.get_ema_metric()
