import torch
from torch import nn

class Channel(nn.Module):
    def __init__(self, seed=0):
        super(Channel, self).__init__()
        self.seed = seed
        self._eval_noise_index = 0

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_eval_noise()
        return self

    def reset_eval_noise(self):
        self._eval_noise_index = 0

    def _noise_like(self, tensor):
        if self.training:
            return torch.randn_like(tensor)

        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(self.seed + self._eval_noise_index)
        self._eval_noise_index += 1
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    def forward(self, data, n_std=0.1):
        if data.get('task_type') != 'VQAv2':
            raise ValueError("当前信道只支持 VQAv2（task_type='VQAv2'）")
        memories = data['memories']
        data['memories'] = memories + self._noise_like(memories) * n_std
        return data
