from torch import nn, Tensor 
import torch
from typing import Tuple

class MaskCNN(nn.Module): 
    """
    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = inputs
        for module in self.sequential:
            output = module(output)
            mask = torch.BoolTensor(output.size()).fill_(0).to(output.device)

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)
            
            output = output.masked_fill(mask, 0)

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        if isinstance(module, nn.Conv2d):
            
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()

class MaskVGG(nn.Module):
    def __init__(self, input_dim, in_channels=1, out_channels=(64, 128), activation='relu'):
        super(MaskVGG, self).__init__()
        act_fn = {
            'hardtanh': nn.Hardtanh(0, 20, inplace=True),
            'relu': nn.ReLU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU()
        }
        self.input_dim = input_dim
        act_fn = act_fn[activation]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                act_fn,
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(num_features=out_channels[0]),
                act_fn,
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(num_features=out_channels[1]),
                act_fn,
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                act_fn,
                nn.MaxPool2d(2, stride=2),
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)

        return outputs, output_lengths
    
    def get_output_dim(self):
        output_dim = (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5
        return output_dim
    
    def get_output_lengths(self, seq_lengths: Tensor):
        assert self.conv is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1

        return seq_lengths.int()
