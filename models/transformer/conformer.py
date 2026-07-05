import torch
from torch import nn

from .conformer_encoder import ConformerEncoder
from .conv_samp import ConvSubsampling


class ConformerModel(nn.Module):
    def __init__(self, hidden_size, mel_filter_size, max_source_positions, intermediate_size, num_attention_heads, conv_depth_wise_kernel_size, num_hidden_layers):
        super().__init__()
        self.subsampling_conv = ConvSubsampling(hidden_size)
        self.encoder = ConformerEncoder(hidden_size, mel_filter_size, max_source_positions, intermediate_size, num_attention_heads, conv_depth_wise_kernel_size, num_hidden_layers)

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            torch.Tensor with shape `(B, L, D)`
            torch.Tensor with shape `(B)`
            )
        """
        hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)

        hidden_states = self.encoder(hidden_states)

        return hidden_states, input_lengths
