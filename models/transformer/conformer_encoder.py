import torch
from torch import nn
import math
from typing import Optional

from .conformer_block import ConformerBlock
from .attention import PositionalEncoder


class ConformerEncoder(nn.Module):
    def __init__(self, hidden_size, mel_filter_size, max_source_positions, intermediate_size, num_attention_heads, conv_depth_wise_kernel_size, num_hidden_layers):
        super().__init__()

        self.linear = nn.Linear(hidden_size * (((mel_filter_size - 1) // 2 - 1) // 2), hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.positional_encoder = PositionalEncoder(max_source_positions, hidden_size)
        self.layers = nn.ModuleList([ConformerBlock(hidden_size, intermediate_size, num_attention_heads, conv_depth_wise_kernel_size) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape `(B, L, D)`
        """
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)

        position_embeddings = self.positional_encoder(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )

        return hidden_states
