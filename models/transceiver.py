import torch.nn as nn
import numpy as np
from .semantic.encoder import SemanticEncoder
from .semantic.decoder import SemanticDecoder
from .channel.channel import Channel
from .channel.encoder import ChannelEncoder
from .channel.decoder import ChannelDecoder


def snr2noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std 

class Transceiver(nn.Module):
    def __init__(self, args):
        super(Transceiver, self).__init__()
        self.semantic_encoder = SemanticEncoder(args)
        semantic_dim = self.semantic_encoder.text_encoder.config.hidden_size
        self.channel_encoder = ChannelEncoder(d_model=semantic_dim)
        self.channel_decoder = ChannelDecoder(d_model=semantic_dim)
        self.semantic_decoder = SemanticDecoder(args)

        self.channel = Channel(seed=getattr(args, "channel_seed", 0))
        self.n_std = snr2noise(args.train_snr)

    def forward(self, data):
        data = self.semantic_encoder(data)
        data = self.channel_encoder(data)
        data = self.channel(data, self.n_std)
        data = self.channel_decoder(data)
        data = self.semantic_decoder(data)
        return data

    def update_std(self, snr):
        self.n_std = snr2noise(snr)
        self.channel.reset_eval_noise()
