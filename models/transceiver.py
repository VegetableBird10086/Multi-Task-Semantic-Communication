import torch.nn as nn
import numpy as np
from .semantic.encoder import SemanticEncoder
from .semantic.decoder import SemanticDecoder
from .channel.encoder import ChannelEncoder
from .channel.decoder import  ChannelDecoder
from .channel.channel import Channel
import torch.nn.init as init


def snr2noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std 

def channel_init(named_parameters):
    for name, param in named_parameters:
        if 'weight' in name:
            init.xavier_uniform_(param)  
        elif 'bias' in name:
            nn.init.zeros_(param)  
class Transceiver(nn.Module):
    def __init__(self, args):
        super(Transceiver, self).__init__()
        self.semantic_encoder = SemanticEncoder(args)

        self.semantic_decoder = SemanticDecoder(args)

        self.channel_encoder = ChannelEncoder(args.enc_d_model)
        self.channel_decoder = ChannelDecoder(args.dec_d_model)

        self.channel_encoder.apply(self.init_weights)
        self.channel_decoder.apply(self.init_weights)


        self.channel = Channel()
        self.n_std = snr2noise(args.train_snr)
        # self.semantic_decoder.classifiers['fastvqa'].apply(self.semantic_encoder.shared_block.init_bert_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias) 

    def forward(self, data):
        data = self.semantic_encoder(data)
        # data = self.channel_encoder(data)
        data = self.channel(data, self.n_std)
        # data = self.channel_decoder(data)
        data = self.semantic_decoder(data)
        return data

    def update_std(self, snr):
        self.n_std = snr2noise(snr)