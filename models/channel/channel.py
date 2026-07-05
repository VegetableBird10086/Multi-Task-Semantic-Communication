from torch import nn
import torch
import math

class Channel(nn.Module):
    def __init__(self):
        super(Channel, self).__init__()
    
    def forward(self, data, n_std=0.1):
        task_type = data['task_type']
        
        if task_type in ['fastvqa', 'mmhs', 'mmdb']:
            for data_type in ['memories']:
                if data_type in data.keys():
                    data[data_type] = data[data_type] + torch.randn_like(data[data_type]) * n_std
        else:
            for data_type in ['text', 'audio', 'video', 'image', 'memories']:
                if data_type in data.keys():
                    data[data_type] = data[data_type] + torch.randn_like(data[data_type]) * n_std
        return data