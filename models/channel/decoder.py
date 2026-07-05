from torch import nn
from ..norm.layer_norm import LayerNorm

class ChannelDecoder(nn.Module):
    def __init__(self, d_model):
        super(ChannelDecoder, self).__init__()
        self.dense1 = nn.Linear(32, d_model)
        self.ac_fun1 = nn.ReLU()
        self.dense2 = nn.Linear(d_model, 512)
        self.ac_fun2 = nn.ReLU()
        self.dense3 = nn.Linear(512, d_model)
        self.layernorm1 = LayerNorm(d_model)

    def forward(self, data):
        task_type = data['task_type']
        
        if task_type in ['fastvqa', 'mmhs', 'mmdb']:
            for data_type in ['memories']:
                if data_type in data.keys():
                    x = data[data_type]
                    x1 = self.dense1(x)
                    x1 = self.ac_fun1(x1)
                    x2 = self.dense2(x1)
                    x2 = self.ac_fun2(x2)
                    x3 = self.dense3(x2)
                    output = self.layernorm1(x1 + x3)
                    data[data_type] = output
        else:
            for data_type in ['text', 'audio', 'video', 'image', 'memories']:
                if data_type in data.keys():
                    x = data[data_type]
                    x1 = self.dense1(x)
                    x1 = self.ac_fun1(x1)
                    x2 = self.dense2(x1)
                    x2 = self.ac_fun2(x2)
                    x3 = self.dense3(x2)
                    output = self.layernorm1(x1 + x3)
                    data[data_type] = output
        return data


# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.dense1 = nn.Linear(64, 256)
#         self.dense2 = nn.Linear(256, 768)
#         self.layernorm1 = LayerNorm(768)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     x1 = self.dense1(x)
#                     x1 = self.dense2(x1)
#                     output = self.layernorm1(x1)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     x1 = self.dense1(x)
#                     x1 = self.dense2(x1)
#                     output = self.layernorm1(x1)
#                     data[data_type] = output
#         return data

# Decoder 1
# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.layernorm1 = LayerNorm(128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         return data


# decoder 2 
# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.layernorm1 = LayerNorm(128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         return data

# decoder 3
# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.layernorm1 = LayerNorm(64)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         return data

# decoder 4 
# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.layernorm1 = LayerNorm(128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         return data


# decoder 5
# class ChannelDecoder(nn.Module):
#     def __init__(self, d_model):
#         super(ChannelDecoder, self).__init__()
#         self.layernorm1 = LayerNorm(128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['fastvqa']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         else:
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     x = data[data_type]
#                     # x1 = self.dense1(x)
#                     # x1 = self.ac1(x1)
#                     # x1 = self.dense2(x1)
#                     output = self.layernorm1(x)
#                     data[data_type] = output
#         return data