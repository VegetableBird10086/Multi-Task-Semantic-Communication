from torch import nn
from ..norm.power_norm import PowerNorm


class ChannelEncoder(nn.Module):
    def __init__(self, d_model=128):
        super(ChannelEncoder, self).__init__()
        self.dense0 = nn.Linear(d_model, 256)
        self.ac_fun1 = nn.ReLU()
        self.dense1 = nn.Linear(256, 32)
        self.powernorm = PowerNorm()

    def forward(self, data):
        task_type = data['task_type']
        
        if task_type in ['VQAv2', 'mmhs', 'mmdb']:
            for data_type in ['memories']:
                if data_type in data.keys():
                    out = data[data_type]  
                    out = self.dense0(out)
                    out = self.ac_fun1(out)
                    out = self.dense1(out)
                    out = self.powernorm(out)
                    data[data_type] = out
        else: 
            for data_type in ['text', 'audio', 'video', 'image', 'memories']:
                if data_type in data.keys():
                    out = data[data_type]  
                    out = self.dense0(out)
                    out = self.ac_fun1(out)
                    out = self.dense1(out)
                    out = self.powernorm(out)
                    data[data_type] = out
        return data


# Encoder 1
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 768 * 2)
#         self.ac1 = nn.ReLU()
#         self.dense1 = nn.Linear(768 * 2, 768)
#         self.ac2 = nn.ReLU()
#         self.dense2 = nn.Linear(768, 128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     out = self.dense1(out)
#                     out = self.ac2(out)
#                     out = self.dense2(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     out = self.dense1(out)
#                     out = self.ac2(out)
#                     data[data_type] = out
#         return data

# encoder 2
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 128)
#         self.ac1 = nn.ReLU()
#         self.dense1 = nn.Linear(128, 256)
#         self.ac2 = nn.ReLU()
#         self.dense2 = nn.Linear(256, 128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     out = self.dense1(out)
#                     out = self.ac2(out)
#                     out = self.dense2(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     out = self.dense1(out)
#                     out = self.ac2(out)
#                     data[data_type] = out
#         return data

# encoder 3 
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 128)
#         self.ac1 = nn.ReLU()
#         self.dense2 = nn.Linear(128, 64)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     out = self.dense2(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac1(out)
#                     data[data_type] = out
#         return data

# encoder 4
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 128)

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     data[data_type] = out
#         return data


# encoder 5
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 768 * 2)
#         self.ac0 = nn.ReLU()
#         self.dense1 = nn.Linear(768 * 2, 128)
#         self.ac1 = nn.ReLU()

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac0(out)
#                     out = self.dense1(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     data[data_type] = out
#         return data

# encoder 6
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 768 * 2)
#         self.ac0 = nn.ReLU()
#         self.dense1 = nn.Linear(768 * 2, 128)
#         self.ac1 = nn.ReLU()

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac0(out)
#                     out = self.dense1(out)
#                     out = self.ac1(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     data[data_type] = out
#         return data

# encoder 7 
# class ChannelEncoder(nn.Module):
#     def __init__(self, d_model=128):
#         super(ChannelEncoder, self).__init__()
#         self.dense0 = nn.Linear(768, 256)
#         self.ac0 = nn.ReLU()
#         self.dense1 = nn.Linear(256, 128)
#         self.ac1 = nn.ReLU()

#     def forward(self, data):
#         task_type = data['task_type']
        
#         if task_type in ['VQAv2']:
#             for data_type in ['memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     out = self.ac0(out)
#                     out = self.dense1(out)
#                     out = self.ac1(out)
#                     data[data_type] = out
#         else: 
#             for data_type in ['text', 'audio', 'video', 'image', 'memories']:
#                 if data_type in data.keys():
#                     out = data[data_type]  
#                     out = self.dense0(out)
#                     data[data_type] = out
#         return data