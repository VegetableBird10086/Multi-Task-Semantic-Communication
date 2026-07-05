import pandas as pd
import torch
from torch.utils.data import Dataset
import os.path as osp
from .tokenizer import BertTokenizer

class SST2Dataset(Dataset):
    def __init__(self, mode):
        self.data = pd.read_csv(osp.join('data/glue_data/SST-2', '%s.tsv' % mode), sep='\t')
        self.tokenizer = BertTokenizer()

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['sentence']
        label = self.data.iloc[index]['label']
        data = self.tokenizer(sentence)
        data['label'] = torch.tensor(label, dtype=torch.long)
        data['task_id'] = torch.tensor([1])
        return data

    def __len__(self):
        return len(self.data)

glue_sst2_trainset = SST2Dataset('train') 
glue_sst2_testset = SST2Dataset('dev')


class SST2TestDataset(Dataset):
    def __init__(self, mode):
        self.data = pd.read_csv(osp.join('data/glue_data/SST-2', '%s.tsv' % mode), sep='\t')
        self.tokenizer = BertTokenizer()

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['sentence']
        index = self.data.iloc[index]['index']
        data = self.tokenizer(sentence)
        data['index'] = torch.tensor(index, dtype=torch.long)
        data['task_id'] = torch.tensor([1])
        return data

    def __len__(self):
        return len(self.data)


glue_sst2_realset = SST2TestDataset('test')


class CoLADataset(Dataset):
    def __init__(self, mode):
        self.data = pd.read_csv(osp.join('data/glue_data/CoLA', '%s.tsv' % mode), sep='\t')
        self.tokenizer = BertTokenizer()

    def __getitem__(self, index):
        sentence = self.data.iloc[index, 3]
        label = self.data.iloc[index, 1]
        data = self.tokenizer(sentence)
        data['label'] = torch.tensor(label, dtype=torch.long)
        data['task_id'] = torch.tensor([3])
        return data

    def __len__(self):
        return len(self.data)

glue_cola_trainset = CoLADataset('train')
glue_cola_testset = CoLADataset('dev')