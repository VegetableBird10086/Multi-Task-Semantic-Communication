'''
Author: LOTEAT
Date: 2023-05-31 16:34:26
'''
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .tokenizer import BertTokenizer



class EuroparlDataset(Dataset):
    def __init__(self, path='data/europarl/train.txt'):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip() for line in lines]
        self.tokenizer = BertTokenizer()
        
    def __getitem__(self, index):
        data = self.tokenizer(self.data[index])
        data.update({'label': data['text'].clone(), 'task_id': torch.tensor([8])})
        return data

    def __len__(self):
        return len(self.data)
    
europarl_trainset = EuroparlDataset('data/europarl/train.txt')
europarl_testset = EuroparlDataset('data/europarl/test.txt')
europarl_realset = europarl_testset