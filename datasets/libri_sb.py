import os  
import csv 
from torch.utils.data import Dataset
import torchaudio
from speechbrain.processing.speech_augmentation import SpeedPerturb
import sentencepiece as spm
import torch

class LibriSBDataset(Dataset):
    def __init__(self, meta_path):
        self.meta_infos = []
        with open(meta_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                self.meta_infos.append(row) 
        self.aug = SpeedPerturb(16000, [95, 100, 105])
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('data/LibriSpeech/preprocess/tokenizer.ckpt')
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2
        
    def __getitem__(self, index):
        data_info = self.meta_infos[index]
        audio_path = data_info['wav']
        audio, _ = torchaudio.load(audio_path)
        audio = audio.transpose(0, 1)
        audio = audio.squeeze(1)
        audio = self.aug(audio.unsqueeze(0)).squeeze(0)
        wrd = data_info['wrd']
        label = self.tokenizer.encode_as_ids(wrd)
        label_bos = torch.LongTensor([self.bos_index, ] + label)
        label_eos = torch.LongTensor(label + [self.eos_index, ])
        label = torch.LongTensor(label)
        
        return {'audio': audio, 'label': label, 'label_bos': label_bos, 'label_eos': label_eos,
                'task_id': torch.tensor([11])} 
    
    def __len__(self):
        return len(self.meta_infos)
        

libri_sb_trainset = LibriSBDataset('data/LibriSpeech/preprocess/train.csv')
libri_sb_testset = LibriSBDataset('data/LibriSpeech/preprocess/test-clean.csv')   