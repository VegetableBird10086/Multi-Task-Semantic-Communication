from .tokenizer import BertTokenizer
import torch
from torch.utils.data import Dataset
import torch 
from glob import glob 
import os
import numpy as np
import torchaudio
from python_speech_features import delta, mfcc, fbank
from sklearn.preprocessing import MinMaxScaler
from .libri_helper import AudioProcessor

def feature_normalization(feature):
    minMaxScaler = MinMaxScaler()
    return minMaxScaler.fit_transform(feature)


def feature_reshape(feature, max_len=1500):
    x, y, z = feature.shape
    if y >= max_len:
        return feature[:, :max_len, :]
    else:
        reshaped_feature = torch.zeros((x, max_len, z))
        reshaped_feature[:, :y, :] = feature
        return reshaped_feature

class LibriDataset(Dataset):
    def __init__(self, transcript, feature_type='fbank'):
        super().__init__()
        self.feature_type = feature_type
        self.audio_paths = []
        self.labels = []
        self.processor = AudioProcessor()

        with open(transcript, 'r') as f:
            lines = f.readlines()
        for each_line in lines:
            each_line = each_line.strip().split('\t')
            self.audio_paths.append(os.path.join('data/LibriSpeech', each_line[0]))
            self.labels.append([int(each_word) + 1 for each_word in each_line[2].split(' ')])   
    
        self.label_lengths = [len(label) for label in self.labels]
        # self.labels = [torch.LongTensor(seq) for seq in self.labels]
        self.labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in self.labels], batch_first=True, padding_value=0)
        
                 
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        log_mel_spectrogram = self.processor(waveform, sample_rate)
        length = 1500 if log_mel_spectrogram.shape[1] > 1500 else log_mel_spectrogram.shape[1] 
        log_mel_spectrogram = feature_reshape(log_mel_spectrogram)
        label = self.labels[index]
        label_bos = torch.zeros(label.shape[0] + 1)
        label_bos[0] = 5001 
        label_bos[1: ] = label
        return {'audio': log_mel_spectrogram.squeeze(0), 'length': torch.tensor(length), 'label_bos': label_bos.long(),
                'label': label, 'task_id': torch.tensor([2]), 'target_length': torch.tensor(self.label_lengths[index])} 
    
    def __len__(self):
        return len(self.audio_paths)

libri_trainset = LibriDataset('./data/libri/train_960-transcript.txt')


libri_testset = LibriDataset('./data/libri/test-clean-transcript.txt')   