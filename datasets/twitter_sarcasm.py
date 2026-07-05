from torch.utils.data import Dataset
import os.path as osp
import random
from PIL import Image
from torchvision.transforms import transforms
import json
from .tokenizer import BertTokenizer
import torch
    
class TwitterSarcasm(Dataset):
    def __init__(self, mode): 
        self.tokenizer = BertTokenizer(15, 15)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        file = open("data/sarcasm_detection/extract_all")
        self.attr_dic = {}
        for line in file:
            ls = eval(line)
            self.attr_dic[int(ls[0])] = ls[1:]
        if mode == 'train':
            file_list = 'data/sarcasm_detection/text/train.txt'
        else:
            file_list = 'data/sarcasm_detection/text/test2.txt'
        
        with open(file_list, 'r') as f:
            lines = f.readlines() 
        self.data = []
        for line in lines:
            ls = eval(line.strip())
            tmpLS = ls[1].split()
            if "sarcasm" in tmpLS:
                continue
            if "sarcastic" in tmpLS:
                continue
            if "reposting" in tmpLS:
                continue
            if "<url>" in tmpLS:
                continue
            if "joke" in tmpLS:
                continue
            if "humour" in tmpLS:
                continue
            if "humor" in tmpLS:
                continue
            if "jokes" in tmpLS:
                continue
            if "irony" in tmpLS:
                continue
            if "ironic" in tmpLS:
                continue
            if "exgag" in tmpLS:
                continue
            self.data.append(ls)

        
    def __getitem__(self, index) :
        meta_info = self.data[index]
        img_id = meta_info[0]
        img_path = 'data/sarcasm_detection/dataset_image/%s.jpg' % img_id  
        image = Image.open(img_path) 
        image = self.transform(image)
        label = int(meta_info[-1])
        text = meta_info[1]
        data = self.tokenizer(text) 
        data.update({'label': torch.tensor(label), 'image': image, 'task_id': torch.tensor([12])})
        return data
    

    def __len__(self):
        return len(self.data)
    
twitter_sarcasm_trainset = TwitterSarcasm('train')
twitter_sarcasm_testset = TwitterSarcasm('test')
