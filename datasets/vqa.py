from torch.utils.data import Dataset
import os.path as osp
import random
from PIL import Image
from torchvision.transforms import transforms
import json
from .tokenizer import BertTokenizer
import torch
    
class VQA(Dataset):
    def __init__(self, mode, img_folder):
        super().__init__()
        self.tokenizer = BertTokenizer(15, 15)
        self.img_folder = img_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        vqa_infos = json.load(open('data/vqav2/%s.json' % mode))
        self.vqa_infos = []
        for vqa_info in vqa_infos:
            if vqa_info['label']:
                self.vqa_infos.append(vqa_info)        
        self.vqa_labels = json.load(open('data/vqav2/labels.json')) 
        
        
    def __getitem__(self, index) :
        vqa_info = self.vqa_infos[index]
        img_id = vqa_info['img_id']
        question = vqa_info['sent']
        img_path = osp.join(self.img_folder, img_id + '.jpg')
        image = Image.open(img_path) 
        image_mode = image.mode
        if image_mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        text = self.tokenizer(question)
        labels = vqa_info['label']
        label, score = None, 0
        for l, s in labels.items():
            if s > score:
                label = l
                score = s
        data = text 
        data.update({'image': image, 'task_id': torch.tensor([5]), 'label': torch.tensor(self.vqa_labels[label])})
        return data
        

    def __len__(self):
        return len(self.vqa_infos)
    
vqa_trainset = VQA('train', 'data/vqav2/train2014')
vqa_testset = VQA('train', 'data/vqav2/train2014')