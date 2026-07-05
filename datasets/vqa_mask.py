from torch.utils.data import Dataset
import json 
import os  
import numpy as np 
import base64
from PIL import Image
from torchvision.transforms import transforms
import json
from .tokenizer import BertTokenizer
import torch
from PIL import Image, ImageDraw
import numpy as np
import random
# 'train':train,nominival
# 'test': minival

file_mappings = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}

class FastVQA(Dataset):
    def __init__(self, file_list):
        self.data_infos = []
        self.tokenizer = BertTokenizer(20, 20)
        
        for file in file_list:
            self.data_infos.extend(json.load(open("data/vqav2_fastrnn/data/%s.json" % file)))
        
        self.qid2info = {
            data_info['question_id']: data_info
            for data_info in self.data_infos
        }
        self.ans2label = json.load(open("data/vqav2_fastrnn/data/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqav2_fastrnn/data/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        self.offset = {}
        for file in file_list:
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_offset.txt' % (file_mappings[file])))
            offset = f.readlines()
            for l in offset:
                self.offset[l.split('\t')[0]] = int(l.split('\t')[1].strip())
                

        self.data = []
        for data_info in self.data_infos:
            if data_info['img_id'] in self.offset.keys():
                self.data.append(data_info)
                
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
                
    def __getitem__(self, index):
        data_info = self.data[index]
        img_id = data_info['img_id']
        ques_id = data_info['question_id']
        ques = data_info['sent']

        img_offset = self.offset[img_id]
        img_split = img_id[5:7]
        
        if(img_split == 'tr'):
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['train'])))
        elif(img_split == 'va'):
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['valid'])))
        else:
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['test'])))
        
        f.seek(img_offset)
        img_info = f.readline()
        f.close()
      
        assert img_info.startswith('COCO') and img_info.endswith('\n'), 'Offset is inappropriate'
        img_info = img_info.split('\t')  
        decode_infos = self.decode(img_info)
        # img_path = decode_infos[0]
        
        # image = Image.open(img_path) 
        # image_mode = image.mode
        # if image_mode == 'L':
        #     image = image.convert('RGB')
        # boxes = decode_infos[-2].tolist()
        # cropped_imgs = []
        # for box in boxes:
        #     cropped_img = image.crop(box=box) 
        #     cropped_imgs.append(self.transform(cropped_img).unsqueeze(0))
        
        # cropped_imgs = torch.cat(cropped_imgs, dim=0)

        label = torch.zeros(self.get_ans_nums())
        prob_labels = data_info['label']
        for ans, score in prob_labels.items():
            label[self.ans2label[ans]] = score    
        
        
        data = dict(
            image=torch.tensor(decode_infos[-1]),#cropped_imgs, 
            task_id=torch.tensor([10]),
            label=label,
            #fast_feats=torch.tensor(decode_infos[-1])
        )
        
        text = self.tokenizer(ques)
        data.update(text)
        return data

        
    def get_ans_nums(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)
    
    def decode(self, img_info):
        img_name = img_info[0]
        img_dir = img_name.split('_')[1] 
        img_path = os.path.join('data/vqav2', img_dir, img_name + '.jpg')
        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36,4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36,-1)
        feats.setflags(write=False)
        return [img_path, img_h, img_w, boxes, feats]
    
fastvqa_trainset = FastVQA(['train', 'nominival'])
fastvqa_testset = FastVQA(['minival'])


class FastVQATest(FastVQA):
    def __getitem__(self, index):
        data_info = self.data[index]
        img_id = data_info['img_id']
        ques_id = data_info['question_id']
        ques = data_info['sent']

        img_offset = self.offset[img_id]
        img_split = img_id[5:7]
        
        if(img_split == 'tr'):
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['train'])))
        elif(img_split == 'va'):
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['valid'])))
        else:
            f = open(os.path.join('data/vqav2_fastrnn/data/img', '%s_obj36.tsv' % (file_mappings['test'])))
        
        f.seek(img_offset)
        img_info = f.readline()
        f.close()
      
        assert img_info.startswith('COCO') and img_info.endswith('\n'), 'Offset is inappropriate'
        img_info = img_info.split('\t')  
        decode_infos = self.decode(img_info)
        # img_path = decode_infos[0]
        
        # image = Image.open(img_path) 
        # image_mode = image.mode
        # if image_mode == 'L':
        #     image = image.convert('RGB')
        # boxes = decode_infos[-2].tolist()
        # cropped_imgs = []
        # for box in boxes:
        #     cropped_img = image.crop(box=box) 
        #     cropped_imgs.append(self.transform(cropped_img).unsqueeze(0))
        
        # cropped_imgs = torch.cat(cropped_imgs, dim=0)
   
        
        
        data = dict(
            image=torch.tensor(decode_infos[-1]),#cropped_imgs, 
            task_id=torch.tensor([10]),
            qid=ques_id
        )
        
        text = self.tokenizer(ques)
        data.update(text)
        return data

    
fastvqa_realset = FastVQATest(['nominival'])
