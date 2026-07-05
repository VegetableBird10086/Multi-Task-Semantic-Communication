from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
from .tokenizer import BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from random import shuffle

annotations = pd.read_table('data/flickr/results_20130124.token', sep='\t', header=None,
                    names=['image', 'caption'])
data = []
for i in tqdm(range(len(annotations))):
    data.append((annotations.iloc[i]['image'].split('#')[0], annotations.iloc[i]['caption']))
shuffle(data) 


class Flickr30(Dataset):
    def __init__(self, mode): 
        length = len(data)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        if mode == 'train':
            self.data = data[: int(0.9 * length)]
        else:
            self.data = data[int(0.9 * length): ]
        self.tokenizer = BertTokenizer(50, 50)
        
    def __getitem__(self, index) :
        img, caption = self.data[index] 
        img_path = 'data/flickr/flickr30k-images/%s' % img  
        image = Image.open(img_path) 
        image = self.transform(image)
        data = self.tokenizer(caption) 
        data.update({'image': image, 'task_id': torch.tensor([13])})
        return data
    

    def __len__(self):
        return len(self.data)
    
flickr30_trainset = Flickr30('train')
flickr30_testset = Flickr30('test')


def flickr_collate(raw_batch):
    batch = {'label': []}
    bs = len(raw_batch)
    idxs = list(range(bs))

    
    mappings = torch.eye(bs, dtype=torch.float32)
    for i in range(bs):
        img = raw_batch[i]['image']
        for j in range(bs):
            for key, value in raw_batch[j].items():
                batch.setdefault(key, [])
                if key == 'image':
                    batch[key].append(img.clone())
                else:
                    batch[key].append(value.clone())
            batch['label'].append(mappings[i, j]) 
    batch = {key: torch.stack(value) for key, value in batch.items()}
    return batch
