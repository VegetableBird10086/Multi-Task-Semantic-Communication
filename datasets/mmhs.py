import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from PIL import Image
from torchvision.transforms import transforms
from .tokenizer import BertTokenizer

# genre_lookup = {
#     'Action':0,
#      'Adult':1,
#      'Adventure':2,
#      'Animation':3,
#      'Biography':4,
#      'Comedy':5,
#      'Crime':6,
#      'Documentary':7,
#      'Drama':8,
#      'Family':9,
#      'Fantasy':10,
#      'Film-Noir':11,
#      'History':12,
#      'Horror':13,
#      'Music':14,
#      'Musical':15,
#      'Mystery':16,
#      'News':17,
#      'Reality-TV':18,
#      'Romance':19,
#      'Sci-Fi':20,
#      'Short':21,
#      'Sport':22,
#      'Talk-Show':23,
#      'Thriller':24,
#      'War':25,
#      'Western':26
# }

class MMHSDataset(Dataset):
    def __init__(self, mode):
        with open('data/mmhs/MMHS150K_GT.json', 'r') as file:
            # Load the JSON data
            gt = json.load(file)
            
        with open('data/mmhs/splits/%s_ids.txt' % mode, 'r') as f:
            ids = f.readlines() 
        ids = [each_id.strip() for each_id in ids]        
        self.img_paths = []
        self.texts = [] 
        self.labels = []
        for each_id in ids:
            data_info = gt[each_id]
            try:
                with open('data/mmhs/img_txt/%s.json' % each_id, 'r') as f:
                    img_txt =json.load(f)
                self.img_paths.append('data/mmhs/img_resized/%s.jpg' % each_id) 
                self.texts.append(img_txt['img_text'] + data_info['tweet_text'])
                self.labels.append(data_info['labels'])
            except:
                self.img_paths.append('data/mmhs/img_resized/%s.jpg' % each_id) 
                self.texts.append(data_info['tweet_text'])
                self.labels.append(data_info['labels'])

        

            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])


        self.tokenizer = BertTokenizer(50, 50)
        
    def __getitem__(self, index):
        text = self.texts[index]
        data = self.tokenizer(text)
        img_path = self.img_paths[index]
        image = Image.open(img_path) 
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.labels[index]
        
        label = [int(l) for l in label]
        max_label = max(label,key=label.count)


        data.update({'image': image, 'label': torch.tensor(max_label), 'task_id': torch.tensor([15])})
        return data

    def __len__(self):
        return len(self.img_paths)
    
mmhs_trainset = MMHSDataset('train')
mmhs_testset = MMHSDataset('test')
