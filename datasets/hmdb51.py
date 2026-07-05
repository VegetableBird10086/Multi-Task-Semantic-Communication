from typing import Any, Tuple
from .transforms import *
from torchvision import datasets, transforms
import torchvision
from PIL import Image
import torch 


# class HMDB51Cache(datasets.HMDB51):
#     def __init__(self, mode):
#         file_path = f'data/hmdb51/cache/{mode}.txt' 
#         with open(file_path) as f:
#             lines = f.readlines() 
#         self.data_paths = [line.strip().split(' ')[0] for line in lines]
#         self.labels = [int(line.strip().split(' ')[1]) for line in lines]

 
#     def __getitem__(self, idx: int):
#        data = torch.load(self.data_paths[idx])
#        return {'video': data, 'task_id': torch.tensor([4]), 'label': torch.tensor(self.labels[idx])}
    
#     def __len__(self):
#         return len(self.data_paths)

# hmdb51_trainset = HMDB51Cache('train')
# hmdb51_testset = HMDB51Cache('test')



transform = transforms.Compose([
    ToFloatTensorInZeroOne(),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    Resize((256, 256)),
    CenterCrop((224, 224))
])

# step_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),  # 添加随机垂直翻转
#     transforms.RandomRotation(degrees=30),  # 添加随机旋转
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加颜色抖动
# ])

transform_test = transforms.Compose([
    ToFloatTensorInZeroOne(),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    Resize((256, 256)),
    CenterCrop((224, 224))])

num_frames = 8 
clip_steps = 2


class HMDB51(datasets.HMDB51):
        
    def __getitem__(self, idx: int):
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        video = video.transpose(0, 1)

        # return video, audio, class_index
        return {'video': video, 'task_id': torch.tensor([4]), 'label': torch.tensor(class_index)}
    

hmdb51_trainset = HMDB51('/mnt/sdh/zzl/video_data', '/mnt/sdh/zzl/test_train_splits/', num_frames,frame_rate=5,
                                                step_between_clips = clip_steps, fold=1, train=True,
                                                transform=transform, num_workers=8)

hmdb51_testset = HMDB51('/mnt/sdh/zzl/video_data', '/mnt/sdh/zzl/test_train_splits/', num_frames,frame_rate=5,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=transform_test, num_workers=8)

hmdb51_realset = hmdb51_testset 

