from typing import Any, Tuple
from torchvision import datasets, transforms
from PIL import Image
import torch

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CIFAR10Recover(datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        label = img / 255.0
        label = transforms.ToTensor()(label).float()
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return {'image': img, 'label': label, 'task_id': torch.tensor([9])}
    
cifar10_recovery_trainset = CIFAR10Recover(root='./data', train=True, download=True, transform=train_transform)
cifar10_recovery_testset = CIFAR10Recover(root='./data', train=False, download=True, transform=test_transform)
