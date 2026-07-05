from typing import Any, Tuple
from torchvision import datasets, transforms
from PIL import Image
import torch

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'image': img, 'label': torch.tensor(target), 'task_id': torch.tensor([0])}

cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
cifar10_realset = cifar10_testset
