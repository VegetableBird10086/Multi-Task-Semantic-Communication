'''
Author: LOTEAT
Date: 2023-05-31 16:34:26
'''
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义多任务数据集类
class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        task_id = index % len(self.datasets)
        dataset = self.datasets[task_id]
        return dataset[index // len(self.datasets)]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

# 自定义多任务数据加载器类
class MultiTaskDataLoader(DataLoader):
    def __init__(self, datasets, batch_size=1, shuffle=True, num_workers=0, **kwargs):
        self.datasets = datasets
        self.dataset = MultiTaskDataset(datasets)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def __iter__(self):
        return _MultiTaskDataLoaderIter(self)

class _MultiTaskDataLoaderIter:
    def __init__(self, loader):
        self.loader = loader
        self.iter_dict = {i: iter(dataset) for i, dataset in enumerate(loader.datasets)}
        self.task_ids = list(self.iter_dict.keys())

    def __iter__(self):
        return self

    def __next__(self):
        task_id_choice = torch.randint(len(self.task_ids), (1,))
        try:
            task_data = []
            for _ in range(self.loader.batch_size):
                task_data.append(next(self.iter_dict[task_id_choice.item()]))
        except StopIteration:
            self.task_ids = [task_id for task_id in self.task_ids if task_id != task_id_choice.item()]
            if len(self.task_ids) == 0:
                raise StopIteration()
            else:
                return self.__next__()
        batch_data = {}
        for key in task_data[0].keys():
            batch_data[key] = torch.stack([data[key] for data in task_data])
        return batch_data