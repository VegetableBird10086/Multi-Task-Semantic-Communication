import datetime
import os

from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import random 
from optimizer import BertAdam

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
import warnings
warnings.filterwarnings("ignore")



from helper import parse_args
import os
import torch.optim as optim
from datasets import libri_padding
import datasets
from torch.utils.data import DataLoader
from models.transceiver import Transceiver
from tqdm import tqdm
from train import train_step, train_step_one
from eval import eval_step
from config import task_mappings

def get_optimizers(args, model):
    optims = args.optims
    lrs = args.lrs
    tasks = args.datasets 
    
    optimizers = {}
    for (lr, task, optim_name) in zip(lrs, tasks, optims):
        try:
            optimizer = getattr(optim, optim_name)(model.parameters(), lr=lr) 
        except:
            optimizer = BertAdam(model.parameters(), lr=lr, warmup=0.1, t_total=int(1944 * args.epochs))
        optimizers[task] = optimizer
        
    return optimizers
    


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def build_multi_loaders(train_sets, args):
    multi_loaders = []
    total_len = 0
    for i, train_set in enumerate(train_sets):
        dataset_name = args.datasets[i] 
        if dataset_name == 'libri_sb':
            train_loader = DataLoader(train_set, batch_size=args.bs[i], shuffle=True, drop_last=True, num_workers=8, pin_memory=True, collate_fn=libri_padding)
        elif dataset_name == 'flickr30':
            train_loader = DataLoader(train_set, batch_size=args.bs[i], shuffle=True, drop_last=True, num_workers=8, pin_memory=True, collate_fn=flickr_collate)
        else:
            train_loader = DataLoader(train_set, batch_size=args.bs[i], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
        total_len += len(train_loader)
        multi_loaders.append(iter(train_loader))
    return multi_loaders, total_len

def get_batch_data(multi_loaders):
    while True:
        loader_nums = len(multi_loaders)
        if loader_nums == 0:
            flag = True 
            data_batch = None
            break
        else:
            random_idx = random.randint(0, loader_nums-1)
            data_loader = multi_loaders[random_idx]
            # data_batch = next(data_loader)
            # flag = False
            try:
                data_batch = next(data_loader)
                flag = False
                break
            except:
                multi_loaders.pop(random_idx)
    return flag, data_batch
    

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    ngpus_per_node  = torch.cuda.device_count()

    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    rank        = int(os.environ["RANK"])
    device      = torch.device("cuda", local_rank)
    if local_rank == 0:
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
        print("Gpu Device Count : ", ngpus_per_node)

    
    
    
    train_sets, test_sets = [], []

    for dataset_name, bs in zip(args.datasets, args.bs):
        train_sets.append(getattr(datasets, dataset_name + '_trainset'))
        test_sets.append(getattr(datasets, dataset_name + '_testset'))


    test_loaders = [DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4) for test_set, bs in zip(test_sets, args.bs)]
    
    
    transceiver = Transceiver(args)
    # transceiver_optim = optim.Adam(transceiver.parameters(), lr=args.trans_lr)
    optimizers = get_optimizers(args, transceiver)
    # optimizers = {'fastvqa': BertAdam(transceiver.parameters(),lr=1e-4, warmup=0.1, t_total=int(19753 * args.epochs))}
    

    transceiver = transceiver.cuda(local_rank)
    transceiver = torch.nn.parallel.DistributedDataParallel(transceiver, device_ids=[local_rank], find_unused_parameters=True)
    
    # Training the model
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loaders, total_len = build_multi_loaders(train_sets, args)
        transceiver.train()
        tqdm.write('Epoch %d training starts' % epoch)
        iter_n = 0 
        cur_loss = None
        cur_batch = 0
        while True:
            flag, data = get_batch_data(train_loaders) 
            if flag:
                break
            data = {key: value.to(device) for key, value in data.items()}
            data['epoch'] = epoch + 1
            task_id = int(data['task_id'][0, 0])
            del data['task_id']
            task_type = task_mappings[task_id]
            data['task_type'] = task_type
            loss, metric, metric_value = train_step(data, transceiver, optimizers[task_type], device)
            # torch.cuda.empty_cache()

            if iter_n % 10 == 0:
                end_time = time.time()

                ela = end_time - start_time 
                eta = (total_len - iter_n) / (iter_n + 1) * ela

                tqdm.write(f"Epoch {epoch}: [{iter_n}]/[{total_len}], task:{task_type}, loss: {loss}, {metric}: {metric_value}, ela: {format_time(ela)}, eta: {format_time(eta)}")
            if iter_n % 200 == 0:
                tqdm.write('Model is saved to "%s/epoch_%d.pth' % (args.save_path, epoch))
                torch.save(transceiver, "%s/epoch_%d.pth" % (args.save_path, epoch))
            iter_n += 1
            
        tqdm.write('Epoch %d training ends' % epoch)
        tqdm.write('Model is saved to "%s/epoch_%d.pth' % (args.save_path, epoch))
        torch.save(transceiver, "%s/epoch_%d.pth" % (args.save_path, epoch))
            

        
        # if epoch % args.test_interval == 0 and data['task_type'] != 'libri':
        #     tqdm.write('Test starts')
        #     for test_loader in test_loaders:
        #         transceiver.eval()
        #         eval_step(transceiver, test_loader, device)
        
        
        