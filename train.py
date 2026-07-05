'''
Author: LOTEAT
Date: 2023-06-17 22:10:24
'''
from torch import nn
import torch
from loss import SparseCategoricalCrossentropyLoss, SpeechLoss, ContrastiveLoss
from torch.nn.utils import clip_grad
from torchvision.utils import save_image
from torch.nn.functional import log_softmax
import json
from metric_meter import MetricMeter
import numpy as np
from math import sqrt 
from sklearn.metrics import f1_score

total = 0 
right = 0
 

def get_vqa_weight():
    vqa_nums = json.load(open('data/vqav2/vqa_nums.json', 'r'))
    vqa_label_maps = json.load(open('data/vqav2/labels.json', 'r'))
    vqa_sums = sum([value for value in vqa_nums.values()])
    weights = torch.zeros(3129, dtype=torch.float32) 
    for key, value in vqa_nums.items():
        weight = vqa_sums / value  
        index = vqa_label_maps[key] 
        weights[index] = weight 
    mask = weights > 20 
    weights[mask] = 20
    return weights


criterions = {
    'cifar10': nn.CrossEntropyLoss(),
    'glue_sst2': nn.CrossEntropyLoss(),
    'libri': SparseCategoricalCrossentropyLoss(0, 5002),
    'hmdb51': nn.CrossEntropyLoss(),
    'glue_cola': nn.CrossEntropyLoss(),
    'ucf11': nn.CrossEntropyLoss(),
    'cifar100': nn.CrossEntropyLoss(),
    'europarl': SparseCategoricalCrossentropyLoss(),
    'cifar10_recovery': nn.MSELoss(),
    'fastvqa': nn.BCEWithLogitsLoss(),
    'libri_sb': SpeechLoss(),
    'twitter_sarcasm': nn.CrossEntropyLoss(),
    'flickr30': ContrastiveLoss(),
    'mmdb': nn.BCEWithLogitsLoss(),
    'mmhs': nn.CrossEntropyLoss(),
} 

metrics = {
    'cifar10': MetricMeter(),
    'glue_sst2': MetricMeter(),
    'libri': MetricMeter(),
    'hmdb51': MetricMeter(),
    'glue_cola': MetricMeter(),
    'ucf11': MetricMeter(),
    'cifar100': MetricMeter(),
    'europarl': MetricMeter(),
    'cifar10_recovery': MetricMeter(),
    'fastvqa': MetricMeter(),
    'libri_sb': MetricMeter(),
    'twitter_sarcasm': MetricMeter(),
    'flickr30': MetricMeter(),
    'mmdb': MetricMeter(),
    'mmhs': MetricMeter()
}

losses = {
    'cifar10': MetricMeter(),
    'glue_sst2': MetricMeter(),
    'libri': MetricMeter(),
    'hmdb51': MetricMeter(),
    'glue_cola': MetricMeter(),
    'ucf11': MetricMeter(),
    'cifar100': MetricMeter(),
    'europarl': MetricMeter(),
    'cifar10_recovery': MetricMeter(),
    'fastvqa': MetricMeter(),
    'libri_sb': MetricMeter(),
    'twitter_sarcasm': MetricMeter(),
    'flickr30': MetricMeter(),
    'mmdb': MetricMeter(),
    'mmhs': MetricMeter()
}


def train_step_one(data, transceiver, optim_net, device, pre_loss, cur_batch):
    task_type = data['task_type']
    criterion = criterions[task_type].to(device)
    # Forward pass
    predictions = transceiver(data)
    print(list(predictions.keys()))
    
    cur_loss = criterion(predictions, data['label'])
    loss = pre_loss + cur_loss if pre_loss else cur_loss  
    cur_batch += 1 
    if cur_batch == 8:
        flag = True
        optim_net.zero_grad()
        loss /= 8 
        loss.backward() 
        nn.utils.clip_grad_norm_(transceiver.parameters(), 5.)
        optim_net.step()
        if task_type in ['libri', 'europarl']:
            loss_ema = losses[task_type]
            loss_ema.update(loss.item())
            return flag, loss_ema.get_ema_metric(), 'No Metric', None
        
        elif task_type in ['cifar10_recovery']:
            loss_ema = losses[task_type]
            loss_ema.update(loss.item())
            return flag, loss_ema.get_ema_metric(), 'PSNR', None
        
        elif task_type in ['fastvqa']:
            pred = torch.argmax(predictions, dim=1)
            target = torch.argmax(data['label'], dim=1)
            acc = torch.sum(pred==target).item() / target.shape[0]
            
            loss_ema = losses[task_type]
            loss_ema.update(loss.item())
            
            metric_ema = metrics[task_type]
            metric_ema.update(acc)
            
            return flag, loss_ema.get_ema_metric(), 'Accuracy', metric_ema.get_ema_metric()

        else:
            pred = torch.argmax(predictions, dim=1)
            target = data['label']
            acc = torch.sum(pred==target).item() / target.shape[0]
            
            loss_ema = losses[task_type]
            loss_ema.update(loss.item())
            
            metric_ema = metrics[task_type]
            metric_ema.update(acc)
            
            return flag, loss_ema.get_ema_metric(), 'Accuracy', metric_ema.get_ema_metric()
    else:  
        flag = False
        return flag, loss, cur_batch
    
    


def train_step(data, transceiver, optim_net, device):
    optim_net.zero_grad()
    task_type = data['task_type']
    
    # Forward pass
    predictions = transceiver(data)
    criterion = criterions[task_type].to(device)
    
    # save_image(predictions, "image.png", normalize=True)
    if task_type == 'libri_sb':
        loss = criterion(predictions)
    elif task_type == 'flickr30':
        predictions = predictions.squeeze(-1)
        scores = predictions.cpu().detach().numpy()
        loss = criterion(predictions)
    else:
        loss = criterion(predictions, data['label'])  
    loss.backward() 
    nn.utils.clip_grad_norm_(transceiver.parameters(), 5.)

    # for name, param in transceiver.named_parameters():
    #     if param.grad is not None:
    #         if torch.isnan(param.grad).any():
    #             transceiver.zero_grad()
    #             optim_net.zero_grad()
    #             return torch.inf, 'No Metric', torch.nan

    optim_net.step()
    if task_type in ['libri', 'europarl', 'libri_sb']:
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        return loss_ema.get_ema_metric(), 'No Metric', None
    
    elif task_type in ['cifar10_recovery']:
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        return loss_ema.get_ema_metric(), 'PSNR', None     

    elif task_type in ['fastvqa']:
        pred = torch.argmax(predictions, dim=1)
        target = torch.argmax(data['label'], dim=1)
        acc = torch.sum(pred==target).item() / target.shape[0]
        
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        
        metric_ema = metrics[task_type]
        metric_ema.update(acc)
        
        return loss_ema.get_ema_metric(), 'Accuracy', metric_ema.get_ema_metric()
    
    elif task_type in ['mmdb']:        
        label = data['label'].cpu().detach().numpy()
        prediction = torch.sigmoid(predictions).cpu().detach().numpy() > 0.5
        macro_f1 = f1_score(label, prediction, average="macro")
        
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        
        metric_ema = metrics[task_type]
        metric_ema.update(macro_f1)
        return loss_ema.get_ema_metric(), 'Macro F1', metric_ema.get_ema_metric()
    
    elif task_type in ['flickr30']:
        n = int(sqrt(scores.shape[0]))
        scores = scores.reshape(n, n)
        pred = np.argmax(scores, axis=1)
        label = np.arange(n) 
        acc = np.sum(pred==label) / n
        
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        
        metric_ema = metrics[task_type]
        metric_ema.update(acc)
        
        return loss_ema.get_ema_metric(), 'Accuracy', metric_ema.get_ema_metric()

    else:
        pred = torch.argmax(predictions, dim=1)
        target = data['label']
        acc = torch.sum(pred==target).item() / target.shape[0]
        
        loss_ema = losses[task_type]
        loss_ema.update(loss.item())
        
        metric_ema = metrics[task_type]
        metric_ema.update(acc)
        
        return loss_ema.get_ema_metric(), 'Accuracy', metric_ema.get_ema_metric()


