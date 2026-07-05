import json
import numpy as np
import torch
from models.transceiver import Transceiver
from helper import parse_args
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm
from config import task_mappings
from datasets.tokenizer import BertTokenizer
from nltk.metrics import edit_distance
from metric import BleuScore
from sklearn.metrics import f1_score
import os 

def vqa_map_reverse():
    with open('data/vqav2_fastrnn/data/trainval_ans2label.json', 'r') as f:
        vqa_map = json.load(f)
    vqa_map_r = {value: key for key, value in vqa_map.items()}
    return vqa_map_r


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]



def truncate(words, trunc_idx=5001):
    start_idx, end_idx = 0, 0 
    while end_idx < len(words) and words[end_idx] != trunc_idx:
        end_idx += 1 
    return words[start_idx:end_idx]



def test_step(transceiver, test_loader, device, dataset_name, snr):
    tokenizer = BertTokenizer()
    vqa_map_r = vqa_map_reverse()
    with torch.no_grad():
        results = []
        total = 0
        correct_predictions = 0
        
        total_words = 0
        total_errors = 0
        
        pred_sentences = []
        target_sentences = []
        
        preds = [] 
        tgts = []
        for data in tqdm(test_loader):
            data = {key: value.to(device) for key, value in data.items()}
            task_id = int(data['task_id'][0, 0])
            del data['task_id']
            task_type = task_mappings[task_id]
            data['task_type'] = task_type

            if task_type in ['europarl']:
                predictions = transceiver(data)
                labels = data['label'] 
                for i in range(labels.shape[0]):
                    pred_words = torch.argmax(predictions[i, :], dim=1)
                    label_words = labels[i, :]
                    label_words = truncate(label_words, 102).cpu().detach().numpy().tolist()
                    pred_words = truncate(pred_words, 102).cpu().detach().numpy().tolist()
                    
                    label_words = tokenizer.ids2tokens(label_words[1: ])
                    pred_words = tokenizer.ids2tokens(pred_words[1: ])
                    label_words = ' '.join(label_words)
                    pred_words = ' '.join(pred_words) 
                    pred_sentences += [pred_words, ]
                    target_sentences += [label_words, ]
            
        

            if task_type in ['fastvqa']:
                predictions = transceiver(data) 
                pred = torch.argmax(predictions, dim=1)
                ans = vqa_map_r[int(pred)]
                qid = int(data['qid'])
                results.append({'answer': ans, 'question_id': qid})
    

            if task_type in ['glue_sst2', 'glue_cola', 'twitter_sarcasm', 'mmhs']: 
                predictions = transceiver(data)           
                predictions = torch.argmax(predictions, dim=1)
                
                results.append((data['index'][0].detach().cpu().item(), predictions[0].detach().cpu().item()))
            
            if task_type in ['cifar10', 'hmdb51']:
                predictions = transceiver(data)           
                predictions = torch.argmax(predictions, dim=1)
                labels = data['label'] 
                correct_predictions += (predictions == labels).sum().item()
                total += predictions.shape[0]

                
                

            
                               
                
        if task_type in ['fastvqa']:
            with open('result/fastvqa.json', 'w') as f:
                json.dump(results, f)

        if task_type == 'europarl':
            with open('result/europarl.txt', 'a') as f:
                metric = BleuScore(1, 0, 0, 0)
                score = metric.compute_score(pred_sentences, target_sentences) 
                score = np.array(score)
                score = np.mean(score)
                f.write(f'{snr} {score}')
            tqdm.write('Task: %s SNR: %d Bleu score: %.2f' % (task_type, snr, score))

                    
        if task_type in ['glue_sst2']:
            with open(f'result/{task_type}_{snr}.tsv', 'w') as f:
                f.write('IDs\tlabels\n')
                for r in results:
                    f.write(f"{r[0]}\t{r[1]}\n")
            tqdm.write(f'Task: {task_type} SNR: {snr} Done!') 
        

        if task_type in ['cifar10', 'hmdb51']:
            with open(f'result/{task_type}_{snr}.tsv', 'w') as f:
                f.write(f'{snr} {correct_predictions/total}')
            tqdm.write(f'Task: {task_type} SNR: {snr} Accuracy: {correct_predictions/total}') 

        


            



if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(5)
    args = parse_args()
    test_sets = []
    for dataset_name in args.datasets:
        test_sets.append(getattr(datasets, dataset_name + '_realset'))

    test_loaders = [DataLoader(test_set, batch_size=1, shuffle=False) for test_set, bs in zip(test_sets, args.bs)]
    
    device = 'cuda:%d' % args.device_id
    
    # Load the model from the checkpoint path
    transceiver = torch.load(args.load_from, map_location='cpu')
    
    transceiver = transceiver.to(device)
    transceiver.eval()

    for test_loader, dataset_name in zip(test_loaders, args.datasets): 
        for snr in range(-6, 21, 3):
            transceiver.update_std(snr)   
            test_step(transceiver, test_loader, device, dataset_name, snr)
        

