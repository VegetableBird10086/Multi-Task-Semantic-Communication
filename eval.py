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

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std



# def word_error_rate(predictions, labels):
#     total_words = 0
#     total_errors = 0

#     # Loop through each batch
#     for pred_sentence, label_sentence in zip(predictions, labels):
#         pred_words = []
#         label_words = []
#         for pred_idx, label_idx in zip(pred_sentence, label_sentence):
#             # Convert the indices to words
#             pred_word = idx_to_word(pred_idx)
#             label_word = idx_to_word(label_idx)
#             pred_words.append(pred_word)
#             label_words.append(label_word)

#         # Calculate Levenshtein distance
#         lev_distance = levenshtein_distance(pred_words, label_words)

#         total_words += len(label_words)
#         total_errors += lev_distance

#     # Calculate Word Error Rate
#     wer = float(total_errors) / float(total_words) * 100.0
#     return wer


# Helper function to calculate Levenshtein distance
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



def eval_step(transceiver, test_loader, device):
    tokenizer = BertTokenizer()
    
    with torch.no_grad():
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
            
            if task_type in ['cifar10', 'glue_sst2', 'glue_cola', 'hmdb51', 'twitter_sarcasm', 'mmhs']: 
                predictions = transceiver(data)           
                predictions = torch.argmax(predictions, dim=1)
                
                labels = data['label'] 
                correct_predictions += (predictions == labels).sum().item()
                total += predictions.shape[0]
            
            if task_type in ['libri']:
                predictions = transceiver(data)
                labels = data['label'] 
                for i in range(labels.shape[0]):
                    pred_words = torch.argmax(predictions[i, :], dim=1)
                    label_words = labels[i, :]
                    label_words = truncate(label_words).cpu().detach().numpy().tolist()
                    pred_words = truncate(pred_words).cpu().detach().numpy().tolist()
                    lev_distance = edit_distance(label_words, pred_words)
                    
                    # lev_distance = levenshtein_distance(pred_words, label_words)
                    total_words += len(label_words)
                    total_errors += lev_distance
                    print(total_errors / total_words)
            
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
            
                    
            if task_type == 'mmdb':
                prediction = transceiver(data)           
                label = data['label'] 
                prediction = torch.sigmoid(prediction).cpu().detach().numpy() > 0.5
                preds.append(prediction)
                label = label.cpu().detach().numpy()
                tgts.append(label)
                
                

            if task_type in ['fastvqa']:
                predictions = transceiver(data) 
                pred = torch.argmax(predictions, dim=1)
                target = torch.argmax(data['label'], dim=1)
                
                correct_predictions += torch.sum(pred==target).item()
                total += target.shape[0]
                print(correct_predictions / total)
            
            


                       
                
        if task_type in ['cifar10', 'glue_sst2', 'glue_cola', 'hmdb51', 'twitter_sarcasm', 'mmhs', 'fastvqa']:
            accuracy = correct_predictions / total
            tqdm.write('Task: %s Accuracy: %.2f' % (task_type, accuracy)) 
            return accuracy
        
        if task_type in ['libri']:
            wer = total_errors / total_words 
            tqdm.write('Task: %s Word Error Rate: %.2f' % (task_type, wer)) 
            return wer
        
        if task_type == 'europarl':
            metric = BleuScore(1, 0, 0, 0)
            score = metric.compute_score(pred_sentences, target_sentences) 
            score = np.array(score)
            score = np.mean(score)
            tqdm.write('Task: %s Bleu score: %.2f' % (task_type, score))
            
        if task_type == 'mmdb':
            tgts = np.vstack(tgts)
            preds = np.vstack(preds)
            macro_f1 = f1_score(tgts, preds, average="macro")
            micro_f1 = f1_score(tgts, preds, average="micro")
            tqdm.write('Task: %s macro_f1: %.2f, micro_f1: %.2f' % (task_type, macro_f1, micro_f1))
            return macro_f1

            



if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(5)
    args = parse_args()

    test_sets = []
    for dataset_name in args.datasets:
        test_sets.append(getattr(datasets, dataset_name + '_testset'))

    test_loaders = [DataLoader(test_set, batch_size=bs, shuffle=True) for test_set, bs in zip(test_sets, args.bs)]
    
    device = 'cuda:%d' % args.device_id
    
    # Load the model from the checkpoint path
    transceiver = torch.load(args.load_from, map_location='cpu')
    
    transceiver = transceiver.to(device)
    transceiver.eval()
    for test_loader in test_loaders:    
        eval_step(transceiver, test_loader, device)
    

