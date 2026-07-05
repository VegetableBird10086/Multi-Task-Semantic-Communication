'''
Author: LOTEAT
Date: 2023-05-31 15:59:28
'''

import argparse
import torch 
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nthreads', default=5, type=int)
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets')
    parser.add_argument('--device_id', default=7, type=int, help='gpu id')
    parser.add_argument('--lrs', default=1e-4, type=float, nargs='+', help='The training learning rate')
    parser.add_argument('--optims', default='Adam', type=str, nargs='+', help='The training learning rate')
    parser.add_argument('--bs', default=32, type=int, nargs='+', help='The training batch size')
    parser.add_argument('--epochs', default=10, type=int, help='The training number of epochs')
    parser.add_argument('--save_path', default='./checkpoint', type=str, help='The path to save model')
    parser.add_argument('--train_snr', default=3, type=int, help='The train SNR')
    parser.add_argument('--test_snr', default=18, type=int, help='The test SNR')
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--test_interval', default=1, type=int, help='test interval') 
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help='bert name')
    parser.add_argument('--vivit_name', default='google/vivit-b-16x2-kinetics400', type=str, help='vivit name')
    parser.add_argument('--nheads', default=8, type=int) 
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float) 
    parser.add_argument('--enc_num_layers', default=4, type=int, help='The number of encoder layers')
    parser.add_argument('--dec_num_layers', default=4, type=int)
    parser.add_argument('--device_ids', type=int, nargs='+', help='distributed training')
    parser.add_argument('--load_from', type=str, help='distributed training')

    
        
    
    
    parser.add_argument('--max-length', default=35, type=int, help='The path to save model')
    parser.add_argument('--channel', default='AWGN', type=str, help='Choose the channel to simulate')

    # Model parameters
    parser.add_argument('--enc_num_layer', default=4, type=int, help='The number of encoder layers')
    parser.add_argument('--enc-d-model', default=128, type=int, help='The output dimension of attention')
    parser.add_argument('--enc-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--enc-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--enc-dropout', default=0.1, type=float, help='The encoder dropout rate')

    parser.add_argument('--dec-num-layer', default=4, type=int, help='The number of decoder layers')
    parser.add_argument('--dec-d-model', default=128, type=int, help='The output dimension of decoder')
    parser.add_argument('--dec-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--dec-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--dec-dropout', default=0.1, type=float, help='The decoder dropout rate')
    
    parser.add_argument('--vocab_size', default=30522)

    # Other parameter settings

    # Mutual Information Model Parameters

    args = parser.parse_args()

    return args

