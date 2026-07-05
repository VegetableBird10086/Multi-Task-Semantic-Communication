import torch.nn as nn 
import torch
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from .image_encoder import resnet50
from .utils import *
from .generator import Generator
import math
from ..transformer.embeddings import *
import transformers
from speechbrain.nnet.linear import Linear

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class SemanticDecoder(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(SemanticDecoder, self).__init__()
        args.bert_config.hidden_size = args.d_model
        decoder_layer = nn.TransformerDecoderLayer(128, 8)
        self.decoder = nn.TransformerDecoder(decoder_layer, 4)
        self.text_embedding = TextEmbeddings(128, 5002)
        self.pos_embedding = PositionEmbedding(128, 0.1)
        self.classifiers = nn.ModuleDict({
            'cifar10': nn.Linear(args.d_model, 10),
            'glue_sst2': nn.Sequential(
                # BertPredictionHeadTransform(args.bert_config),
                nn.Flatten(1),
                nn.Linear(128 * args.bert_config.hidden_size, 2),
            ),
            'glue_cola': nn.Sequential(
                # BertPredictionHeadTransform(args.bert_config),
                nn.Flatten(1),
                nn.Linear(128 * args.bert_config.hidden_size, 2),
            ),
            'europarl': nn.Linear(128, 30522),
            'fastvqa': nn.Sequential(
                # nn.Linear(128, 128 * 2),
                # GeLU(),
                # nn.LayerNorm(128 * 2, eps=1e-12),
                # nn.Linear(768, 3129)
                # nn.Linear(768, 768 * 2),
                # GeLU(),
                # nn.LayerNorm(128, eps=1e-12),
                # nn.Linear(128, 3129)
                
                
                # not fusion 
                nn.Linear(768, 768 * 2),
                nn.ReLU(), 
                nn.Linear(768 * 2, 768),
                nn.LayerNorm(768),
                nn.Flatten(1),
                nn.Linear(768, 3129)
                
            ),
            'hmdb51': nn.Sequential(
                nn.Flatten(1),
                nn.Linear(128, 51)
            ),
            'cifar10_recovery': Generator(),
            'libri_sb': nn.ModuleDict({
                'ctc_lin': Linear(input_size=512, n_neurons=5000),
                'softmax': nn.LogSoftmax(dim=-1),
                'seq_lin': Linear(input_size=512, n_neurons=5000),
            }),
            'twitter_sarcasm': nn.Sequential(
                # nn.Flatten(1),
                # nn.Linear(768 * 129, 2)
                nn.Flatten(1),
                nn.Linear(768, 2)
            ),
            
            'flickr30': nn.Sequential(
                # nn.Flatten(1),
                # nn.Linear(768 * 51, 1),
                
                
                nn.Flatten(1),
                nn.Linear(768, 1),
                
                nn.Sigmoid()
            ),
            'mmdb': nn.Sequential(
                # nn.Flatten(1),
                # nn.Linear(129 * 768, 27)
                
                # nn.Linear(768, 768 * 2),
                # GeLU(),
                # nn.LayerNorm(768 * 2, eps=1e-12),
                # nn.Linear(768 * 2, 27)
                
                
            
                # nn.LayerNorm(768, eps=1e-12),
                nn.Flatten(1),
                nn.Linear(128, 27)
            ),
            'mmhs': nn.Sequential(
                nn.Flatten(1),
                nn.Linear(128, 6)
                
                # nn.Flatten(1),
                # nn.Linear(768, 6)
            )
        })
        
        
    
        
    def forward(self, data):
        task_type = data['task_type']
        
        if task_type in ['cifar10']:
            feature = data['image']
            
        if task_type in ['cifar10_recovery']:
            feature = data['image'].unsqueeze(-1).unsqueeze(-1)
        
        if task_type in ['glue_sst2', 'glue_cola', 'europarl']:
            feature = data['text']
            
        if task_type == 'libri':
            feature = data['audio']
            length = data['length'] 
            tgt = data['label_bos']
            tgt = self.text_embedding(tgt)
            tgt = self.pos_embedding(tgt) 
            (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
            ) = make_masks(feature, tgt, length, pad_idx=0)
            print(tgt.shape, feature.shape)
            print(src_key_padding_mask.shape, tgt_key_padding_mask.shape, src_mask.shape, tgt_mask.shape)
            decoder_output = self.decoder(tgt, feature, tgt_mask, src_mask, tgt_key_padding_mask, src_key_padding_mask)
        
        if task_type == 'libri_sb':
            classifier = self.classifiers[task_type]
            audio = data['audio']
            logits = classifier['ctc_lin'](audio)  
            data['p_ctc'] = classifier['softmax'](logits)
            audio_pred = data['audio_pred']
            audio_pred = classifier['seq_lin'](audio_pred)
            data['p_seq'] = classifier['softmax'](audio_pred)
            return data
            
        
        
        if task_type in ['fastvqa', 'twitter_sarcasm', 'flickr30', 'mmdb', 'mmhs']:
            feature = data['memories']
    

        if task_type == 'hmdb51':
            feature = data['video']

        
        classifier = self.classifiers[task_type]
        results = classifier(feature)

        
        return results
    
