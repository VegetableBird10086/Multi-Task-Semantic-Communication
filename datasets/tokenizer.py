from transformers import AutoTokenizer
import torch

class BertTokenizer():
    def __init__(self, max_tokens=128, max_seq_length=128):
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = 0
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lowercase=True
        )
        self.max_tokens = max_tokens
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
    
    def tokens2ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids2tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def truncate(self, tokens):
        while len(tokens) > self.max_tokens - 2:
            tokens.pop()
        return tokens
    
    def tokens2indices(self, tokens, labels):
        tokens = [self.cls_token] + tokens + [self.sep_token]
        segment_ids = [0] * len(tokens) 
        lm_label_ids = [-1] + labels + [-1]
        input_ids = self.tokens2ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.pad_token)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            "text": input_ids,
            "text_mask": input_mask,
            "segment_ids": segment_ids,
        }
    
    def __call__(self, sentence):
        tokens = self.tokenize(sentence)
        tokens = self.truncate(tokens)
        labels = self.tokens2ids(tokens)
        output = self.tokens2indices(tokens, labels)
        return output