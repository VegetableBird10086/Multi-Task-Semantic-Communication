import torch
import torch.nn as nn
from speechbrain.nnet.losses import ctc_loss, kldiv_loss
from math import sqrt 
from torch.autograd import Variable
class SparseCategoricalCrossentropyLoss(nn.Module):
    def __init__(self, ignore_index=0, vocab_size=30522):
        super(SparseCategoricalCrossentropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.loss_object = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, real):
        mask = real != self.ignore_index
        bs = pred.shape[0]
        loss_ = self.loss_object(pred.contiguous().view(-1, self.vocab_size), real.contiguous().view(-1))#  * mask.float()
        loss_ = loss_.view(bs, -1)
        loss_ *= mask.float()
        return torch.mean(loss_)
    

class SpeechLoss(nn.Module):
    def __init__(self):
        super(SpeechLoss, self).__init__()
        self.ctc_loss = lambda x, y, feat_length, label_length: ctc_loss(x, y, feat_length, label_length, blank_index=0, reduction='batchmean')
        self.seq_loss = lambda x, y, label_length: kldiv_loss(x, y, label_length, label_smoothing=0, reduction='batchmean')
        
    def forward(self, results):
        loss_seq = self.seq_loss(
            results['p_seq'], results['label_eos'], results['label_eos_length']
        ).sum()

        loss_ctc = self.ctc_loss(
            results['p_ctc'], results['label'], results['audio_length'], results['label_length']
        ).sum()

        loss = (
            0.3 * loss_ctc
            + (1 - 0.3) * loss_seq
        )
        return loss 
    
    
    


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        n = int(sqrt(scores.shape[0]))
        scores = scores.view(n, n)
        diagonal = scores.diag().view(n, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.to(scores.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()