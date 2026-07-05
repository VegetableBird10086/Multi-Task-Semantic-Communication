import json
from datasets.libri import libri_trainset
import torch
from torch import nn
from torch.nn.functional import log_softmax
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers.models.auto.configuration_auto import AutoConfig
from omegaconf import OmegaConf
from models.semantic.text_encoder import BertModel

class SparseCategoricalCrossentropyLoss(nn.Module):
    def __init__(self, ignore_index=0, vocab_size=5002):
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

def plot_mel_spectrogram(wave, title=None, y_label="freq_bin", aspect="auto", x_max=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(y_label)
    axs.set_xlabel("frame")
    im = axs.imshow(wave.transpose(0, 1), origin="lower", aspect=aspect)
    if x_max:
        axs.set_xlim((0, x_max))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    bert_name = 'bert-base-uncased'
    bert_config = OmegaConf.create({'bert_model_name': bert_name})
    params_config = {'config': AutoConfig.from_pretrained(bert_name, 
                                **OmegaConf.to_container(bert_config))}
    model = BertModel.from_pretrained(bert_name, **params_config).to(device)
        
       
    
    
    criterion = SparseCategoricalCrossentropyLoss()
    print(f"Use device: {device}")


    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(libri_trainset, batch_size=32, shuffle=True)
    for data in dataloader:
        optim.zero_grad()
        print(data['audio'].shape)
        out = model(data['audio'].to(device))
        target = data['label'].to(device)
        target_length = data['target_length'].to(device)
        #probs = log_softmax(out, dim=-1)
        print(out.shape, target.shape)
        loss = criterion(out, target)
        print(loss)
        loss.backward()
        optim.step()
        print('target: ', target[0])
        print('pred: ', torch.argmax(out[0, :, :], dim=1))
