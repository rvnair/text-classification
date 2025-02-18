import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time
from transformers import BertModel

class DAN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim = 300, n_hidden_units = 300, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units, self.n_classes)
        )
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self._softmax = nn.Softmax(dim=1)
        self.device = device
    
    def getLengthSample(self, sample):
        count = 0
        for w in sample:
            if w == 0:
                break
            count += 1
        return count

    # text = tensor of size batch x seq_length
    def forward(self, text):
        # text_embeddings = list of word embedding vecs
        text_embed = self.embeddings(text)

        # Get list that excludes padding token
        lenList = []
        for sample in text:
            lenList.append(self.getLengthSample(sample))
        lenList = np.array(lenList)
        lenList.shape = (len(text),1)
        if str(self.device) == "cuda":
            lenList = torch.from_numpy(lenList).float().cuda()
            # Sum over all vectors in text, take average
            encoded = text_embed.sum(dim=1).cuda()
            encoded /= lenList

            # Pass into 2 hidden linear layers, and take softmax
            logits = self.classifier(encoded)
            return logits
            
        lenList = torch.from_numpy(lenList).float()
        # Sum over all vectors in text, take average
        encoded = text_embed.sum(dim=1)
        encoded /= lenList

        # Pass into 2 hidden linear layers, and take softmax
        logits = self.classifier(encoded)
        return logits

class CNN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim = 300, n_hidden_units = 300, n_filters = 100, filter_sizes = [3,4,5],\
		 dropout = .5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], emb_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], emb_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], emb_dim))
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.conv_0.weight.data.uniform_(-initrange, initrange)
        self.conv_0.bias.data.zero_()
        self.conv_1.weight.data.uniform_(-initrange, initrange)
        self.conv_1.bias.data.zero_()
        self.conv_2.weight.data.uniform_(-initrange, initrange)
        self.conv_2.bias.data.zero_()
    
    def forward(self,text):
        #text = [batch size, sent len]
        embedded = self.embeddings(text)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = nn.functional.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = nn.functional.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = nn.functional.relu(self.conv_2(embedded).squeeze(3))
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = nn.functional.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = nn.functional.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = nn.functional.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

class BertMLP(nn.Module):
    def __init__(self, n_classes, args=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
        )
        self.fc = nn.Linear(768, n_classes)

    def forward(self, x):
        # input: [B, T]
        attention_mask = (x != 0).float()
        rep, _, = self.bert(
            x,
            attention_mask=attention_mask,
        )
        cls_rep = rep[:, 0, :]  # [B, T, H]
        logits = self.fc(cls_rep)
        return logits
