from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import dataset
import bertmodels as bertmodels

from transformers.tokenization_bert import BertTokenizer, WordpieceTokenizer
from transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification


DATA_PATH = "data/data.csv"
VALID_PATH = "data/valid.csv"
TEST_PATH = "data/test.csv"
MODEL_SAV = "models/bert.th"

def train(epoch, device):
    mdl.train()
    trainLoss = 0
    for i,sampleSet in enumerate(dl_trn):
        batch = sampleSet[0]
        labels = sampleSet[1]
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()

        output = mdl(input_ids=batch, labels=labels)
        loss = output[0]
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
        if args.interval > 0 and i % args.interval == 0 and not args.silent:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(dl_trn.dataset),
                100.*(args.batch_size*i)/len(dl_trn.dataset),
                loss.item()
            ))
    trainLoss /= len(dl_trn)
    print('* (Train) Epoch: {} | Loss: {:.4f}'.format(epoch, trainLoss))
    return validate_model(device, epoch)

def validate_model(device, epoch):
    mdl.eval()
    validLoss = 0
    for i,sampleSet in enumerate(dl_val):
        batch = sampleSet[0]
        labels = sampleSet[1]
        batch, labels = batch.to(device), labels.to(device)
        output = mdl(input_ids = batch, labels = labels)
        loss = output[0]
        validLoss += loss.item()
        
        if args.interval > 0 and i % args.interval == 0 and not args.silent:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(dl_val.dataset),
                100.*(args.batch_size*i)/len(dl_val.dataset),
                loss.item()
            ))
    validLoss /= len(dl_val)
    print('* (Validation) Epoch: {} | Loss: {:.4f}'.format(epoch, validLoss))
    return validLoss
    
def test_model(device):
    check = torch.load(MODEL_SAV)
    mdl.load_state_dict(check)
    mdl.eval()
    labelSet = []
    outputSet = []
    for i,sampleSet in enumerate(dl_tst):
        batch = sampleSet[0]
        labels = sampleSet[1]
        labelSet.append(labels.cpu().numpy()[0])
        batch, labels = batch.to(device), labels.to(device)
        output = mdl(input_ids = batch, labels = labels)
        outputSet.append(torch.argmax(output[1], dim=1).detach().cpu().numpy()[0])
        if args.interval > 0 and i % (args.interval * args.batch_size) == 0 and not args.silent:
            print('(Test) Sample: {}/{} ({:.0f}%) '.format(
                i, len(dl_tst.dataset),
                100.*(i)/len(dl_tst.dataset)
            ))
    valLoss = accuracy_score(labelSet, outputSet)
    print('* Best Model Validation Accuracy: {}'.format(valLoss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq-len', type=int, default=50)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--silent', type=bool, default=False)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dl_trn, vocab, lenc, ldec = dataset.load_bert(batchSize=args.batch_size, seqLen=args.seq_len, \
        path=DATA_PATH, cl=args.clip, voc=None, lenc=None, ldec=None, tok=tokenizer)
    dl_val, val_vocab, vlenc, vldec = dataset.load_bert(batchSize =args.batch_size, seqLen = args.seq_len, \
        path = VALID_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec, tok=tokenizer)
    dl_tst, tst_vocab, tlenc, tdec = dataset.load_bert(batchSize = 1, seqLen = args.seq_len, \
        path = TEST_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec, tok=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    mdl = bertmodels.BertBinClf.from_pretrained('bert-base-uncased').to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=args.lr)
    
    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, device)
        if loss < best_loss:
            best_loss = loss
            print('* Saved')
            torch.save(mdl.state_dict(), MODEL_SAV)
    
    test_model(device)
