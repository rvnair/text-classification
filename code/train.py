from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import dataset
import model

DATA_PATH = "data/data.csv"
VALID_PATH = "data/valid.csv"
TEST_PATH = "data/test.csv"
MODEL_SAV = "models/dan.th"

def train(epoch, device):
    mdl.train()
    trainLoss = 0
    for i,sampleSet in enumerate(dl_trn):
        batch = sampleSet[0]
        labels = sampleSet[1]
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()

        output = mdl(batch)
        loss = criterion(output, labels)
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
        
        if args.interval > 0 and i % args.interval == 0:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(dl_trn.dataset),
                100.*(args.batch_size*i)/len(dl_trn.dataset),
                loss.item()
            ))
    trainLoss /= len(dl_trn)
    print('* (Train) Epoch: {} | Loss: {:.4f}'.format(epoch, trainLoss))
    return trainLoss

def test_model(device):
    check = torch.load(MODEL_SAV)
    mdl = model.DAN(n_classes = 2, vocab_size = vocab.size(), emb_dim = args.embedding_dim, \
        n_hidden_units = args.embedding_dim)
    mdl = mdl.to(device)
    mdl.load_state_dict(check)
    mdl.eval()
    labelSet = []
    outputSet = []
    for i,sampleSet in enumerate(dl_tst):
        batch = sampleSet[0]
        labels = sampleSet[1]
        labelSet.append(labels.cpu().numpy()[0])
        batch, labels = batch.to(device), labels.to(device)
	#with torch.no_grad():
        output = mdl(batch)
        outputSet.append(torch.argmax(output, dim=1).detach().cpu().numpy()[0])
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
    args = parser.parse_args()

    dl_trn, vocab, lenc, ldec = dataset.load(batchSize=args.batch_size, seqLen=args.seq_len, path=DATA_PATH, cl=args.clip, voc=None, lenc=None, ldec=None)
    dl_val, val_vocab, vlenc, vldec = dataset.load(batchSize = 1, seqLen = args.seq_len, path = VALID_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec)
    dl_tst, tst_vocab, tlenc, tdec = dataset.load(batchSize = 1, seqLen = args.seq_len, path = TEST_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    mdl = model.DAN(n_classes = 2, vocab_size = vocab.size(), emb_dim = args.embedding_dim, \
        n_hidden_units = args.embedding_dim, device=device).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=args.lr)

    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, device)
        if loss < best_loss:
            best_loss = loss
            print('* Saved')
            torch.save(mdl.state_dict(), MODEL_SAV)
    
    # validate(device)
