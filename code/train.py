from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import dataset
import model

def train(epoch, device):
    mdl.train()
    trainLoss = 0

    loss_json = []
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
        
        loss_json.append({"batch {}".format(i):loss.item()})
        
        if args.interval > 0 and i % args.interval == 0 and not args.silent:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(dl_trn.dataset),
                100.*(args.batch_size*i)/len(dl_trn.dataset),
                loss.item()
            ))
    trainLoss /= len(dl_trn)
    print('* (Train) Epoch: {} | Loss: {:.4f}'.format(epoch, trainLoss))
    return validate_model(device, epoch), loss_json

def validate_model(device, epoch):
    mdl.eval()
    validLoss = 0
    for i,sampleSet in enumerate(dl_val):
        batch = sampleSet[0]
        labels = sampleSet[1]
        batch, labels = batch.to(device), labels.to(device)

        output = mdl(batch)
        loss = criterion(output, labels)
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
        output = mdl(batch)
        outputSet.append(torch.argmax(output, dim=1).detach().cpu().numpy()[0])
        if args.interval > 0 and i % (args.interval * args.batch_size) == 0 and not args.silent:
            print('(Test) Sample: {}/{} ({:.0f}%) '.format(
                i, len(dl_tst.dataset),
                100.*(i)/len(dl_tst.dataset)
            ))
    valLoss = accuracy_score(labelSet, outputSet)
    f1 = f1_score(labelSet, outputSet)
    print('* Best Model Validation Accuracy: {}'.format(valLoss))
    print('* Best Model f1 score: {}'.format(f1))
    return valLoss, f1

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
    parser.add_argument('--bert-tokens', type=bool, default=False)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--train', type=str, default="data/data.csv")
    parser.add_argument('--valid', type=str, default="data/valid.csv")
    parser.add_argument('--test', type=str, default="data/test.csv")
    parser.add_argument('--save-mdl', type=str, default="models/mdl.th")
    parser.add_argument('--data-name', type=str, default="imbd")
    args = parser.parse_args()

    DATA_PATH = args.train
    VALID_PATH = args.valid
    TEST_PATH = args.test
    MODEL_SAV = args.save_mdl
    TRAIN_LOG_SAV = "logs/" + args.data_name + ".json"
    
    dl_trn, vocab, lenc, ldec = dataset.load(batchSize=args.batch_size, seqLen=args.seq_len, \
        path=DATA_PATH, cl=args.clip, voc=None, lenc=None, ldec=None, bertToks=args.bert_tokens)
    dl_val, val_vocab, vlenc, vldec = dataset.load(batchSize =args.batch_size, seqLen = args.seq_len, \
        path = VALID_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec, bertToks=args.bert_tokens)
    dl_tst, tst_vocab, tlenc, tdec = dataset.load(batchSize = 1, seqLen = args.seq_len, \
        path = TEST_PATH, cl=args.clip, voc=vocab, lenc=lenc, ldec=ldec, bertToks=args.bert_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    modeldict = {
        "bert": model.BertMLP(n_classes = len(lenc)),
        "cnn": model.CNN(n_classes = len(lenc), vocab_size = len(vocab), emb_dim = args.embedding_dim, \
            n_hidden_units = args.embedding_dim, n_filters = 100, filter_sizes = [3,4,5], device=device),
        "dan": model.DAN(n_classes = len(lenc), vocab_size = len(vocab), emb_dim = args.embedding_dim, \
            n_hidden_units = args.embedding_dim, device=device)
    }
    mdl = modeldict[args.model].to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=args.lr)

    json_obj = {"name":args.data_name}
    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss, json_loss = train(epoch, device)
        json_obj["epoch {}".format(epoch)] = json_loss
        if loss < best_loss:
            best_loss = loss
            print('* Saved')
            torch.save(mdl.state_dict(), MODEL_SAV)
    
    accuracy, f1 = test_model(device)
    json_obj["accuracy"] = accuracy 
    json_obj["macro_f1"] = f1
    with open(TRAIN_LOG_SAV, "w") as write_file:
        json.dump(json_obj, write_file)
