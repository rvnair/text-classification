import collections
import csv 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_bert import BertTokenizer, WordpieceTokenizer

class Vocab:
    # Corpus is a list of samples (list of lists)
    def __init__(self, corpus, clip):
        self.words = self.build_words(corpus, clip)

        # Word mapped to uuid
        # Reserve padding mapped to 0
        self.encoding = {w:i for i,w in enumerate(self.words, 2)}
        self.decoding = {i:w for i,w in enumerate(self.words, 2)}

    def build_words(self, corpus, clip = 1):
        vocab = collections.Counter()
        
        for sample in corpus:
            vocab.update(sample)
        
        for word in list(vocab.keys()):
            if vocab[word] < clip:
                vocab.pop(word)
        
        return list(sorted(vocab.keys()))

    def size(self):
        return len(self.words) + 2

# Class on a list of samples
# Needs to support indexing into samples for Dataset
# Loads and provides index for a given sample
class Corpus(Dataset):
    def __init__(self, seqLen = 50, path = "", clip=5, vocab=None, labelEnc = None, labelDec = None):
        self.seqLen = seqLen
        self.samples,self.labels = self.loadCSV(path)
        self.labelEnc, self.labelDec = self.genLabelMap(self.labels) 
        if not labelEnc is None and not labelDec is None:
            self.labelEnc, self.labelDec = labelEnc,labelDec
        self.vocab =  Vocab(self.samples, clip) if vocab is None else vocab

    def loadCSV(self, path):
        samples = []
        labels = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    review = row[0].lower().split(' ')
                    samples.append(review[:self.seqLen])
                    labels.append(row[1])
                line_count += 1
        return (samples, labels)
    
    def genLabelMap(self, labels):
        labelSet = set(labels)
        enc = {l:i for i,l in enumerate(labelSet)}
        dec = {i:l for i,l in enumerate(labelSet)}
        return enc,dec

    def pad(self, sample):
        l,r = 0, self.seqLen - len(sample)
        return np.pad(sample, (l,r), 'constant')

    def encode(self, sample):
        enc = self.vocab.encoding
        return np.array([enc.get(w,1) for w in sample])
    
    def decode(self, sample):
        dec = self.vocab.decoding
        return [dec.get(w,"<UNK>") for w in sample]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.pad(self.encode(self.samples[i]))), self.labelEnc[self.labels[i]]

class BertCorpus(Corpus):
    def __init__(self, seqLen = 50, path = "", clip=5, vocab=None, labelEnc = None, labelDec = None, tokenizer=None):
        super(BertCorpus, self).__init__(seqLen, path, clip, vocab, labelEnc, labelDec)
        self.tokenizer = tokenizer if tokenizer != None else BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __getitem__(self, i):
        text = self.decode(self.pad(self.encode(self.samples[i])))
        return torch.tensor(self.tokenizer.encode(text)), torch.tensor([self.labelEnc[self.labels[i]]])

# Returns a data loader as well as a vocabulary
def load(batchSize, seqLen, path, cl, voc, lenc, ldec):
    dataset = Corpus(seqLen, path, clip = cl, vocab = voc, labelEnc = lenc, labelDec = ldec)
    return (DataLoader(dataset, batchSize, shuffle = True), dataset.vocab, dataset.labelEnc, dataset.labelDec)

def load_bert(batchSize, seqLen, path, cl, voc, lenc, ldec, tok):
    dataset = BertCorpus(seqLen, path, clip = cl, vocab = voc, labelEnc = lenc, labelDec = ldec, tokenizer=tok)
    return (DataLoader(dataset, batchSize, shuffle = True), dataset.vocab, dataset.labelEnc, dataset.labelDec)
