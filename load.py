import torch

from config import MAX_LENGTH

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readVocs():
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open('data/train_x.txt', 'r', encoding='utf-8') as f:
        train_x = f.read().splitlines()
    with open('data/train_y.txt', 'r', encoding='utf-8') as f:
        train_y = f.read().splitlines()
    
    content = zip(train_x, train_y)
    pairs = [[x, y] for x, y in content]

    voc = Voc()
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData():
    voc, pairs = readVocs()
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.n_words)

    return voc, pairs

if __name__ == '__main__':
    loadPrepareData()