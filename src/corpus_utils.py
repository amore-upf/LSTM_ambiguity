import os
import torch


class Dictionary(object):

    def __init__(self, vocab_file=None):
        self.word2idx = {}
        self.idx2word = []

        if vocab_file:
            with open(vocab_file, "rt") as f:
                for l in f:
                    self.add_word(l.strip())

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def tokenize(self, path, eval=True):
        """ Tokenizes a text file.
            Assumes that the text has already <eos> symbols in place and is preprocessed with <unk>s
            If eval=True, assumes that all words not in vocab will be converted to <unk> """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if not eval:
                    for word in words:
                        self.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.word2idx:
                        ids[token] = self.word2idx["<unk>"]
                    else:
                        ids[token] = self.word2idx[word]
                    token += 1

        return ids


class Corpus(object):
    def __init__(self, path, dictionary=None):
        vocab_path = os.path.join(path, 'vocab.txt')
        if dictionary:
            self.dictionary = dictionary
            fix_vocab = True
        elif os.path.isfile(vocab_path):
            self.dictionary = Dictionary(vocab_path)
            fix_vocab = True
        else:
            self.dictionary = Dictionary()
            fix_vocab = False
        print(fix_vocab, self.dictionary)
        self.train = self.dictionary.tokenize(os.path.join(path, 'train.txt'), eval=fix_vocab)
        self.valid = self.dictionary.tokenize(os.path.join(path, 'valid.txt'), eval=fix_vocab)
        self.test = self.dictionary.tokenize(os.path.join(path, 'test.txt'), eval=fix_vocab)

        #if not os.path.isfile(vocab_path):
        #    with open(vocab_path, "w") as f:
        #        f.write("\n".join([w for w in self.dictionary.idx2word]))

import sys

def main():
    files = sys.argv[1:]
    dictionary = Dictionary()
    for f in files:
        dictionary.tokenize(f, eval=False)
    open("vocab.txt", "w").write("\n".join([w for w in dictionary.idx2word]))

if __name__ == "__main__":
    main()
