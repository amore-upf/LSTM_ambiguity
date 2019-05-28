
import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):

    def __init__(self, w2idx, emb_size, pretrained_embs=None, fixed_embs=False):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(len(w2idx), emb_size)
        self.emb_size = emb_size

        if pretrained_embs is not None:
            embedding_weights = load_embeddings(pretrained_embs, w2idx, emb_size)
            self.embedding.weight.data = torch.FloatTensor(embedding_weights)
            if fixed_embs:
                self.embedding.weight.requires_grad = False
        else:
            self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        return self.embedding(input)


def load_embeddings(filename, w2idx, emb_size):
    """ Works for fasttext-type of input files """

    emb_matrix = np.zeros((len(w2idx), emb_size)) + np.random.rand(emb_size)
    # there is no embedding for "unk" in pre-trained embs, so it will be random
    loaded = 1

    with open(filename, "rt") as f:
        for i, line in enumerate(f):
            if i == 0:
                ntokens, size = line.split()
                if int(size) != emb_size:
                    raise Exception("The size of embeddings in the file is different "
                                    "from the embedding size of the model")
                continue
            word, *values = line.split()
            if word in w2idx:
                emb_matrix[w2idx[word]] = np.array(values)
                loaded += 1
            if word == "</s>":    # </s> is fasttext equivalent of <eos>
                emb_matrix[w2idx["<eos>"]] = np.array(values)
                loaded += 1

            if loaded >= len(w2idx):
                break

    print("Loaded N out of total vocab", loaded, len(w2idx))

    return emb_matrix


class Decoder(nn.Module):

    def __init__(self, ntoken, hidden_size, tie_weights="", encoder=None):
        super(Decoder, self).__init__()

        self.linear = None
        if tie_weights == "standard":
            if hidden_size != encoder.emb_size:
                raise ValueError('When using the tied flag, hidden size of last RNN layer must be equal to emb size')
            self.decoder = nn.Linear(encoder.emb_size, ntoken, bias=False)
            self.decoder.weight = encoder.embedding.weight
        elif tie_weights == "plusL":
            self.linear = nn.Linear(hidden_size, encoder.emb_size, bias=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)
            self.decoder = nn.Linear(encoder.emb_size, ntoken, bias=False)
            self.decoder.weight = encoder.embedding.weight
        else:
            # no tying
            self.decoder = nn.Linear(hidden_size, ntoken, bias=False)
            self.init_weights()

    def forward(self, input):
        if self.linear:
            input = self.linear(input)
        return self.decoder(input)

    def init_weights(self):
        initrange = 0.1
        if self.linear:
            self.linear.weight.data.uniform_(-initrange / 10, initrange / 10)
        self.decoder.weight.data.uniform_(-initrange, initrange)


class StackedRNN(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_sizes, dropout):
        super(StackedRNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.output_size = hidden_sizes[-1]

        self.nlayers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()

        self.drop = nn.Dropout(dropout)

        sizes = [input_size, ] + hidden_sizes

        for i in range(1, self.nlayers + 1):
            input_size = sizes[i-1]
            hidden_size = sizes[i]

            if rnn_type in ['LSTM', 'GRU']:
                rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--rnncell` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity=nonlinearity)
            self.layers.append(rnn)

    def forward(self, input, hidden):
        new_h = []
        for rnn, h in zip(self.layers, hidden):
            output, h = rnn(input, h)
            new_h.append(h)
            input = self.drop(output)

        return output, new_h

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = []
        for nhid in self.hidden_sizes:
            if self.rnn_type == 'LSTM':
                hidden.append((weight.new(1, bsz, nhid).fill_(0.01),
                               weight.new(1, bsz, nhid).fill_(0.01)))
            else:
                hidden.append(weight.new_full((1, bsz, nhid), 0.01))
        return hidden
