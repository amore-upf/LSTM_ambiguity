import torch
from utils import batchify, get_batch
import pandas as pd
import string
import corpus_utils
import numpy as np

def load_vocabulary_frequencies(vocab_freq_file):
    """
    Load word frequency information, used for negative samples
    :param vocab_freq_file: file with word frequencies
    :return: list of words in frequency quartiles, and mapping from words to quartiles
    """
    vocab_freq = pd.read_csv(vocab_freq_file, sep=',\t', header=None, engine = 'python')
    vocab_freq.columns = ['word', 'freq']
    vocab_freq.word = vocab_freq.word.astype('str')
    vocab_freq['bin'] = pd.qcut(vocab_freq['freq'], 4, labels=["freq1", "freq2", "freq3", "freq4"]) # assign frequency quartile

    freq1 = list(vocab_freq[vocab_freq.bin == 'freq1'].word)
    freq2 = list(vocab_freq[vocab_freq.bin == 'freq2'].word)
    freq3 = list(vocab_freq[vocab_freq.bin == 'freq3'].word)
    freq4 = list(vocab_freq[vocab_freq.bin == 'freq4'].word)

    freq_bins = [freq1, freq2, freq3, freq4]

    word2bin = {}
    for word in freq1:
        word2bin[word] = 0
    for word in freq2:
        word2bin[word] = 1
    for word in freq3:
        word2bin[word] = 2
    for word in freq4:
        word2bin[word] = 3
    return freq_bins, word2bin


def load_languagemodel(language_model, cuda = False):
    """
    Load language model, vocabulary and word frequency information
    :param language_model: path to pre-trained language model
    :param cuda: cpu or gpu
    :return: loaded language model, vocabulary and frequency information (tuple:  list of words in frequency quartiles, and mapping from words to quartiles)
    """
    seed = 1111
    vocab = 'language_model/training_corpus_vocab.txt'
    vocab_freqs = 'language_model/training_corpus_word_frequencies.txt'
    vocab_freq_info = load_vocabulary_frequencies(vocab_freqs)
    if cuda:
        torch.cuda.manual_seed(seed)
    with open(language_model, 'rb') as f:
        language_model = torch.load(f, map_location=lambda storage, loc: storage)
    if cuda:
        language_model.cuda()
    vocab = corpus_utils.Dictionary(vocab)  # Load vocabulary
    return language_model, vocab, vocab_freq_info

def prepare_input(context, index_target, vocab, mode = 'bidir', cuda = False):
    '''
    Adds end-of-sentence and unknown word symbols,  get new indices of target word, turn data into tensor
    :param context:
    :param target:
    :return: list of context words, index of target word, target word
    '''

    context = context.split()
    context[index_target] = '<target> ' + context[index_target]
    context = ' '.join(context)
    context.replace('.', '. <eos> ')
    context = ['<eos>'] + context.split()
    index_target = context.index('<target>')
    del context[index_target]
    context.append('<eos>')
    context.append('<eos>')
    target = context[index_target]
    data_context = torch.cuda.LongTensor(len(context)) if cuda else torch.LongTensor(len(context))
    token = 0
    for word in context:
        if word not in vocab.word2idx:
            data_context[token] = vocab.word2idx["<unk>"]
        else:
            data_context[token] = vocab.word2idx[word]
        token += 1
    data_context = batchify(data_context, 1)
    seq_len = len(data_context)
    data_context, _ = get_batch(data_context, 0, seq_len, mode, evaluation=True)
    return data_context, index_target, target


def extract_vectors(context, index_target, language_model, vocab, to_extract,  cuda = False):
    """
    Extracts internal representations of the language model for target word occurrence
    :param context: context sequence
    :param index_target: index of target word in context
    :param language_model: pre-trained language  model
    :param word_emb_matrix: word embedding matrix
    :param vocab: vocabulary
    :param to_extract: list of representation types to extract
    :param cuda: GPU
    :return: dictionary with extracted vectors per representations type
    """
    data_context, index_target, word = prepare_input(context, index_target, vocab, mode = 'bidir', cuda = cuda)

    word_emb = language_model.encoder.embedding.weight.data[vocab.word2idx[word]]

    hidden = language_model.init_hidden(1)

    # Extract hidden layers (current and predictive) for each layer
    predictive_hidden_layers, hidden_layers = language_model.extract_hidden_layers(data_context, hidden, index_target)

    extracted_vectors =  {i:[] for i in to_extract}

    for vec in to_extract:
        if 'hidden' in vec:
            n_layer = int(vec[-1]) - 1
            if 'current' in vec:
                toadd = hidden_layers[n_layer]
            elif 'predictive' in vec:
                toadd = predictive_hidden_layers[n_layer]
        if vec == 'wordemb':
            toadd = word_emb
        if vec == 'avg_context':

            to_avg = []
            window = 10

            start = index_target - window / 2
            end = index_target + window / 2
            if end >= len(data_context[0]):
                start = start - (end - len(data_context[0]))
                end = len(data_context[0])
            if start < 0:
                end = end - start + 1
                start = 0
            window = data_context[0][int(start):int(end)]
            x = []
            for token in window:
                #Skip unknown words, end of sentence symbols and punctuation
                if vocab.idx2word[token] != "<unk>" and vocab.idx2word[token] != "<eos>" and vocab.idx2word[token] not in string.punctuation + '’”“':
                    to_avg.append(language_model.encoder.embedding.weight.data[token].cpu().detach().numpy())
                    x.append(vocab.idx2word[token])
            toadd = np.average(to_avg, axis = 0)
            toadd = torch.tensor(toadd).cuda() if cuda else torch.tensor(toadd)
            toadd = toadd.squeeze()
        toadd = toadd.cpu().detach().numpy()
        extracted_vectors[vec] = toadd
    return extracted_vectors


def get_best_settings(combinations_inputs, output_folder):
    """
    After training, retrieves for each input type the hyperparameter combination that yield the smallest validation loss
    :param combinations_inputs: input types
    :param output_folder: folder where validation loss information is stored
    :return: dictionary with best combinations for each input type
    """
    valid_scores = pd.read_csv(output_folder + '/valid_scores_models.csv', sep = '\t')
    combinations_inputs_settings = {}
    for combination in combinations_inputs:
        if combination == 'wordemb' or combination == 'avg_context':
            # For unsupervised baselines, set placeholders
            combinations_inputs_settings[combination] = ('-', '-')
        else:
            # Find trained model with smallest validation loss
            data_combination = valid_scores[valid_scores.inputs == combination]
            best_model_row = data_combination.ix[data_combination['valid_loss'].idxmin()]
            combinations_inputs_settings[combination] = (best_model_row['batch_size'], best_model_row['lr'])
    return combinations_inputs_settings


def get_list_inputs_type(phase, n_layers=3):
    """
    Returns list of input types
    :param phase: train or test
    :param n_layers: number of hidden layers
    :return: list of input types
    """
    n = n_layers + 1
    combinations_inputs = []
    combinations_inputs += ['currenthidden' + str(i) for i in range(1, n)]
    combinations_inputs += ['predictivehidden' + str(i) for i in range(1, n)]
    if phase == 'test':
        # Adds unsupervised baselines only at testing time
        combinations_inputs.append('wordemb')
        combinations_inputs.append('avg_context')
    return combinations_inputs