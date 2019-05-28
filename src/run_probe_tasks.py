import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message='source.*has changed')
import random
import torch.nn as nn
import torch
import argparse
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from probetasks_utils import extract_vectors, get_best_settings, load_languagemodel, get_list_inputs_type
import os
import corpus_utils
import pickle
import math
from itertools import product


class NonLinear_Transformation(nn.Module):
    """
    Non-linear transformation (used for diagnostic models)
    Linear + Tanh
    From input size to size of word embeddings (output size)
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.hidden = nn.Sequential(nn.Linear(input_size, output_size), nn.Tanh())

    def forward(self, input):
        output = self.hidden(input)
        return output


def cosine(v, t ):
    """
    Pair-wise cosine vector - matrix
    :param v:  vector
    :param t: matrix
    :return: pair-wise cosine scores
    """
    v = v / torch.norm(v, dim=0, keepdim=True)
    t = t / torch.norm(t, dim=0, keepdim=True)
    sims = torch.matmul(v, t.t())
    return sims


def nearest_neighbours(vector, word_emb_matrix, vocab, n=10, with_scores=False, into_words = False):
    '''
    Get nearest neighbors of a vector wrt word embedding matrix
    :param vector: vector
    :param word_emb_matrix: word embedding matrix
    :param vocab: vocabulary
    :param n: number of neighbors; if 'all' gives back all scores
    :param with_scores: return also similarity scores
    :into_words: return words as forms and not as indices
    :return: list of neighbors , and list of scores if with_scores is set to True
    '''
    scores = cosine(vector, word_emb_matrix).squeeze(0)
    if n == 'all':
        n = len(vocab.idx2word)
    scores = scores.cpu().detach().numpy()
    nns = list(np.argsort(scores, axis=0))[::-1][:n]
    scores = [scores[i] for i in nns]
    if into_words:
        nns = [vocab.idx2word[i] for i in nns]
    if with_scores:
        return nns, scores
    else:
        return nns


def get_similarity_scores_and_neighbors(y_hat, y, vocab, criterion, word_emb_matrix, cuda = False):
    """
    Computes similarity scores between predicted and output representations, and list of nearest neighbors (predicted and output)
    :param y_hat: predicted vectors
    :param y: target vectors
    :param vocab: vocabulary
    :param criterion: loss (cosine embedding loss)
    :param word_emb_matrix: word embedding matrix
    :param cuda: gpu
    :return: similarity scores, neighbors of target, neighbors of predicted
    """
    total = len(y)
    neighbours_target = []
    neighbours_predicted = []
    sims = []
    for i in range(total):
        target_vector = y[i].unsqueeze(0)
        predicted_vector = y_hat[i].unsqueeze(0)
        nns_target = nearest_neighbours(target_vector, word_emb_matrix, vocab, n = 10, with_scores = False, into_words= True)
        nns_predicted = nearest_neighbours(predicted_vector, word_emb_matrix, vocab, n = 10, with_scores = False, into_words = True)
        neighbours_predicted.append(nns_predicted)
        neighbours_target.append(nns_target)
        target = torch.ones(1).cuda() if cuda else torch.ones(1)
        sim = 1 - criterion(target_vector, predicted_vector, target)
        sims.append(float(sim))
    return sims, neighbours_target, neighbours_predicted



def add_negative_examples(X, Y, targets, word_emb_matrix, freqs_info, labels, words, vocab, task,
                          cuda = False, shuffle = True):
    """
    Adds negative instances to training data:
    For each datapoint, 5 words from the same frequency quartile of the targert word are sampled;
    their vectors is used as negative instance.
    By default, the whole training data is shuffled.
    :param X: input data
    :param Y: output data
    :param targets: targets for cosine embedding loss ; 1 if positive instance, 0 if negative
    :param word_emb_matrix: word embedding matrix
    :param freqs_info: frequency information about the words in the vocabulary
    :param labels: list of target words in word task, list of substitutes in other tasks
    :param words: list of target words
    :param vocab: vocabulary
    :param task: word, sub or word+sub
    :param cuda: gpu
    :param shuffle: if True data is shuffled
    :return: input data, output data, targets --> now with negative instances
    """
    freq_bins, word2bin = freqs_info
    len_data = len(X)
    n_to_sample = 5
    for i in range(len_data):
        word = words[i]
        x = X[i]
        if task == 'word':
            to_sample = n_to_sample + 1
        else:
            to_sample = n_to_sample + len(labels[i]) + 1
        negatives = random.sample(freq_bins[word2bin[vocab.idx2word[word]]], to_sample )
        for n in negatives:
            if type(n) != str:
                print("Error in sampling: ", negatives)
        negatives = [vocab.word2idx[n] for n in negatives]
        if word in negatives:
            negatives.remove(word) # exclude word itself (always)
        if task != 'word':
            for n in negatives:
                if n in list(labels[i]):
                    negatives.remove(n) #if sub or word+sub task: exclude the 5 substitutes
        negatives = negatives[:n_to_sample]
        for n in negatives:
            vec = word_emb_matrix[n].unsqueeze(0)
            vec = vec.cuda() if cuda else vec
            Y = torch.cat((Y, vec))
            X = torch.cat((X, x.unsqueeze(0)))
            target = - torch.ones(1).cuda() if cuda else - torch.ones(1)
            targets = torch.cat((targets, target))
            words.append(n)
    if shuffle:
        #shuffling data
        perm = torch.randperm(len(Y))
        X = X[perm][:,]
        Y = Y[perm]
        targets = targets[perm]
    return X, Y, targets

def train_epoch(X, Y, model, optimizer, criterion, vocab, task, labels, words, word_emb_matrix, freqs_info,
                batch_size= 20, cuda = False):
    """
    Runs training epoch: deploys the diagnostic model on data and backpropagates batch by batch
    :param X: training input data
    :param Y: training output data
    :param model: model to be trained
    :param optimizer: optimizer
    :param criterion: loss
    :param vocab: vocabulary
    :param task: word, sub or word+sub
    ::param labels: list of target words in word task, list of substitutes in other tasks
    :param words: list of target words
    :param word_emb_matrix: word embedding matrix
    :param freqs_info: frequency information about the words in the vocabulary
    :param batch_size: batch size
    :param cuda: gpu
    :return: mean loss, model
    """
    model.train()
    losses = []
    criterion_eval = nn.CosineEmbeddingLoss(reduction='none')
    targets = torch.ones(len(Y)).cuda() if cuda else torch.ones(len(Y)) # 1 for all positive instances (to be used for cosine embedding loss)
    X, Y, targets = add_negative_examples(X, Y, targets, word_emb_matrix, freqs_info, labels, words, vocab, task,
                                                 cuda = cuda)

    for i in range(0, X.size(0), batch_size): # deploy and backpropagate batch by batch
        x_batch = X[i:i + batch_size, :]
        y_batch = Y[i:i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)
        target_batch = targets[i:i+batch_size]
        optimizer.zero_grad()
        y_hat_batch = model(x_batch)

        loss = criterion(y_hat_batch, y_batch, target_batch)
        losses += list(criterion_eval(y_hat_batch, y_batch, target_batch).data.cpu().detach().numpy())

        # backward step
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses) # mean cosine distance
    return avg_loss, model

def evaluate_model(X, Y, model, vocab, word_emb_matrix = None, batch_size= 50, with_info = False, cuda = False,
                   unsupervised = False):
    """
    Deploys and evaluate diagnostic model on target data (without backpropagation step)
    :param X: Input data
    :param Y: output data
    :param model: model
    :param vocab: vocabulary
    :param word_emb_matrix: word embedding matrix
    :param batch_size: batch size
    :param with_info: if True, returns neighbors and similarities for each datapoint
    :param cuda: gpu
    :param unsupervised: if True, computes the similarity between input and output representations (used for baselines, whereas input size = output size)
    :return: if with_info is True: mean loss, list of neighbors of target representations, , list of neighbors of predicted representations, similarities scores; else, mean loss
    """
    if not unsupervised:
        model.eval()
    else:
        batch_size = 50

    losses = []
    neighbours_target = []
    neighbours_predicted =[]
    sims = []

    criterion = nn.CosineEmbeddingLoss(reduction = 'none')

    for i in range(0, X.size(0), batch_size): #deploy and collect predictions batch by batch
        x_batch = X[i:i + batch_size, :]
        y_batch = Y[i:i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        target = torch.ones(len(y_batch)).cuda() if cuda else torch.ones(len(y_batch))
        if not unsupervised:
            y_hat_batch = model(x_batch)
            losses += list(criterion(y_hat_batch, y_batch, target).data.cpu().detach().numpy())
        else:
            losses += list(criterion(x_batch, y_batch, target).data.cpu().detach().numpy())

        if with_info:
            if not unsupervised:
                sims_batch, neighbours_target_batch, neighbours_predicted_batch = get_similarity_scores_and_neighbors(y_hat_batch, y_batch, vocab, criterion, word_emb_matrix, cuda = cuda)
            else:
                # Directly evsaluates input representations (used for baselines, whereas input size = output size)
                sims_batch, neighbours_target_batch, neighbours_predicted_batch = get_similarity_scores_and_neighbors(x_batch, y_batch, vocab, criterion, word_emb_matrix, cuda=cuda)

            neighbours_target += neighbours_target_batch
            neighbours_predicted += neighbours_predicted_batch
            sims += sims_batch
    avg_loss = np.mean(losses)

    if with_info:
        return avg_loss, neighbours_target, neighbours_predicted, sims
    else:
        return avg_loss

def get_target(task, word, vocab, word_emb_matrix, substs):
    """
    Returns the target representation, depending on task
    word: embedding of target word
    sub: average of embeddings of substitute words
    word+sub: average of embeddings of subtitute words and target word
    :param task: word, sub or word+sub
    :param word: target word
    :param vocab: vocabulary
    :param word_emb_matrix: word embedding matrix
    :param substs: list of substitute words
    :return: target vector
    """
    word_emb = word_emb_matrix[vocab.word2idx[word]]
    word_emb = word_emb.cpu().detach().numpy()
    if task == 'word':
        target_vector = word_emb
        target = vocab.word2idx[word]
    elif task == 'sub' or task == 'word+sub':
        subst_vecs = []
        substs = substs[:5]
        for subst in substs:
            subst_vec = word_emb_matrix[vocab.word2idx[subst]].cpu().detach().numpy()
            subst_vecs.append(subst_vec)
        if task == 'word+sub':
            subst_vecs.append(word_emb)
        target_vector = np.average(subst_vecs, axis=0)
    return target_vector


def get_inputs_and_targets(dataset, language_model, vocab, combinations, task = 'sub', cut_point = None, cuda = False):
    """
    Deploys the language model on LexSub data and collects input and output representations for probe tasks
    :param dataset: Pandas dataframe with LexSub data
    :param language_model: pre-trained language model
    :param vocab: vocabulary
    :param combinations: list of inputs types
    :param task: word, sub, word+sub
    :param cut_point: number of datapoints to consider; by default, all are used
    :param cuda: gpu
    :return: dictionary with input data for combinations (tuple: torch matrix, input size), output matrix, indices of datapoints in vocabulary, list of words, list of labels
    """

    language_model.eval()
    Y_vecs = []
    count_datapoints = 0
    combinations_X = {combination:[] for combination in combinations}
    datapoints_invocab = []
    words = [] # words of datapoints
    labels = [] # word task: word indices (identical to words); sub and word+sub task: list of substitute word indices (to be used for excluding them in negative samples)
    for index, row in dataset.iterrows():
        word = row.word
        context = row.context
        in_vocab = word in vocab.word2idx # word in vocabulary
        target = eval(row['substitutes']) # list of substitutes
        substs_invocab = []
        for s in target: #at least 5 substitute words in the vocabulary
            if s in vocab.word2idx:
                substs_invocab.append(s)
        in_vocab = in_vocab and len(substs_invocab) >= 5
        if in_vocab:
            datapoints_invocab.append(index)
            words.append(vocab.word2idx[word]) # list of target words
            # labels: target words for word task; else substitute words
            if task == 'word':
                labels.append(vocab.word2idx[word])
            else:
                labels.append([vocab.word2idx[s] for s in substs_invocab[:5]])
            index_target = row.idx_in_context
            # Extract internal representations of language model
            vectors_datapoint = extract_vectors(context, index_target, language_model, vocab, combinations, cuda=cuda)
            for vec in combinations:
                combinations_X[vec].append(vectors_datapoint[vec])
            # Get target representations depending on the task
            target_vector = get_target(task, word, vocab, language_model.encoder.embedding.weight.data, substs_invocab)
            if count_datapoints == 0:
                Y_vecs = np.array([target_vector])
            else:
                Y_vecs = np.append(Y_vecs, [target_vector], axis = 0)
            count_datapoints += 1
            # If trial run, stops when cut point is reached
            if cut_point != None:             # break for trial runs
                if count_datapoints == cut_point:
                    break
    for c in combinations_X:
        # For each input type, create Torch input data matrix
        inputs = combinations_X[c]
        input_size = len(inputs[0]) # dimensionality of input vector
        X = np.stack(inputs)
        X = torch.cuda.FloatTensor(X) if cuda else torch.FloatTensor(X)
        combinations_X[c] = (X, input_size)
    Y_vecs = np.ascontiguousarray(Y_vecs)
    Y_vecs = torch.cuda.FloatTensor(Y_vecs) if cuda else torch.FloatTensor(Y_vecs) # Torch output data matrix
    return combinations_X, Y_vecs, datapoints_invocab, words, labels


def train_model(model, setting, task, output_folder, X, Y, labels, words, X_valid, Y_valid, optimizer, criterion, word_emb_matrix, freqs_info, vocab,
             cuda = False, epochs = 'early_stopping'):
    """
    Trains diagnostic model for input type and task
    :param model: diagnostic model (nonlinear transformation)
    :param setting: batch size, learning rate
    :param task: word, sub, word+sub
    :param output_folder: folder where to output training information and saving models
    :param X: training input data
    :param Y:  training output data
    :param labels: list of words for word, list of substitutes for other tasks
    :param words: list of target words
    :param X_valid:  validation input data
    :param Y_valid: validation output data
    :param optimizer: optimizer
    :param criterion: loss
    :param word_emb_matrix: word embedding matrix
    :param freqs_info: frequency information about words in vocabulary
    :param vocab: vocabulary
    :param cuda: gpu
    :param epochs: number of epochs; by default, it uses early stopping based on validation loss
    :return: validation loss after training
    """
    train_losses = []
    valid_losses = []

    batch_size, learning_rate = setting

    model_name = '_'.join(['batch' + str(batch_size), 'lr' + str(learning_rate)])
    output_model = output_folder + model_name + '.pt'
    output_file = open(output_folder + model_name + '.log', 'w')
    output_file.write('\t'.join(['epoch', 'train_loss', 'valid_loss'])+ '\n')
    max_epochs = 2000
    eval_after = 5 # evaluating whether stopping training after 5 epochs
    early_stopping = epochs == 'early_stopping'
    if not early_stopping:
        # If not early_stopping, until maximum epoch is reached
        max_epochs = eval(epochs)
        eval_after = 1
    i = 0
    #Epoch
    for epoch in range(max_epochs):
        #Training epoch
        train_loss_epoch, model = train_epoch(X, Y, model, optimizer, criterion, vocab, task, labels, words, word_emb_matrix, freqs_info,
                                        batch_size = batch_size, cuda = cuda)
        train_losses.append(train_loss_epoch)
        # Deploy on validation
        valid_loss_epoch = evaluate_model(X_valid, Y_valid, model, vocab, with_info = False, batch_size = batch_size, cuda = cuda)
        valid_losses.append(valid_loss_epoch)

        output_file.write('\t'.join([str(epoch), str(train_loss_epoch), str(valid_loss_epoch)]) + '\n')

        # Decide if continuing training (if validation loss hasn't decreased in the last 5 epochs)
        if i % eval_after == 0 or epoch == max_epochs - 1:
            if early_stopping:
                if epoch != 0:
                    if valid_loss_epoch < current_valid_loss:
                        # Saving best model
                        torch.save(model, output_model)
                        epoch_saved = i
                        current_valid_loss = valid_loss_epoch
                    else:
                        break
                else:
                    torch.save(model, output_model)
                    epoch_saved = i
                    current_valid_loss = valid_loss_epoch
            else:
                current_valid_loss = valid_loss_epoch
        i += 1
    if not early_stopping:
        # Saving last model
        torch.save(model, output_model)
        epoch_saved = i
    output_file.write('Model saved at epoch ' + str(epoch_saved))
    return current_valid_loss



def test_model(model, setting, combination, test_dataset, X_test, Y_test, output_folder_tmp, invocab, vocab,
            acc_file = None, cut_point= None, cuda = False, save_info = False, word_emb_matrix= None, unsupervised = False):
    """
    Tests diagnostic model
    :param model: trained model
    :param setting: batch size, learning rate
    :param combination: input type
    :param test_dataset: Pandas dataframe with test data
    :param X_test: testing input data
    :param Y_test: testing output data
    :param output_folder_tmp: folder where to store scores and labels file
    :param invocab: datapoints in vocabulary
    :param vocab: vocabulary
    :param acc_file: file where to write scores
    :param cut_point: number of datapoints to consider; by default, all are used
    :param cuda: gpu
    :param save_info: if True, it saves to a csv file neighbors and scores information
    :param word_emb_matrix: word embedding matrix
    :param unsupervised: if True, input vectors are directly evaluated, with no diagnostic model (used for baselines, whereas input size = output size)
    :return: test loss
    """
    if not unsupervised:
        if cuda:
            model.cuda()
        model.eval()

    batch_size, learning_rate = setting

    if save_info:
        #get scores and neighbors on test data
        test_loss, neighbours_target, neighbours_predicted, sims = evaluate_model(X_test, Y_test, model, vocab,
                                                                                  word_emb_matrix= word_emb_matrix, batch_size = batch_size, with_info = True, cuda = cuda, unsupervised = unsupervised)
        output_file = output_folder_tmp + 'labels_test.csv'
        # Filters data based on vocabulary coverage and cut point (if any)
        test_dataset = test_dataset[test_dataset.index.isin(invocab)]
        if cut_point != None:
            test_dataset = test_dataset.head(cut_point)
        #Writes test information (for each datapoint) on csv file [for further analyses]
        test_dataset['nns_target'] = neighbours_target
        test_dataset['nns_predicted'] = neighbours_predicted
        test_dataset['sim_target_predicted'] = sims
        test_dataset.to_csv(output_file, sep='\t')
        test_cosine = np.mean(sims), np.std(sims)
        if acc_file != None:
            if unsupervised: combination = combination + "_unsupervised"
            # Saves scores
            acc_file.write('\t'.join([combination, str(batch_size), str(learning_rate), str(test_loss), str(test_cosine[0]),  str(test_cosine[1])]) + '\n')
    else:
        test_loss = evaluate_model(X_test, Y_test, model, vocab, batch_size=batch_size, with_info=False, cuda=cuda)

    return test_loss


def get_probe_task_data(phases, task, combinations_inputs, cuda = False, cut_point = None):
    """
    Loads probe task data for a set of phases (e.g., train).
    Input data are derived running the language model on the LexSub data and extracting representations. Target data are derived depending on the task.
    If data already exists in pkl file, loads them; else runs language model on LexSub data and saves them.
    :param phases: list of phases e.g. ['train', 'valid']
    :param task: word, sub, word+sub
    :param combinations_inputs: list of input types
    :param cuda: gpu
    :param cut_point: number of datapoints considered; by default, all
    :return:
    """
    data = {phase:{} for phase in phases}
    for phase in phases:
        data[phase]['dataframe'] = pd.read_csv('data/probe_tasks/'+ phase+'_data.csv', index_col=0)
        data_dir =  'data/probe_tasks/' + task + '/'
        data[phase]['data_path'] = data_dir + phase + '.pkl' # Path to pkl file with probe tasks data (extracted from language model)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

    language_model = 'language_model/model.pt'

    language_model, vocab, freqs_info = load_languagemodel(language_model, cuda=cuda)


    for phase in phases:
        #If pkl file with data does not  exists, extract and save data
        if not os.path.isfile(data[phase]['data_path']):
            print('Extracting ' + phase + ' data ...')
            combinations_X, Y_vecs, invocab, words, labels = get_inputs_and_targets(data[phase]['dataframe'], language_model, vocab, combinations_inputs,
                                                                               task=task, cut_point=cut_point, cuda=cuda)

            print('Saving ' + phase + ' data ...')
            pickle.dump((combinations_X, Y_vecs, invocab, words, labels), open(data[phase]['data_path'], 'wb'))
        else:
            #Load data
            print('Loading ' + phase + ' data ...')
            combinations_X, Y_vecs, invocab, words, labels = pickle.load(open(data[phase]['data_path'], 'rb'))
        data[phase]['data'] = combinations_X, Y_vecs, invocab, words, labels
    word_emb_matrix = language_model.encoder.embedding.weight.data
    language_model = ''
    return data, word_emb_matrix, vocab, freqs_info

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true", default=False,
                        help = 'run on GPU') #cpu or gpu
    parser.add_argument('--task', action="store", default='sub', type = str,
                        help = 'probe task (word, sub, word+sub)')

    parser.add_argument('--phase', action='store', default='train', type = str,
                        help = 'phase (train or test)') #train or test

    parser.add_argument('--n_datapoints', action='store', default=None,
                        help = 'number of datapoints to consider; by default, all are used')  # cut the dataset for trial
    parser.add_argument('--n_epochs', action='store', default='early_stopping',
                        help = 'number of training epochs; by default, early stopping based on validation loss')


    args = parser.parse_args()
    cut_point = args.n_datapoints
    if cut_point != None:
        cut_point = eval(args.n_datapoints)
    phase = args.phase
    epochs = args.n_epochs
    cuda = args.cuda
    task = args.task

    output_folder = 'probe_tasks_output/' + task + '/'

    combinations_inputs = get_list_inputs_type(phase)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if phase == 'train':

        batch_size = [16, 32, 64, 128]
        learning_rate = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
        hyperparameters = [batch_size, learning_rate]
        hyperparameters_combinations = list(product(*hyperparameters))

        data_train_valid, word_emb_matrix, vocab, freqs_info = get_probe_task_data(['train', 'valid'], task, combinations_inputs, cuda = cuda, cut_point = cut_point)

        combinations_X_train, Y_vecs_train, invocab_train, words_train, labels_train = data_train_valid['train']['data']
        combinations_X_valid, Y_vecs_valid, invocab_valid, words_valid, labels_valid = data_train_valid['valid']['data']


        valid_scores_file = open(output_folder +  '/valid_scores_models.csv', 'w')
        valid_scores_file.write('inputs\tbatch_size\tlr\tvalid_loss\n')

        _, output_size = word_emb_matrix.shape

        for combination in combinations_inputs: # Training models for different inputs
            X_train, input_size = combinations_X_train[combination]
            X_valid, _ = combinations_X_valid[combination]
            output_folder_tmp = output_folder + '/' + combination + '/'
            if not os.path.isdir(output_folder_tmp):
                os.makedirs(output_folder_tmp)
            n_settings = len(hyperparameters_combinations)

            for s in range(n_settings): # Training models with different hyperparameter settings
                setting = hyperparameters_combinations[s]
                batch_size, learning_rate = setting
                model_name = '_'.join(['batch' + str(batch_size), 'lr' + str(learning_rate)])

                model = NonLinear_Transformation(input_size, output_size)
                if cuda:
                    model.cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.CosineEmbeddingLoss()

                print('Training with input ', combination, ' and hyperparameters ', model_name, '... ')

                final_valid_loss = train_model(model, setting, task, output_folder_tmp, X_train, Y_vecs_train, labels_train, words_train,
                                           X_valid, Y_vecs_valid, optimizer, criterion, word_emb_matrix, freqs_info, vocab,
                                            cuda=cuda, epochs = epochs)

                valid_scores_file.write('\t'.join([combination, str(batch_size), str(learning_rate), str(final_valid_loss)]) + '\n')

                if s != 0:
                    if final_valid_loss > current_best_loss:
                        os.remove(output_folder_tmp + model_name + '.pt')
                    elif final_valid_loss < current_best_loss:
                        for best in current_bests:
                            os.remove(output_folder_tmp + best + '.pt')
                        current_bests = [model_name]
                        current_best_loss = final_valid_loss
                    else:
                        current_bests.append(model_name)
                else:
                    current_bests = [model_name]
                    current_best_loss = final_valid_loss

        valid_scores_file.close()

    # UP TO HERE

    elif phase == 'test':
        data_test, word_emb_matrix, vocab, freqs_info = get_probe_task_data(['test'], task, combinations_inputs, cuda=cuda, cut_point=cut_point)

        combinations_X_test, Y_vecs_test, invocab_test, words_test, _ = data_test['test']['data']

        test_dataset = data_test['test']['dataframe']

        test_accuracies_file = open(output_folder + '/test_scores_models.csv', 'w')
        test_accuracies_file.write( 'inputs\tbatch_size\tlr\ttest_loss\ttest_cosine_mean\ttest_cosine_std\n')

        _, output_size = word_emb_matrix.shape

        combinations_inputs_settings = get_best_settings(combinations_inputs, output_folder)
        for combination in combinations_inputs_settings:
            if not (combination == 'wordemb' or combination == 'avg_context'):
                setting = combinations_inputs_settings[combination]
                batch_size, learning_rate = setting
                model_name = '_'.join(['batch' + str(batch_size), 'lr' + str(learning_rate)])
                print('Testing with input ', combination, ' and hyperparameters ', model_name, '... ')
                output_folder_tmp = output_folder + '/' + combination + '/'
                model = output_folder_tmp + model_name + '.pt'
                model = torch.load(model)
                if cuda:
                    model.cuda()
                unsupervised = False
            else:
                setting = ('-', '-')
                output_folder_tmp = output_folder + '/' + combination + '_unsupervised/'
                model = None
                print('Testing with input ', combination, ' unsupervised ... ')
                if not os.path.isdir(output_folder_tmp):
                    os.makedirs(output_folder_tmp)
                unsupervised = True
            X_test, input_size = combinations_X_test[combination]
            test_model(model, setting, combination, test_dataset, X_test, Y_vecs_test, output_folder_tmp,
                       invocab_test, vocab,
                       acc_file=test_accuracies_file, cut_point=cut_point, cuda=cuda, save_info=True,
                       word_emb_matrix=word_emb_matrix, unsupervised = unsupervised)
        test_accuracies_file.close()


if __name__ == "__main__":
    main()