import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message='source.*has changed')

import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
from probetasks_utils import *
from itertools import product, permutations

def get_data(task, inputs, concat_sum, model_type):
    data_word_file = 'data/'+task + '/test_'+ model_type +'_'+inputs +'_'+concat_sum +'.pkl'
    combinations_X, _, vecs, invocab, _ = pickle.load(open(data_word_file, 'rb'))
    avg_context = combinations_X[str(['avg_context'])]
    return vecs, invocab, avg_context

inputs = 'all'
concat_sum = 'concat'
model_type = 'none'

language_model = 'models1/bidir_'+ model_type +'_lr0.0005_batchsize32_dropout0.2_embsize300_pretrainedNO_fixedNO.pt'
_, language_model, vocab, _ = load_languagemodel(language_model, cuda=True)
# word_emb_matrix = language_model.encoder.embedding.weight.data

vecs_substitution, invocab, avg_context_vecs  =  get_data ('substitution', inputs, concat_sum, model_type)
vecs_wordsubstitution, _, _ =  get_data('word+substitution', inputs, concat_sum, model_type)
vecs_word, _, _ =  get_data ('word', inputs, concat_sum, model_type)


sims_word_substitution = []
sims_word_wordsubstitution = []
sims_avg_word = []
sims_avg_substitution = []
sims_avg_wordsubstitution = []

cosine = torch.nn.CosineEmbeddingLoss()

for i in range(vecs_word.shape[0]):
    word = vecs_word[i].unsqueeze(0)
    substitution = vecs_substitution[i].unsqueeze(0)
    wordsubstitution = vecs_wordsubstitution[i].unsqueeze(0)
    avg_context = avg_context_vecs[0][i].unsqueeze(0)
    sim = 1 - float(cosine(word, substitution, torch.ones(1).cuda()))
    sims_word_substitution.append(sim)
    sim = 1 - float(cosine(word, wordsubstitution, torch.ones(1).cuda()))
    sims_word_wordsubstitution.append(sim)

    sim = 1 - float(cosine(word, avg_context, torch.ones(1).cuda()))
    sims_avg_word.append(sim)
    sim = 1 - float(cosine(avg_context, substitution, torch.ones(1).cuda()))
    sims_avg_substitution.append(sim)
    sim = 1 - float(cosine(avg_context, wordsubstitution, torch.ones(1).cuda()))
    sims_avg_wordsubstitution.append(sim)


print('Word embedding -substitution', np.mean(sims_word_substitution), np.std(sims_word_substitution))
print('Word embedding -word+substitution', np.mean(sims_word_wordsubstitution), np.std(sims_word_wordsubstitution))

print('Avg context embedding -word', np.mean(sims_avg_word), np.std(sims_avg_word))
print('Avg context embedding -substitution', np.mean(sims_avg_substitution), np.std(sims_avg_substitution))
print('Avg context embedding -word+substitution', np.mean(sims_avg_wordsubstitution), np.std(sims_avg_wordsubstitution))

pickle.dump((sims_word_substitution, sims_word_wordsubstitution), open('similarities-targets.pkl','wb'))

# language_model = 'models1/bidir_'+ model_type +'_lr0.0005_batchsize32_dropout0.2_embsize300_pretrainedNO_fixedNO.pt'
# _, language_model, vocab, _ = load_languagemodel(language_model, cuda=False)
# word_emb_matrix = language_model.encoder.embedding.weight.data