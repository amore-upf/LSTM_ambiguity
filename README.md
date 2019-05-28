# Putting words in context: LSTM language models and lexical ambiguity

This repository contains the code and language model used for the experiments reported in " _Putting words in context_: LSTM language models and lexical ambiguity " (to appear in Proc. ACL 2019 - Short paper). 

##### Citation

```
@inproceedings{aina-gulordava-boleda:2019:CL,
    title     = {Putting words in context: LSTM language models and lexical ambiguity},
    author    = {Aina, Laura and Gulordava, Kristina and Boleda, Gemma},
    booktitle = {Proceedings of ACL 2019 57th Annual Meeting of the Association for Computational Linguistics},
    year      = {2019},
    address   = {Florence, Italy},
    publisher = {Association for Computational Linguistics},
}
```

### Contents of this repository
* Code for deriving probe tasks data, based on Lexical Substitution data: tasks WORD, SUB, WORD+SUB (`src/get_probe_tasks_data.py`). 

* Bidirectional language model used in the experiments reported in the paper (`language_model/model.pt`)
Model trained in PyTorch using code in https://github.com/xsway/language-models 

* Code for extracting hidden representations out of the bidirectional language model (`src/rnn_models.py`, `src/probe_tasks_utils.py`) 

* Code for training and testing different types of input representations on the probe tasks: word, sub and word+sub 
(`src/run_probe_tasks.py`)

### Instructions

#### Data

To construct train/valid/test data: 

* Download the CoInCo Lexical Subsitution data from:
https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/coinco.html (File: `coinco.xml`)

* Then: 

```
python src/get_probe_tasks_data.py [-h] [--xml_file XML_FILE] [--new_split]
```

* `--xml_file`: path to xml file with Lexical Substitution data (`coinco.xml`)
* `--new_split`: to create a new random partition of the data; by default, it uses the original random split

##### Output

* Csv file with all data `data/lexsub_data.csv.`

* Csv files with data for supervised probe tasks in `data/probe_tasks/` - e.g., `data/probe_tasks/train_data.csv.`

 #### Probe tasks
 
To train or test the diagnostic models:

```
python src/run_probe_tasks.py [-h] [--phase PHASE] [--task TASK]  [--cuda] [--n_epochs N_EPOCHS] [--n_datapoints N_DATAPOINTS] 
```
* `--phase`: `train` or `test`; by default, `train`
* `--task`: among `word`, `sub`, `word+sub`; default: `sub`
* `--cuda` : set to run on GPU ; default : `False`
* `--n_epochs` (int): set number of epochs; default: `None` -->  It uses early stopping based on validation loss)
* `--n_datapoints` (int): cut the data up to n datapoin for trial runs; default: `all`

Different input types are considered: at training time, current and predictive hidden states at different layers (1-3) (e.g., `currenthidden1` == current hidden state layer 1); at test times, the unsupervised baselines are added (embedding of target word, and average embedding of 10 context words, including the target word itself). See paper for details.

During training phase, for each input type, diagnostic models with different hyperparameters combinations are trained (batch size, learning rate), and evaluated on validation data. For each model, train and validation loss at each epoch is saved into a log file: e.g., for SUB task and current hidden state at layer 1, batch size 16 and learning rate 0.0001: `probe_tasks_output/sub/currenthidden1/batch16_lr0.0001.log`. At the end of the hyperparameter search, only the model with the best validation loss is saved (e.g., `probe_tasks_output/sub/currenthidden1/batch16_lr0.0001.pt`). The final validation loss for each model and input type is saved into a csv file (e.g., for SUB task, `probe_tasks_output/sub/valid_scores_models.csv`).

During testing phase, for each input type, the model with the smallest validation loss is loaded and evaluated on test data. The test scores for each model (one per input type) is saved into a csv file (e.g., for SUB task, `probe_tasks_output/sub/test_scores_models.csv`).  For each input type, the information yielded for each datapoint (cosine score and neighbors) is saved into a csv file (e.g., for SUB task and current hidden state at layer 1, `probe_tasks_output/sub/currenthidden1/labels_test.csv`). 
