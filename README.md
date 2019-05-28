# _Putting words in context_: LSTM language models and lexical ambiguity

This repository contains the code relevant to the paper " _Putting words in context_: LSTM language models and lexical ambiguity " (to appear in Proc. ACL 2019 - Short paper). 

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
* Code for deriving probe tasks data, based on Lexical Substitution data: tasks word, sub and word+sub (`src/get_probe_tasks_data.py`). 

* Bidirectional language model used in the experiments reported in the paper (`language_model/model.pt`)
Model trained in PyTorch using code in https://github.com/xsway/language-models 

* Code for extracting hidden representations out of the bidirectional language model (`src/rnn_models.py`, `src/probe_tasks_utils.py`) 

* Code for training and testing different types of input representations on the probe tasks: word, sub and word+sub 
(`src/run_probe_tasks.py`)

### Instructions

#### Data

To reconstruct the train/valid/test data used for the experiments: 

* Download the CoInCo Lexical Subsitution data from https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/coinco.html (`coinco.xml`)

* Then: 

```
python src/get_probe_tasks_data.py --xml ADD_PATH_TO_XML_FILE
```

If you instead want to create a new random partition of the data:
```
python src/get_probe_tasks_data.py --xml ADD_PATH_TO_XML_FILE --original_split False
```
##### Output

* Csv file with all data `data/lexsub_data.csv.`

* Csv files with data for supervised probe tasks in `data/probe_tasks/` - e.g., `data/probe_tasks/train_data.csv.`

 #### Probe tasks
 
To train the diagnostic models:

```
python src/run_probe_tasks.py --task ADD_TASK  [--cuda] [--epochs N] [--cut_point N] 

```
* `--task`: among `word`, `sub`, `word+sub`
Optional parameters:
* `--cuda` : set to run on GPU 
* `--epochs` (int): set number of epochs (by default, it uses early stopping based on validation loss)
* `--cut_point` (int): cut the data up to n datapoints, for trial runs

To choose and test the final diagnostic models:
```
python src/run_probe_tasks.py --task ADD_TASK --phase test 

```
