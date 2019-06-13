import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message='source.*has changed')
import pandas as pd

from lxml import etree
import string
import argparse
import random
from nltk.tokenize import TreebankWordTokenizer


def process_sentence(sentence):
    """
    Tokenize sentence
    :param sentence: sentence as str
    :return: tokenized sentence as str
    """
    sentence = tokenizer.tokenize(sentence)
    return ' '.join(sentence)

def parse_element(element):
    """
    Parse element of XML tree, whereas element = context with multiple words annotated
    :param element: element of a context
    :return: data as list of lists
    """
    records = [] # Store data for each word annotated for the context
    precontext, targetsentence, postcontext, tokens = iter(element) # Context: previous sentence, sentence, following sentence
    context_id = element.attrib['MASCfile'] + element.attrib['MASCsentID'] #Context identifier: document id + sentence identifier
    precontext = process_sentence(precontext.text.replace('\n', ''))
    postcontext = process_sentence(postcontext.text.replace('\n', ''))
    targetsentence = process_sentence(targetsentence.text.replace('\n', ''))
    # Necessary to track which occurrence of the word (in case there are many)
    data_sentence = {i:[] for i in set(targetsentence.split())}
    counts_sentence = {i:0 for i in set(targetsentence.split())}
    for i in range(len(targetsentence.split())):
        word = targetsentence.split()[i]
        data_sentence[word].append(i) # counts occurrences of words in context
        counts_sentence[word] += 1
    # Iterate through tokens in sentence
    for token in iter(tokens):
        token_id = token.attrib['id']
        if token_id != 'XXX':
            token_pos = token.attrib['posMASC']
            token_lemma = token.attrib['lemma']
            token_form = token.attrib['wordform']
            token_count = targetsentence.split().count(token_form)
            if token_count > 0 and data_sentence[token_form] != []: #not part of a compound
                token_index = data_sentence[token_form].pop(0)
                sent_for_token = targetsentence.split()
                sent_for_token[token_index] = '<target> '+ token_form + ' </target>'
                sent_for_token = ' '.join(sent_for_token)
                if precontext == targetsentence:
                    precontext = ''
                if postcontext == targetsentence:
                    postcontext = ''
                context = ' '.join([precontext, sent_for_token, postcontext])
                context = context.split()
                index = context.index('<target>')
                del context[index]
                del context[index + 1]
                context = ' '.join(context)
                substitutions_freq = {}
                for subst in iter(token[0]):
                    lemma = subst.attrib['lemma']
                    if len(lemma.split()) == 1: #NO MWE EXPRESSIONS
                        substitutions_freq[lemma] = int(subst.attrib['freq'])
                if substitutions_freq != {}:
                    substitutions = sorted(substitutions_freq.keys(), reverse = True, key = lambda x: substitutions_freq[x])
                    chart_freq = sorted(substitutions_freq.values(), reverse= True)
                    if token_form.lower() == token_lemma: #form == lemma
                        record = [token_form, token_pos, index, context, substitutions, chart_freq, context_id]
                        records.append(record)
            else:
                pass
    return records

def xml_to_df(xml_file):
    """
    Process Lexical substitution data from XML format to Pandas dataframe
    :param xml_data:
    :return: Pandas dataframe
    """
    print('Processing XML data ... ')
    parser = etree.XMLParser(recover=True)
    data_xml = etree.parse(xml_file, parser=parser)
    root = data_xml.getroot()
    structure_data = []
    for child in iter(root):
        structure_data += parse_element(child) # Collect data for each context
    print('Turning data into Pandas dataframe ... ')
    # Create dataframe from records (list of lists)
    df = pd.DataFrame.from_records(structure_data, columns = ['word', 'pos', 'idx_in_context','context', 'substitutes', 'frequencies', 'context_id'])
    df.index = range(1, len(df) + 1)
    return df

def split_data(data, new_split = False):
    """
    Split data into train, validation and test data, with no overallaping contexts
    :param data: Pandas dataframe with all data
    :param new_split: create new random split; by default, it uses original ACL2019 paper splits
    :return: train/valid/test Pandas dataframes
    """
    if not new_split:
        'Reconstructing original dataset  ...'
        splits = {}
        for phase in ['train', 'valid', 'test']:
            with open('data/probe_tasks/' + phase + '_context_ids.txt', 'r') as ids_file:
                splits[phase] = eval(ids_file.read())
    else:
        print('New partition of the data ...')
        context_ids = list(set(data.context_id))
        total = len(context_ids)
        train = int(round(total/100 * 70, 0))
        valid = int(round(total/100 * 10, 0))
        random.shuffle(context_ids)
        splits = {}
        splits['train'] = context_ids[:train]
        splits['valid'] = context_ids[train:train+valid]
        splits['test'] = context_ids[train+valid:total]
    train_df = data[data.context_id.isin(splits['train'])].sample(frac=1).reset_index(drop=True)
    valid_df = data[data.context_id.isin(splits['valid'])].sample(frac=1).reset_index(drop=True)
    test_df = data[data.context_id.isin(splits['test'])].sample(frac=1).reset_index(drop=True)
    return train_df, valid_df, test_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--xml_file', action="store", default = None, type = str,
                        help = 'Path to xml file with Lexical Subsitution data')  #path to XML file with LexSub data
    parser.add_argument('--new_split', action="store_true", default=False,
                        help = 'If true, a new random partition of the data is created')
    args = parser.parse_args()

    tokenizer = TreebankWordTokenizer() # Used to tokenize context sequences
    data_file = args.xml_file

    if data_file == None:
        print('No xml file provided')
        quit()

    new_split = args.new_split
    data = xml_to_df(data_file) # Parses XML and store info in Pandas dataframe
    print('Saving data into csv file ...')
    data.to_csv('data/lexsub_data.csv', sep = '\t')
    print('Splitting into train/valid/test ...')
    train_df, valid_df, test_df = split_data(data, new_split = new_split) #Create train/valid/test split
    print('Saving split data into csv files ...')
    train_df.to_csv('data/probe_tasks/train_data.csv')
    valid_df.to_csv('data/probe_tasks/valid_data.csv')
    test_df.to_csv('data/probe_tasks/test_data.csv')
