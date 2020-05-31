from utils import build_dataset, build_iterator, get_time_dif
from importlib import import_module
import pickle, os
import pandas as pd
from itertools import groupby

def main():
    dataset = 'C:/Users/USER/Documents/Capstone_Project/datalogs'
    x = import_module('models.{}'.format('RNN'))
    config = x.Config(dataset)
    train_data, dev_data, test_data = build_dataset(config, True, False)
    dev_iter = build_iterator(dev_data, config, do_dev=True)

    vocab = pickle.load(open(config.vocab_path, 'rb'))
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    re_vocab = {token_id: token for token, token_id in vocab.items()}    

    x = []
    real_y = []
    for i, (Queries, Responses) in enumerate(dev_iter):
        x = x + [sentence(q, re_vocab) for q in Queries[0].cpu().tolist()]
        real_y = real_y + [sentence(r, re_vocab) for r in Responses[0].cpu().tolist()]

    data = []
    for i in range(len(x)):
        data.append('Pair {}'.format(i+1))
        data.append('Query: {}'.format(x[i]))
        data.append('Original Response: {}'.format(real_y[i]))
        data.append(' ')

    data = pd.DataFrame(data)
    data.to_csv(os.path.join('results_token.txt'), sep='\t', encoding='utf8', header=False, index=False)
    
    

def sentence(WordList, re_vocab):
    string = []
    for idx in WordList:
        if re_vocab[idx] == '[EOS]':
            break
        if re_vocab[idx] == '[PAD]':
            continue
        string.append(re_vocab[idx])
    string = [x[0] for x in groupby(string)]
    string = postdecorate(string)
    string = ' '.join(string)
    return string.strip()

def postdecorate(string):
    indexes = [i for i, x in enumerate(string) if x in ['，','。']]
    delete = []
    if len(indexes) > 0:
        if string[-1] not in ['。', '！']:
            string.append('end')
        if indexes[-1] != len(string)-1: 
            indexes.append(len(string)-1)
        start = 0
        for idx in range(len(indexes)-1):
            if string[start:indexes[idx]] == string[indexes[idx]+1:indexes[idx+1]]:
                delete += [i for i in range(start, indexes[idx]+1)]
            start = indexes[idx]+1
        if string[-1] == 'end':
            string = string[:-1]
        newString = [string[i] for i in range(len(string)) if i not in delete]
        
        return newString
    return string

main()