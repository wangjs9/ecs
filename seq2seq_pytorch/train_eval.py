import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time, os, re, pickle, math
from itertools import groupby
from utils import get_time_dif 
from pytorch_pretrained_bert.optimization import BertAdam
from keras.utils import plot_model

def train(config, model, train_iter, dev_iter):
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    emotion_dict, _ = pickle.load(open(config.emotion_path, 'rb'))
    emotion = [emotion_dict[i] for i in range(len(emotion_dict))]
    emotion = torch.LongTensor(emotion).to(config.device)
    grammar = open(config.stopwords_path, 'r', encoding='UTF-8').readlines()
    grammar = [vocab.get(x.strip()) for x in grammar if x.strip() in vocab.keys()]
    grammar = torch.LongTensor(grammar).to(config.device)
    start_time = time.time()
    
    total_batch = 0  
    no_improve = 0
    last_no_improve = 0
    dev_best_loss = float('inf')
    dev_last_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    if os.path.exists(config.save_path):
        checkpoint = torch.load(config.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dev_best_loss = checkpoint['dev_best_loss']
    
    dev_loss = evaluate(config, model, emotion, grammar, dev_iter)
    print(dev_loss)

    model.train()
    
    criterion = MaskNLLLoss()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (Queries, Responses) in enumerate(train_iter):
            outputs = model(Queries, Responses, emotion, grammar)[0]
            model.zero_grad()
            loss = criterion(outputs, Responses[0], Responses[1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if total_batch % 100 == 0:
                train_loss = loss.item()
                dev_loss = evaluate(config, model, emotion, grammar, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dev_best_loss': dev_best_loss,
                    }
                    if not os.path.exists(config.save_dic):
                        os.makedirs(config.save_dic)
                    torch.save(state, config.save_dic + config.model_name + '.pth')
                    # torch.save(state, config.save_dic + config.model_name +'_' + str(total_batch) + '.pth')
                    improve = '*'
                    no_improve = 0
                    del state
                else:
                    no_improve += 1
                    improve = ''

                if dev_last_loss > dev_loss:
                    last_no_improve = 0
                    if improve == '':
                        improve = '-'
                elif last_no_improve % 5 == 0:
                    last_no_improve += 1
                    scheduler.step()
                else:
                    last_no_improve += 1
                    
                dev_last_loss = dev_loss
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4f},  Val Loss: {2:>5.4f}, Perplexity: {5:5.4f}, Time: {3} {4}'
                print(msg.format(total_batch, train_loss, dev_loss, time_dif, improve, math.exp(dev_loss)))
                model.train()

            total_batch += 1
            if no_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                break

class MaskNLLLoss(nn.Module):
    def __init__(self):
        super(MaskNLLLoss, self).__init__()

    def forward(self, inp, tgt, tgt_length):
        mask = torch.arange(tgt.size(1), device=inp.device)[None, :] < tgt_length[:, None]
        mask = mask.float()
        inp = inp.contiguous().view(-1, inp.size(2))
        tgt = tgt.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -inp.gather(1, tgt) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def test(config, model, test_iter):
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    emotion_dict, _ = pickle.load(open(config.emotion_path, 'rb'))
    emotion = [emotion_dict[i] for i in range(len(emotion_dict))]
    emotion = torch.LongTensor(emotion).to(config.device)
    grammar = open(config.stopwords_path, 'r', encoding='UTF-8').readlines()
    grammar = [vocab.get(x.strip()) for x in grammar if x.strip() in vocab.keys()]
    grammar = torch.LongTensor(grammar).to(config.device)

    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    start_time = time.time()
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    re_vocab = {token_id: token for token, token_id in vocab.items()}    
    x = []
    # real_y = []
    pred_y = []
    with torch.no_grad():
        for i, (Queries, Responses) in enumerate(test_iter):
            output = model.response(Queries, Responses[2], emotion, grammar)
            x = x + [sentence(q, re_vocab) for q in Queries[0].cpu().tolist()]
            # real_y = real_y + [sentence(r, re_vocab) for r in Responses[0].cpu().tolist()]
            pred_y = pred_y + [sentence(r, re_vocab) for r in output]

    data = []
    for i in range(len(x)):
        data.append('Pair {}'.format(i+1))
        data.append('Query: {}'.format(x[i]))
        # data.append('Original Response: {}'.format(real_y[i]))
        data.append('Generated Response: {}'.format(pred_y[i]))
        data.append(' ')
        
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(config.save_dic, 'results_token.txt'), sep='\t', encoding='utf8', header=False, index=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

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

def evaluate(config, model, emotion, grammar, data_iter, test=False):
    model.eval()
    loss_total = 0.
    len_total = 0
    criterion = MaskNLLLoss()
    with torch.no_grad():
        for i, (Queries, Responses) in enumerate(data_iter):
            outputs = model(Queries, Responses, emotion, grammar)[0]
            length = Queries[0].size(0)
            loss_total += length * criterion(outputs, Responses[0], Responses[1]).item()                
            len_total += length

    return loss_total / len_total #len(data_iter)


