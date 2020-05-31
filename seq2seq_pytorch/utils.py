import torch, time, os, pickle, random
from datetime import timedelta
import pandas as pd 
from tqdm import tqdm
from WordCluster import WordCluster

MAX_VOCAB_SIZE = 33000
min_freq = 10
PAD, BOS, EOS = '[PAD]', '[BOS]', '[EOS]'

def build_vocab(vocab_path, file_path, tokenizer):
    vocab_dict = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            words_list = lin.split('\t')[0] + ' ' + lin.split('\t')[2]
            for word in tokenizer(words_list):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        
        vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
        vocab_dict = {PAD: 0, BOS: 1, EOS:2}
        vocab_dict.update({word_count[0]: idx + 3 for idx, word_count in enumerate(vocab_list)})
    pickle.dump(vocab_dict, open(vocab_path, 'wb'))
    return vocab_dict

def build_topics(vocab, file_path, topic_path, stoplist):
    documents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if lin:
                documents.append([w for w in lin.split('\t')[0].split() + lin.split('\t')[2].split() if w not in stoplist and not w.isdigit() and w in vocab])
                # documents.append([w for w in  if w not in stoplist and not w.isdigit() and w in vocab])

    model = WordCluster()
    model.train(documents, topic_path)
    return model

def build_emotions(vocab_dict, emotion_path):
    emotion_files = 'C:/Users/USER/Documents/Capstone_Project/datalogs/data/emotions/words'
    emotion_dict = dict()
    emotion_score = dict()
    emotion_idxs = []
    for i in range(1, 6):
        emotions = pd.read_csv('{}/{}'.format(emotion_files, str(i)), sep='\t', encoding='UTF8')
        emotions.columns = ['word', 'score']
        emotions['word'] = emotions['word'].apply(lambda x: vocab_dict.get(x, vocab_dict.get(PAD)))
        emotions = emotions[emotions['word']!=vocab_dict.get(PAD)]
        emotion_dict[i] = [1 if i in list(emotions['word']) else 0 for i in range(len(vocab_dict))]
        emotion_idxs.extend(list(emotions['word']) )
        for index, line in emotions.iterrows():
            emotion_score[int(line['word'])] = line['score']
    emotion_dict[0] = [0 if i in emotion_idxs else 0 for i in range(len(vocab_dict))]
    pickle.dump((emotion_dict, emotion_score), open(emotion_path, 'wb'))
    return emotion_idxs

class InputExamples(object):
    def __init__(self, Query, Response=None):
        self.Query = Query
        self.Response = Response

class InputFeatures(object):
    def __init__(self, token_ids, seq_len, label, topics=None):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.label = int(label)
        self.topics = topics

def build_dataset(config, do_train, do_test):
    tokenizer = lambda x: x.split(' ')  # word-level
    stoplist = open(config.stopwords_path, 'r', encoding='UTF-8').readlines()
    stoplist = [x.strip() for x in stoplist]

    if not os.path.exists(config.vocab_path):
        vocab = build_vocab(config.vocab_path, config.train_path, tokenizer)
    else:
        vocab = pickle.load(open(config.vocab_path, 'rb'))

    if not os.path.exists(config.emotion_path):
        emotion_idxs = build_emotions(vocab, config.emotion_path)
        emotion_idxs = [i for i in range(len(emotion_idxs)) if emotion_idxs[i] ]
    else:
        emotion_dict, _ = pickle.load(open(config.emotion_path, 'rb'))
        emotion_idxs = [idx for idx, i in enumerate(emotion_dict[0]) if i != 0 ]

    if not os.path.exists(config.topic_path+'.classes'):
        topic_model = build_topics(vocab.keys(), config.train_path, config.topic_path, stoplist)
    else:
        topic_model = WordCluster()
        topic_model.load(config.topic_path)

    def load_dataset(path, pad_size=32, do_train=True):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue

                query, label_Q, response, label_R = lin.split('\t')
                label_Q = int(label_Q)
                label_R = int(label_R)
        
                # Query
                token_ids = []
                token = [w for w in tokenizer(query) if w in vocab.keys()]
                seq_len = len(token)
                if seq_len == 0:
                    continue

                # r_token = [w for w in tokenizer(response) if w in vocab.keys()] + [EOS]
                # r_seq_len = len(r_token)
                # if seq_len == 2:
                #     print(lin)
                #     with open('_de.txt','a',encoding='UTF8') as f:
                #         f.write(response+'\n')
                #     with open('_del.txt','a',encoding='UTF8') as f:
                #         f.write(query+'\n')
                #     continue

                for word in token:
                    token_ids.append(vocab.get(word))          

                if pad_size:
                    if len(token_ids) < pad_size:
                        token_ids += ([0] * (pad_size - len(token_ids)))
                    else:
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                if config.topic_num > 0:
                    topicslist = topic_model.predict(token) 
                    random.shuffle(topicslist)
                    topicslist = [vocab[w] for w in topicslist]
                    while len(topicslist) < config.topic_num:
                        topicslist.append(0)
                    topicslist = topicslist[:config.topic_num]
                else:
                    topicslist = None
                    
                Query = InputFeatures(token_ids=token_ids, seq_len=seq_len, label=label_Q, topics=topicslist)
                
                if do_train:
                    token_ids = []
                    token = [w for w in tokenizer(response) if w in vocab.keys()] + [EOS]
                    seq_len = len(token)
                    for word in token:
                        token_ids.append(vocab.get(word))
                    if seq_len == 0:
                        continue
                    if pad_size:
                        if len(token_ids) < pad_size:
                            token_ids += ([0] * (pad_size - len(token_ids)))
                        else:
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size

                    Response = InputFeatures(token_ids=token_ids, seq_len=seq_len, label=label_R)

                else:
                    Response = InputFeatures(token_ids=[], seq_len=0, label=label_R)
                    
                content = InputExamples(Query=Query, Response=Response)
                contents.append(content)
        if do_train:
            random.shuffle(contents)
        return contents
    
    train = None
    dev = None
    test = None
    if do_train:
        train = load_dataset(config.train_path, config.pad_size)
        dev = load_dataset(config.dev_path, config.pad_size)
    if do_test:
        test = load_dataset(config.test_path, config.pad_size, do_train=False)
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False 
        if self.n_batches == 0 or len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        Queries = [_.Query for _ in datas]
        Responses = [_.Response for _ in datas]

        q = torch.LongTensor([_.token_ids for _ in Queries]).to(self.device)
        seq_len_q = torch.LongTensor([_.seq_len for _ in Queries]).to(self.device)
        label_q = torch.LongTensor([_.label for _ in Queries]).to(self.device)
        if Queries[0].topics == None:
            topics_q = None
        else:
            topics_q = torch.LongTensor([_.topics for _ in Queries]).to(self.device)
        if Responses[0] == None:
            return (q, seq_len_q, label_q, topics_q), None

        r = torch.LongTensor([_.token_ids for _ in Responses]).to(self.device)
        seq_len_r = torch.LongTensor([_.seq_len for _ in Responses]).to(self.device)
        label_r = torch.LongTensor([_.label for _ in Responses]).to(self.device)

        return (q, seq_len_q, label_q, topics_q), (r, seq_len_r, label_r)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config, do_dev=False):
    if do_dev:
        iter = DatasetIterater(dataset, 90, config.device)
    else:
        iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """get time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

