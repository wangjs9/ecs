import pickle
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture
import numpy as np

class WordCluster(object):
    def __init__(self, num_class=500, vector_size=300, window=10, min_count=1):
        self.num_class = num_class
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def train(self, text, save_path, original=True):
        if original == False:
            self.load(save_path)
        try:
            self.Vector = Word2Vec.load(save_path+'wv')
        except FileNotFoundError:
            self.Vector = Word2Vec(text, size=self.vector_size, window=self.window, sg=1) # skip-gram
            self.Vector.save(save_path+'.wv')

        self.keys = self.Vector.wv.vocab.keys()
        wordvector = []
        for key in self.keys:
            wordvector.append(self.Vector[key])

        try:
            self.GMM = pickle.load(open(save_path, 'rb'))
            labels = self.GMM.predict(wordvector)
            
        except FileNotFoundError:
            self.GMM = GaussianMixture(self.num_class, n_init=20)
            labels = self.GMM.fit_predict(wordvector)
            pickle.dump(self.GMM, open(save_path, 'wb'))

        try:
            self.classes = pickle.load(open(save_path+'.cls', 'rb'))
        except FileNotFoundError:
            self.classes = {label:[] for label in labels}
            for word, label in zip(self.keys, labels):
                self.classes[label].append(word)
            pickle.dump(self.classes, open(save_path+'.cls', 'wb'))

        self.topic = dict()
        for word, label in zip(self.keys, labels):
            target = self.classes[label].copy()
            if len(target) <= 10:
                result = target
            else:
                result = []
                for i in range(10):
                    w = self.Vector.wv.most_similar_to_given(word, target)
                    result.append(w)
                    target.remove(w)
            self.topic[word] = result
        
        pickle.dump(self.topic, open(save_path+'.classes', 'wb'))
        
    def predict(self, sentence):
        result = []
        for word in sentence:
            if word not in self.keys:
                continue
            else:
                result += self.topic[word]
        return result

    def load(self, save_path):
        self.classes = pickle.load(open(save_path+'.cls', 'rb'))
        self.Vector = Word2Vec.load(save_path+'.wv')
        self.GMM = pickle.load(open(save_path, 'rb'))
        self.topic = pickle.load(open(save_path+'.classes', 'rb'))
        self.keys = self.Vector.wv.vocab.keys()
    