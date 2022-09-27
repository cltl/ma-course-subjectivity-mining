from sklearn.base import TransformerMixin
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path
from tqdm import tqdm

from sklearn.pipeline import FeatureUnion


class Text2Embedding(TransformerMixin):

    def __init__(self, embed_source):
        print(f"Using embed_source: {embed_source}")
        self.embed_source = embed_source

    def fit_transform(self, X, parameters=[]):
        print('transforming data using customized transformer')
        model = None
        if self.embed_source == 'glove':
            path = f'/Users/ryansaeta/Desktop/Vrije Universiteit/Y2/S1P1/Subjectivity Mining/ma-course-subjectivity-mining/pynlp/data/glove.twitter.27B/glove.twitter.27B.100d.txt'
            w2vfile = f'/Users/ryansaeta/Desktop/Vrije Universiteit/Y2/S1P1/Subjectivity Mining/ma-course-subjectivity-mining/pynlp/data/glove.twitter.27B/glove.twitter.27B.100d.vec'
            if not Path(w2vfile).is_file():
                glove2word2vec(path, w2vfile)
            print('loading model from file')
            model = KeyedVectors.load_word2vec_format(w2vfile, binary=False)
            print('finished loading model from file')
        else:
            path = f'/Users/ryansaeta/Desktop/Vrije Universiteit/Y2/S1P1/Subjectivity Mining/ma-course-subjectivity-mining/pynlp/data/wiki-news-300d-1M.vec'
            model = KeyedVectors.load_word2vec_format(path, binary=False)
        n_d = len(model['the'])
        data = []
        for tokenized_tweet in tqdm(X):
            tokens = tokenized_tweet.split(' ')
            tweet_matrix = np.array([model[t] for t in tokens if t in list(model.index_to_key)])
            if len(tweet_matrix) == 0:
                data.append(np.zeros(n_d))
            else:
                data.append(np.mean(tweet_matrix, axis=0))
        return np.array(data)

    def transform(self, X):
        return self.fit_transform(X)


# --------------- standard formatters ----------------------

def count_vectorizer(kwargs={}):
    return CountVectorizer(**kwargs)


def tfidf_vectorizer(kwargs={}):
    return TfidfVectorizer(**kwargs)


def text2embeddings(embed_source='glove'):
    return Text2Embedding(embed_source)
