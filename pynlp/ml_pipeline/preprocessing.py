from nltk import TweetTokenizer
import spacy
from sklearn.base import TransformerMixin

from ml_pipeline.utils import hate_lexicon


class Preprocessor(TransformerMixin):
    """preprocesses the data with NLTK and Spacy (lemmatizer)"""
    def __init__(self, tokenize, normalize_tweet, lowercase, lemmatize, lexicon={}):
        tt_args = {}
        tt_args['reduce_len'] = normalize_tweet
        tt_args['strip_handles'] = normalize_tweet
        tt_args['preserve_case'] = not lowercase
        self.processors = []
        self.tokens_from_lexicon = 0
        if tokenize:
            self.processors.append(tokenize_with(tt_args))
        if lemmatize:
            self.processors.append(lemmatize_with_spacy)
        if lexicon:
            self.processors.append(self.identify_in_lexicon(lexicon))

    def transform(self, data):
        for p in self.processors:
            data = p(data)
        return data

    def fit_transform(self, data, y=None):
        return self.transform(data)

    def identify_in_lexicon(self, lexicon):
        """replaces words in a tweet by a label from a lexicon (pos/neg); defaults to 'NEUTRAL'"""

        def apply_lexicon(data):
            self.tokens_from_lexicon = 0
            processed = []
            for tw in data:
                processed_tweet = []
                for token in tw.split():
                    lex_id = 'neutral'
                    if token in lexicon:
                        lex_id = lexicon[token]['label']
                        self.tokens_from_lexicon += 1
                    processed_tweet.append(lex_id.upper())
                processed.append(' '.join(t for t in processed_tweet))
            return processed

        return apply_lexicon


def tokenize_with(kwargs):
    tokenizer = TweetTokenizer(**kwargs)

    def tweet_tokenizer(data):
        return [' '.join(tokenizer.tokenize(tweet)) for tweet in data]
    return tweet_tokenizer


def lemmatize_with_spacy(data):
    nlp = spacy.load("en_core_web_sm")

    def apply_spacy(tw):
        return ' '.join([token.lemma_ for token in nlp(tw)])
    return [apply_spacy(tweet) for tweet in data]


# -------------- standard preprocessor --------------------------------

def std_prep():
    return Preprocessor(tokenize=True, normalize_tweet=True, lowercase=True, lemmatize=False)


def lex_prep():
    return Preprocessor(tokenize=True, normalize_tweet=True, lowercase=True, lemmatize=False, lexicon=hate_lexicon())
