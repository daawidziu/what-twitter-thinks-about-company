import re
import string
import numpy as np
from nltk import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class TweetNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn custom transformer to clean and normalize tweets"""

    def __init__(self, language='english'):
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def remove_reudant(self, tweet):
        """Remove urls, hashtags and random punctuation from tweets"""
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
        tweet = re.sub(r'@\w+|#', '', tweet)
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        return tweet

    def tokenize(self, tweet):
        """Tokenize tweet"""
        tweet_tokens = word_tokenize(tweet)
        return tweet_tokens

    def remove_stopwords(self, tweet_tokens):
        """Remove stopwords from "tweet tokens"""
        filtered_words = [w for w in tweet_tokens if w not in self.stopwords]
        return filtered_words

    def lemmatize(self, tweet_tokens):
        """Lemmatize tweet tokens using WordNetLemmatizer"""
        lemma_words = [self.lemmatizer.lemmatize(word) for word in tweet_tokens]
        return lemma_words

    def normalize(self, tweet):
        """Remove reduants, stopwords and lemmatize tweets"""
        tweet = self.remove_reudant(tweet.lower())
        tweet_tokens = self.tokenize(tweet)
        filtered_words = self.remove_stopwords(tweet_tokens)
        lemma_words = self.lemmatize(filtered_words)
        return ' '.join(lemma_words)

    def fit(self, X, y=None):
        """Return self"""
        return self

    def transform(self, tweets):
        """Return normalized tweets"""
        return [self.normalize(tweet) for tweet in tweets]


class SentimentAnalyzer(BaseEstimator, ClassifierMixin):
    """Sklearn wrapper for nltk Vader SentimentIntensityAnalyzer"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        """Return self"""
        return self

    def predict(self, tweets):
        """Return polarity scores for each tweet"""
        return [self.sia.polarity_scores(tweet) for tweet in tweets]

    def fit_predict(self, tweets):
        """Return polarity scores for each tweet"""
        return [self.sia.polarity_scores(tweet) for tweet in tweets]


class TopWords(BaseEstimator):
    """Sklearn class to be used in pipeline to get top n words for each topic(LDA).
    Assumes LDA and vectorizer(CountVectorizer/TFidVectorizer) has been already fitted"""

    def __init__(self, vectorizer, lda, n=10):
        self.vocab = vectorizer.get_feature_names()
        self.lda = lda
        self.n = n
        self.topic_words = {}

    def fit_predict(self, X=None, y=None):
        """Return top n words for each topic"""
        for topic, comp in enumerate(self.lda.components_):
            word_idx = np.argsort(comp)[::-1][:self.n]
            self.topic_words[topic] = {self.vocab[i]: float(comp[i]) for i in word_idx}
        return self.topic_words
