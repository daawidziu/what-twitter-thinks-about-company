import re
import string
from nltk import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class TweetNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def remove_reudant(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
        tweet = re.sub(r'@\w+|#', '', tweet)
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        return tweet

    def tokenize(self, tweet):
        tweet_tokens = word_tokenize(tweet)
        return tweet_tokens

    def remove_stopwords(self, tweet_tokens):
        filtered_words = [w for w in tweet_tokens if not w in self.stopwords]
        return filtered_words

    def lemmatize(self, tweet_tokens):
        lemma_words = [self.lemmatizer.lemmatize(word) for word in tweet_tokens]
        return lemma_words

    def normalize(self, tweet):
        tweet = self.remove_reudant(tweet.lower())
        tweet_tokens = self.tokenize(tweet)
        filtered_words = self.remove_stopwords(tweet_tokens)
        lemma_words = self.lemmatize(filtered_words)
        return ' '.join(lemma_words)

    def fit(self, X, y=None):
        return self

    def transform(self, tweets):
        return [self.normalize(tweet) for tweet in tweets]


class SentimentAnalyzer(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def predict(self, tweets):
        return [self.sia.polarity_scores(tweet) for tweet in tweets]

    def fit_predict(self, tweets):
        return [self.sia.polarity_scores(tweet) for tweet in tweets]
