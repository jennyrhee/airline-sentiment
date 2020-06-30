import pandas as pd
import numpy as np

import re
import pickle

import emoji
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

import tweepy

from dotenv import load_dotenv
import os

tfidf = pickle.load(open('models/tfidf.pickle', 'rb'))
model = pickle.load(open('models/model.pickle', 'rb'))
load_dotenv()


def connect_twitter():
    API_KEY = os.environ.get('API_KEY')
    API_SECRET_KEY = os.environ.get('API_SECRET_KEY')
    ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
    ACCESS_TOKEN_SECRET = os.environ.get('ACCESS_TOKEN_SECRET')
    auth = tweepy.OAuthHandler(API_KEY,
                               API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN,
                          ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    return api


def get_tweets(airline):
    api = connect_twitter()
    results = api.search(airline, lang='en', count=100, exclude='retweets')
    all_tweets = []
    for tweet in results:
        all_tweets.append({'created_at': tweet.created_at,
                           'user': tweet.user.screen_name,
                           'tweet_id': tweet.id,
                           'text': tweet.text})
    return pd.DataFrame.from_dict(all_tweets)


def clean_tweets(tweet):
    no_mentions = re.sub("@([a-zA-Z0-9]{1,15})", '', tweet)
    lower = no_mentions.lower()
    no_urls = re.sub(r'http\S+', '', lower)
    no_num = re.sub(r'\d+', '', no_urls)
    emoji_to_word = emoji.demojize(no_num).replace(':', ' ').replace('  ', ' ')

    emoticon_pattern = '[:;]{1}-?[dDpPsS)(]+'
    emoticons = re.findall(emoticon_pattern, emoji_to_word)
    all_words = []
    for word in no_num.split():
        if word in emoticons:
            all_words.append(word)
        else:
            word = re.sub('[^A-Za-z0-9]+', '', word)
            word = re.sub('…', ' ', word)
            all_words.append(word)
    no_punc = ' '.join(all_words)

    no_whitespace = no_punc.strip()

    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(no_whitespace)
    no_stop = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(tweet) for tweet in no_stop]

    return ' '.join(stems)


def get_features(df):
    df['length'] = df.text.apply(lambda tweet: len(tweet))
    df['capitals'] = df.text.apply(lambda tweet: sum(1 for letter in tweet
                                                     if letter.isupper()))
    df['cap_length_ratio'] = df.capitals / df.length
    df['n_words'] = df.text.apply(lambda tweet: len(tweet.split()))
    df['n_happy'] = df.text.apply(lambda tweet: sum(tweet.count(w)
                                                    for w
                                                    in [':-)', ':)', ';-)',
                                                        ';)', ':-D', ':D']))
    df['n_sad'] = df.text.apply(lambda tweet: sum(tweet.count(w)
                                                  for w
                                                  in (':-<', ':<', ':-(',
                                                      ':(', ';-(', ';(')))
    df['n_exclamations'] = df.text.apply(lambda tweet: tweet.count('!'))
    df['n_questions'] = df.text.apply(lambda tweet: tweet.count('?'))

    return df


def preprocess(tweets_df):
    tweets_df['cleaned_tweet'] = tweets_df.text.apply(clean_tweets)
    tweets_df = get_features(tweets_df)

    return tweets_df


def predict(tweets_df):
    num_features = tweets_df.drop(['cleaned_tweet', 'tweet_id', 'text',
                                   'user', 'created_at'], axis=1)
    text_features = tweets_df.cleaned_tweet

    tfidf_matrix = tfidf.transform(text_features)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=tfidf.get_feature_names())
    final_df = pd.concat([num_features, tfidf_df], axis=1)

    predicted_values = model.predict(final_df)
    probs = model.predict_proba(final_df)

    return predicted_values, probs


def get_probs(tweets_df, pred, probs):
    tweets_df['airline_sentiment'] =  pred

    for i, row in tweets_df.iterrows():
        if row['airline_sentiment'] == 'negative':
            tweets_df.loc[i, 'prob'] = probs[i, 0]
        elif row['airline_sentiment'] == 'neutral':
            tweets_df.loc[i, 'prob'] = probs[i, 1]
        else:
            tweets_df.loc[i, 'prob'] = probs[i, 2]

    return tweets_df


def process_predictions(tweets_df):
    preds, probs = predict(tweets_df)
    tweets_df = get_probs(tweets_df, preds, probs)

    return tweets_df


def get_most_freq_sentiment(tweets_df):
    counts = tweets_df.airline_sentiment.value_counts().reset_index()
    most_freq_sentiment = (counts[
        counts.airline_sentiment == counts.airline_sentiment.max()
        ]
        ['index'].values[0])
    return most_freq_sentiment


def get_representative_tweet(tweets_df, most_freq_sentiment):
    rep_tweet = tweets_df[
        (tweets_df.airline_sentiment == most_freq_sentiment) &
        (tweets_df.prob == tweets_df.prob.max())
    ]
    user, tweet_id = rep_tweet.user.values[0], rep_tweet.tweet_id.values[0]
    url = f'https://twitter.com/{user}/status/{tweet_id}'

    return url, rep_tweet.prob.values[0]
