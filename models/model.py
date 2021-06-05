import pandas as pd
import numpy as np

import re

import emoji
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

import tweepy

from dotenv import load_dotenv
import os

load_dotenv()


def connect_twitter():
    '''Connects to Twitter API. 

    Returns
    -------
    tweepy.API
    '''
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
    '''Searches for and returns recent mentions for airline.

    Parameter
    ---------
    airline (str): airline Twitter handle

    Returns
    -------
    pd.DataFrame with mentions
    '''
    api = connect_twitter()
    results = api.search(airline, lang='en', count=100, exclude='retweets')
    all_tweets = []
    for tweet in results:
        all_tweets.append({'created_at': tweet.created_at,
                           'user': tweet.user.screen_name,
                           'tweet_id': tweet.id,
                           'text': tweet.text})
    return pd.DataFrame.from_dict(all_tweets)


def tokenize_and_stem(cleaned_tweet):
    '''Tokenizes the cleaned tweet using TweetTokenizer and stems using PorterStemmer.

    Parameter
    ---------
    cleaned_tweet (str)

    Returns
    -------
    str
    '''
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(cleaned_tweet)
    no_stop = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(tweet) for tweet in no_stop]

    return  ' '.join(stems)


def clean_tweets(tweet):
    '''Cleans tweet by removing mentions, URLs, numbers, punctuation,  and whitespace,
    converting to lowercase, and converting emojis to word representations.

    Parameter
    ---------
    tweet (str)

    Returns
    -------
    no_whitespace (str): final cleaned tweet
    '''
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

    return no_whitespace


def create_features(df):
    '''Creates new features based on original tweets: character count, capital letter
    count, capital letter to character count ratio, number of words, number of happy
    and sad emoticons, number of exclamation marks, and number of question marks.

    Parameter
    ---------
    df (pd.DataFrame)

    Returns
    -------
    df (pd.DataFrame): with new features
    '''
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
    '''Prepares data for prediction. Cleans tweet, tokenizes and stem, and creates new features.

    Parameter
    ---------
    tweets_df (pd.DataFrame)

    Returns
    -------
    tweets_df (pd.DataFrame): processed DataFrame
    '''
    tweets_df['cleaned_tweet'] = tweets_df.text.apply(clean_tweets)
    tweets_df['cleaned_tweet'] = tweets_df.cleaned_tweet.apply(tokenize_and_stem)
    tweets_df = create_features(tweets_df)

    return tweets_df


def predict(tweets_df, tfidf, model):
    '''Uses pickled models to predict sentiment.

    Parameter
    ---------
    tweets_df (pd.DataFrame)

    Returns
    -------
    predicted_values (np.ndarray): predicted values,
    probs (np.ndarray): probabilities for each class
    '''
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


def get_probs(tweets_df, predicted_values, probs):
    '''Helper function to add new columns: airline_sentiment and prob from model.

    Parameters
    ----------
    tweets_df (pd.DataFrame),
    predicted_values (np.ndarray): predicted values,
    probs (np.ndarray): probabilities for each class

    Returns
    -------
    tweets_df (pd.DataFrame)
    '''
    tweets_df['airline_sentiment'] = predicted_values
    tweets_df['prob'] = np.amax(probs, axis=1)

    return tweets_df


def process_predictions(tweets_df, tfidf, model):
    '''Helper function to predict new values and process new DataFrame.

    Parameter
    ---------
    tweets_df (pd.DataFrame)

    Returns
    -------
    tweets_df (pd.DataFrame)
    '''
    preds, probs = predict(tweets_df, tfidf, model)
    tweets_df = get_probs(tweets_df, preds, probs)

    return tweets_df


def get_most_freq_sentiment(tweets_df):
    '''Finds the sentiment with the highest count.

    Parameter
    ---------
    tweets_df (pd.DataFrame)

    Returns
    -------
    most_freq_sentiment (str)
    '''
    counts = tweets_df.airline_sentiment.value_counts().reset_index()
    most_freq_sentiment = (counts[
        counts.airline_sentiment == counts.airline_sentiment.max()
        ]
        ['index'].values[0])
    return most_freq_sentiment


def get_representative_tweet(tweets_df, most_freq_sentiment):
    '''Finds the URL for the most representative tweet (i.e., has the highest
    probability of the most frequent sentiment tweets).

    Parameters
    ----------
    tweets_df (pd.DataFrame),
    most_freq_sentiment (str)

    Returns
    -------
    url (str): url of the tweet,
    highest_prob (float): probability of the tweet
    '''
    # Highest probability for most frequent sentiment
    highest_prob = tweets_df[
        tweets_df.airline_sentiment == most_freq_sentiment
        ].prob.max()
    rep_tweet = tweets_df[
        (tweets_df.airline_sentiment == most_freq_sentiment) &
        (tweets_df.prob == highest_prob)
    ]

    user, tweet_id = rep_tweet.user.values[0], rep_tweet.tweet_id.values[0]
    url = f'https://twitter.com/{user}/status/{tweet_id}'

    return url, highest_prob


def process_for_wordcloud(tweets_df, most_freq_sentiment):
    '''Gets unique tokens from the most frequent sentiment tweets and creates
    numerical features needed for the model.

    Parameters
    ----------
    tweets_df (pd.DataFrame),
    most_freq_sentiment (str)

    Returns
    -------
    cloud_df (pd.DataFrame): DataFrame for the word cloud
    '''
    all_words = ' '.join(([str(text) for text
                          in tweets_df[
                              tweets_df.airline_sentiment ==
                              most_freq_sentiment]
                         ['cleaned_tweet']]))
    cloud_df = pd.DataFrame(all_words.split(), columns=['text'])
    # have to get all numerical features for model
    cloud_df = create_features(cloud_df)

    return cloud_df


def wordcloud_probs(cloud_df, most_freq_sentiment, tfidf, model):
    '''Finds the probability of each unique token being most_freq_sentiment.

    Parameters
    ----------
    cloud_df (pd.DataFrame),
    most_freq_sentiment (str)

    Returns
    -------
    probs (np.ndarray): probabilities for most_freq_sentiment
    '''
    # Drop duplicate tokens
    cloud_df.drop_duplicates(subset='text', inplace=True)
    cloud_df.reset_index(inplace=True, drop=True)

    num_features = cloud_df.drop(['text'], axis=1)
    text_features = cloud_df.text

    tfidf_matrix = tfidf.transform(text_features)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=tfidf.get_feature_names())
    final_df = pd.concat([num_features, tfidf_df], axis=1)

    probs = model.predict_proba(final_df)
    # Find index for most frequent sentiment
    most_freq_idx = np.where(model.classes_ == most_freq_sentiment)[0][0]
    probs = probs[:, most_freq_idx]

    return probs


def combine_for_cloud(cloud_df, probs):
    '''Creates a new DataFrame with the tokens and probabilities for the word cloud.

    Parameters
    ----------
    cloud_df (pd.DataFrame),
    most_freq_sentiment (str)

    Returns
    -------
    cloud_df (pd.DataFrame)
    '''
    cloud_df = (pd.DataFrame(zip(cloud_df.text, list(probs)),
                columns=['token', 'prob'])
                .sort_values(by='prob', ascending=False)
                .reset_index(drop=True))

    return cloud_df


def get_cloud_frequencies(tweets_df, most_freq_sentiment, tfidf, model):
    '''Helper function to process the tokens for the word cloud, find the
    probabilities, and create the new DataFrame.

    Parameters
    ----------
    tweets_df (pd.DataFrame),
    most_freq_sentiment (str)
    '''
    cloud_df = process_for_wordcloud(tweets_df, most_freq_sentiment)
    probs = wordcloud_probs(cloud_df, most_freq_sentiment, tfidf, model)
    cloud_df = combine_for_cloud(cloud_df, probs)

    return cloud_df
