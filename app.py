from flask import Flask, render_template, request, redirect, url_for, Markup

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly

import pickle

import json
from wordcloud import WordCloud
from PIL import Image
import io
import base64

from models import model

app = Flask(__name__)
with open('models/tfidf.pkl', 'rb') as tfidf_f, \
     open('models/model.pkl', 'rb') as model_f:
    tfidf = pickle.load(tfidf_f)
    rf = pickle.load(model_f)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        twitter_handle = request.form['text_in'].lower()
        if '@' not in twitter_handle:
            twitter_handle = '@' + twitter_handle
        tweets_df = model.get_tweets(twitter_handle)

        if len(tweets_df) > 0:
            return redirect(url_for('prediction',
                                    twitter_handle=twitter_handle))
        else:
            error = f'No recent mentions for {twitter_handle}! Try ' + \
                    'another handle.'
            return render_template('index.html', error=error)

    return render_template('index.html')


@app.route('/prediction.html', methods=['GET', 'POST'])
def prediction():
    twitter_handle = request.args.get('twitter_handle')
    tweets_df = model.get_tweets(twitter_handle)
    tweets_df = model.preprocess(tweets_df)
    tweets_df = model.process_predictions(tweets_df, tfidf, rf)
    most_freq_sentiment = model.get_most_freq_sentiment(tweets_df)
    rep_tweet, prob = model.get_representative_tweet(tweets_df,
                                                     most_freq_sentiment)

    fig = generate_plot(tweets_df)
    word_cloud = generate_wordcloud(tweets_df, most_freq_sentiment, tfidf, rf)
    return render_template('prediction.html',
                           twitter_handle=twitter_handle,
                           most_freq_sentiment=most_freq_sentiment,
                           prob=round(prob*100, 2),
                           rep_tweet=rep_tweet,
                           fig=fig,
                           word_cloud=word_cloud)


def generate_wordcloud(tweets_df, most_freq_sentiment, tfidf, rf):
    '''Generates a word cloud image based on the probability of each unique token
    being classified as most_freq_sentiment.

    Parameters
    ----------
    tweets_df (pd.DataFrame),
    most_freq_sentiment (str)

    Returns
    -------
    image_64 (str): string that is ready to be inserted into HTML
    '''
    plane_mask = np.array(Image.open('docs/img/airplane.jpg'))

    cloud_df = model.get_cloud_frequencies(tweets_df, most_freq_sentiment,
                                           tfidf, rf)
    matplotlib.use('agg')
    wordcloud = (WordCloud(mask=plane_mask, background_color='white')
                 .generate_from_frequencies(dict(cloud_df.values)))
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)  # rewind the data
    string = base64.b64encode(image.read()).decode()
    image_64 = Markup(f'<img src="data:image/png;base64,{string}">')

    return image_64


def generate_plot(tweets_df):
    '''Generates bar graph from all the tweets. Shows sentiment, probability, and text
    for each tweet in the tooltip.

    Parameter
    ---------
    tweets_df (pd.DataFrame)

    Returns
    -------
    graphJSON (str): JSON formatted string for graph to be inserted into HTML
    '''
    fig = px.bar(tweets_df.sort_values(by='prob'), x='prob',
                 y='airline_sentiment', orientation='h',
                 hover_data=['text'])
    fig.update_layout(xaxis_title={'text': 'sum of probabilities'},
                      yaxis_title={'text': None})
    fig.update_traces(marker_color='rgb(51, 102, 153)')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


if __name__ == '__main__':
    app.run()
