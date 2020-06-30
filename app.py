from flask import Flask, render_template, request, redirect, url_for, Markup

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly

import json
from wordcloud import WordCloud
from PIL import Image
import io
import base64

from models import model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        twitter_handle = request.form['text_in']
        return redirect(url_for('prediction',
                                twitter_handle=twitter_handle))

    return render_template('index.html')


@app.route('/prediction.html', methods=['GET', 'POST'])
def prediction():
    twitter_handle = '@' + request.args.get('twitter_handle')
    tweets_df = model.get_tweets(twitter_handle)
    tweets_df = model.preprocess(tweets_df)
    tweets_df = model.process_predictions(tweets_df)
    most_freq_sentiment = model.get_most_freq_sentiment(tweets_df)
    rep_tweet, prob = model.get_representative_tweet(tweets_df,
                                                     most_freq_sentiment)

    fig = generate_plot(tweets_df)
    # word_cloud = generate_wordcloud(tweets_df, most_freq_sentiment)
    return render_template('prediction.html',
                           twitter_handle=twitter_handle,
                           most_freq_sentiment=most_freq_sentiment,
                           prob=round(prob*100, 2),
                           rep_tweet=rep_tweet,
                           fig=fig)
                           # word_cloud=word_cloud)


def generate_wordcloud(tweets_df, most_freq_sentiment):
    plane_mask = np.array(Image.open('../docs/img/airplane.jpg'))

    all_words = (' '.join(
        [text for text
         in tweets_df[tweets_df.airline_sentiment == most_freq_sentiment]
         ['cleaned_tweet']]))
    wordcloud = (WordCloud(mask=plane_mask, background_color='white')
                 .generate(all_words))
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    matplotlib.use('agg')
    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)  # rewind the data
    string = base64.b64encode(image.read()).decode()
    image_64 = Markup(f'<img src="data:image/png;base64,{string}">')

    return image_64


def generate_plot(tweets_df):
    fig = px.bar(tweets_df, x="prob", y="airline_sentiment", orientation="h")
    fig.update_layout(xaxis_title={'text': 'number of tweets'},
                      yaxis_title={'text': None})
    fig.update_traces(marker_color='rgb(51, 102, 153)',
                      hovertemplate=None, hoverinfo='skip')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


if __name__ == '__main__':
    app.run(debug=True)
