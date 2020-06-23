from flask import Flask, render_template, request, redirect, url_for, Markup

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        twitter_handle = request.form['text_in']
        return redirect(url_for('prediction',
                                twitter_handle='@'+twitter_handle))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
