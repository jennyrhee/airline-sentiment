from flask import Flask, render_template

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
