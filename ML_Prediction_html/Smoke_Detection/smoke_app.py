# from requests import request
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/")
def root_page():
    return render_template('index.html')


@app.route("/predict")
def form_page():
    return render_template('forms.html')


if __name__ == "__main__":
    app.run(debug=True)
