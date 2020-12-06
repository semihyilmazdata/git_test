from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

import pandas

model = pickle.load(open("linear.pkl", "rb"))

app = Flask(__name__)


@app.route('/')
def hello_world():


    return render_template("index.html")


@app.route('/', methods=['POST'])
def my_form_post():
    last = float(request.form["last-price"])
    lowest = float(request.form["lowest-price"])
    volume = float(request.form["volume"])
    print(last, lowest, volume)
    arr = np.array([[last,lowest,volume]])
    pred = model.predict(arr)
    print(pred)

    return render_template("post.html", prediction = pred[0].round(4))

if __name__ == '__main__':
    app.run()
