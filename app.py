import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import flask
import pickle
from flask import Flask, request, render_template, redirect, url_for

#create instance of Flask
app = Flask(__name__, template_folder='templates')

carMpg = r'"C:\Users\HP\Downloads\car-mpg.csv'

carMpg = pd.read_csv(carMpg)

carMpg.drop('car_type', axis=1)  # Features
carMpg.drop('car_name', axis=1)  # Features
carMpg.drop('yr', axis=1)  # Features
carMpg.drop('acc', axis=1)  # Features
carMpg.drop('origin', axis=1)  # Features

carMpg[[ 'mpg', 'cyl', 'disp', 'hp', 'wt']] = carMpg[['mpg', 'cyl', 'disp', 'hp', 'wt']].apply(pd.to_numeric, errors='coerce')
carMpg = carMpg.dropna()
carMpg = carMpg.reset_index(drop=True)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

#function to predict the output
def ValuePredictor(to_predict_list):
    X = np.array(carMpg[[ 'mpg', 'cyl', 'disp', 'hp', 'wt']].values)

    X_new_df = np.array(to_predict_list).reshape(1, 5)

    # Gabungkan X dan X_new_df
    X_combined = np.concatenate((X, X_new_df), axis=0)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(X_combined)

    to_predict = np.array(data[-1]).reshape(1, 5)

    loaded_model = pickle.load(
        open("./model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result

#function to get the input from the user
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        MPG = request.form['mpg']
        Cylinders = request.form['cyl']
        Displacement = request.form['disp']
        Horsepower = request.form['hp']
        Weight = request.form['wt']
        to_predict_list = list(map(float, [MPG, Cylinders, Displacement, Horsepower, Weight]))
        result = ValuePredictor(to_predict_list)
        if int(result) == 0:
            prediction = 'Cluster 0'
        elif int(result) == 1:
            prediction = 'Cluster 1'
        elif int(result) == 2:
            prediction = 'Cluster 2'
        return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
