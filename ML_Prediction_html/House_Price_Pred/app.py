# from requests import request
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route("/")
def root_page():
    return render_template('index.html')


@app.route("/predict")
def form_page():
    return render_template('forms.html')


@app.route("/form", methods=["GET", "POST"])
def predict_page():
    no_bed_room = request.form.get("bedroom")
    no_bath_room = request.form.get("bathroom")
    square_feet_living = request.form.get("sqftliv")
    square_feet_lot = request.form.get("sqftlot")
    no_of_floors = request.form.get("floors")
    water_front = request.form.get("waterfront")
    no_of_views = request.form.get("view")
    condition = request.form.get("condition")
    square_feet_above = request.form.get("sqftabv")
    square_feet_basement = request.form.get("sqftbas")
    year_built = request.form.get("yrbuilt")
    year_renovated = request.form.get("yrrenov")
    city = request.form.get("city")

    # Using the file in src+Notebook_test\Prediction_TensorFlow_model we have already saved the trained model in the below link.

    # Loading the model

    # os.chdir(r'C:\Users\aravindv\Documents\Github-projects\ML_PredictionsinHTML\House_Price_Pred')
    os.chdir("/Users/aravindv/Documents/Programming/Github/ML_PredictionsinHTML/House_Price_Pred")
    model = load_model(os.path.join(os.getcwd(), "housepred_model.h5"))

    # Prediction with reference to new dataset

    data_new = {'bedrooms': float(no_bed_room), 'bathrooms': float(no_bath_room),
                'sqft_living': int(square_feet_living),
                'sqft_lot': int(square_feet_lot), 'floors': float(no_of_floors),
                'waterfront': int(water_front), 'view': int(no_of_views), 'condition': int(condition),
                'sqft_above': int(square_feet_above),
                'sqft_basement': int(square_feet_basement),
                'yr_built': int(year_built), 'yr_renovated': int(year_renovated), 'city': str(city)}
    df1 = pd.DataFrame.from_dict([data_new])
    df1.to_csv('test.csv', index=None)  # index=None prevents index being added as column 1
    df2 = pd.read_csv('test.csv')

    df2['city'] = df2['city'].apply(
        {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4, 'Maple Valley': 5, 'North Bend': 6,
         'Lake Forest Park': 7, 'Sammamish': 8, 'Auburn': 9, 'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12,
         'Kirkland': 13, 'Issaquah': 14, 'Woodinville': 15, 'Normandy Park': 16, 'Fall City': 17, 'Renton': 18,
         'Carnation': 19, 'Snoqualmie': 20, 'Duvall': 21, 'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24,
         'Kenmore': 25, 'Newcastle': 26, 'Mercer Island': 27, 'Black Diamond': 28, 'Ravensdale': 29,
         'Clyde Hill': 30,
         'Algona': 31, 'Skykomish': 32, 'Tukwila': 33, 'Vashon': 34, 'Yarrow Point': 35, 'SeaTac': 36, 'Medina': 37,
         'Enumclaw': 38, 'Snoqualmie Pass': 39, 'Pacific': 40, 'Beaux Arts Village': 41, 'Preston': 42,
         'Milton': 43}.get)

    arr = df2.to_numpy()
    prediction = model.predict(arr)
    round_pred = np.round(float(prediction[0]), 2)

    return render_template("predicted.html", price=round_pred)


print(predict_page)
if __name__ == "__main__":
    app.run(debug=True, port=8000)
