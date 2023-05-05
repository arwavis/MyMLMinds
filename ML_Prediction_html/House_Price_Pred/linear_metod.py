# from requests import request
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

    # return f"Test Output {no_bed_room}+{no_bath_room}+{square_feet_living}+{square_feet_lot}+{no_of_floors}+{
    # water_front}+" \ f"{no_of_views}+{condition}+{square_feet_above}+{square_feet_basement}+{year_built}+{
    # year_renovated}+{city}"
    df = pd.read_csv("data.csv")
    # print(df)

    # *************************** DATA PROCESSING **********************************
    # Removing the columns that are not required for the analysis
    df.drop(df.columns[[0, 14, 16, 17]], axis=1, inplace=True)
    df = df.reset_index()

    # Converting categorical variables of the dataset into numerical variables - using ONE HOT ENCODING technique

    df['city'] = df['city'].apply(
        {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4, 'Maple Valley': 5, 'North Bend': 6,
         'Lake Forest Park': 7, 'Sammamish': 8, 'Auburn': 9, 'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12,
         'Kirkland': 13, 'Issaquah': 14, 'Woodinville': 15, 'Normandy Park': 16, 'Fall City': 17, 'Renton': 18,
         'Carnation': 19, 'Snoqualmie': 20, 'Duvall': 21, 'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24,
         'Kenmore': 25, 'Newcastle': 26, 'Mercer Island': 27, 'Black Diamond': 28, 'Ravensdale': 29,
         'Clyde Hill': 30,
         'Algona': 31, 'Skykomish': 32, 'Tukwila': 33, 'Vashon': 34, 'Yarrow Point': 35, 'SeaTac': 36, 'Medina': 37,
         'Enumclaw': 38, 'Snoqualmie Pass': 39, 'Pacific': 40, 'Beaux Arts Village': 41, 'Preston': 42,
         'Milton': 43}.get)

    # Dividing the dataset into dependent and independent column Here X is Independent Column and y is dependent column

    X = df.drop('price', axis=1)
    y = df['price']

    # ***************** SPLITTING THE DATASET INTO TRAINING AND TESTING DATA ***********************
    # 20% of the dataset will be used for testing(evaluation) and 80% of the data will be used for training purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # creating the machine learning model
    linearmodel = LinearRegression()
    linearmodel.fit(X_train, y_train)  # training is taking place

    # predictions
    linearmodel.predict(X_test)
    # calculating the accuracy of the model
    linearmodel.score(X, y)  # evaluation is taking place
    accuracy_sc = linearmodel.score(X, y)
    rounded_acc = np.round(accuracy_sc, 2)
    print(f"The Accuracy of the Model is: {rounded_acc * 100}")

    # predictions with respect to new dataset

    data_new = {'bedrooms': float(no_bed_room), 'bathrooms': float(no_bath_room),
                'sqft_living': int(square_feet_living),
                'sqft_lot': int(square_feet_lot), 'floors': float(no_of_floors),
                'waterfront': int(water_front), 'view': int(no_of_views), 'condition': int(condition),
                'sqft_above': int(square_feet_above),
                'sqft_basement': int(square_feet_basement),
                'yr_built': int(year_built), 'yr_renovated': int(year_renovated), 'city': str(city)}

    # with open('test.csv', 'w') as f:
    #     for key in data_new:
    #         f.write("%s,%s\n" % (key, data_new[key]))
    #
    # df1 = pd.read_csv("test.csv")
    # tdf1 = df1.T
    # tdf1.columns = tdf1.iloc[0]
    # df_new = tdf1[1:]
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

    index = [1]  # serial number

    my_data = pd.DataFrame(df2, index)

    # Pricing allocated
    # Here the linearmodel is the variable  used in 13
    my_data_price = linearmodel.predict(my_data)
    rounded_price = np.round(my_data_price, 2)
    # return f"The predicted price for the given data is :{rounded_price}"
    # print(f" The predicted price for the given data is :{rounded_price}")
    return render_template("predicted.html", price=rounded_price)


print(predict_page)
if __name__ == "__main__":
    app.run(debug=True, port=8000)
