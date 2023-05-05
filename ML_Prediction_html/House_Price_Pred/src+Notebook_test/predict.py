import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
# print(df)

# *************************** DATA PROCESSING **********************************
# Removing the columns that are not required for the analysis
df.drop(df.columns[[0, 14, 16, 17]], axis=1, inplace=True)


# Converting categorical variables of the dataset into numerical variables - using ONE HOT ENCODING technique
def city_con():
    df['city'] = df['city'].apply(
        {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4, 'Maple Valley': 5, 'North Bend': 6,
         'Lake Forest Park': 7, 'Sammamish': 8, 'Auburn': 9, 'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12,
         'Kirkland': 13, 'Issaquah': 14, 'Woodinville': 15, 'Normandy Park': 16, 'Fall City': 17, 'Renton': 18,
         'Carnation': 19, 'Snoqualmie': 20, 'Duvall': 21, 'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24,
         'Kenmore': 25, 'Newcastle': 26, 'Mercer Island': 27, 'Black Diamond': 28, 'Ravensdale': 29, 'Clyde Hill': 30,
         'Algona': 31, 'Skykomish': 32, 'Tukwila': 33, 'Vashon': 34, 'Yarrow Point': 35, 'SeaTac': 36, 'Medina': 37,
         'Enumclaw': 38, 'Snoqualmie Pass': 39, 'Pacific': 40, 'Beaux Arts Village': 41, 'Preston': 42,
         'Milton': 43}.get)


# Dividing the dataset into dependent and independent column Here X is Independent Column and y is dependent column
city_con()
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
data_new = {'bedrooms': 3.0, 'bathrooms': 2.00, 'sqft_living': 1570, 'sqft_lot': 7500, 'floors': 2.0,
            'waterfront': 0, 'view': 4, 'condition': 5, 'sqft_above': 3560, 'sqft_basement': 300,
            'yr_built': 1932, 'yr_renovated': 2007, 'city': 2}
index = [1]  # serial number
my_data = pd.DataFrame(data_new, index)

# Pricing allocated
# Here the linearmodel is the variable  used in 13
my_data_price = linearmodel.predict(my_data)
rounded_price = np.round(my_data_price, 2)
print(f" The predicted price for the given data is :{rounded_price}")
