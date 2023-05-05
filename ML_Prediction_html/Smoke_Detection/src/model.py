import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Data Reading

data = pd.read_csv("../smoke_detection_iot.csv")

"""
Data Cleaning
1. Removing Unnamed column
2. Removing UTC as I find this column is not relevant for evaluation
"""

data = data.drop(data.filter(regex="Unnamed"), axis=1, inplace=False)
data = data.drop(['UTC'], axis=1)

# Dividing the dataset into dependent and independent columns
X = data.drop(['Fire Alarm'], axis=1)  # independent features from the dataset
y = data['Fire Alarm']  # dependent column from the dataset

# Splitting the dataset into training and testing set
# 20% of the dataset will be used for testing(evaluation) and 80% of the data will be used for training purposes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Using Decision Tree Algorithm as I see a good accuracy rate
# Refer : https://www.kaggle.com/code/arwavis/smoke-detection/edit/run/104093001 for all algorithm validation.


tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
predictions = tree_classifier.predict(X_test)
cm = confusion_matrix(y_test, predictions)
ac = accuracy_score(y_test, predictions)
print(f"The Model confusion Matrix value is: {cm}")
print(f"The model accuracy is: {ac*100}")

# predictions with respect to new dataset
data_new = {'Temperature[C]': 9.381, 'Humidity[%]': 56.86, 'TVOC[ppb]': 11, 'eCO2[ppm]': 400, 'Raw H2': 13347,
            'Raw Ethanol': 20160, 'Pressure[hPa]': 939.575, 'PM1.0': 1.78,
            'PM2.5': 1.85, 'NC0.5': 12.25, 'NC1.0': 1.911, 'NC2.5': 0.043, 'CNT': 3178}
index = [1]  # serial number
my_data = pd.DataFrame(data_new, index)

# Predicting the Fire Alarm with new data input "data_new"
my_data_validation = tree_classifier.predict(my_data)

if my_data_validation == 1:
    print("The Fire Alarm is ON")
else:
    print("Fire Alarm is OFF")
