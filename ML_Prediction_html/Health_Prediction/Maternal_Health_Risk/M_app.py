# IMPORTING LIBRARIES
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
# TO IGNORE ANY WARNINGS
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter(action='ignore', category=FutureWarning)
simplefilter("ignore", category=ConvergenceWarning)
# Reading Data File
# from Health_Prediction.health_pred import l_reg_alg, d_alg, k_alg, r_f_alg, s_v_alg

data = pd.read_csv(
    "../Maternal Health Risk Data Set.csv")
# print(data)

# Data Processing
# Converting categorical variables of the dataset into numerical variables - using ONE HOT ENCODING technique
data['RiskLevel'] = data['RiskLevel'].apply({'low risk': 0, 'mid risk': 1, 'high risk': 2}.get)

# Dividing the Dataset into Dependent and Independent Columns
X = data.drop('RiskLevel', axis=1)  # independent features from the dataset
y = data['RiskLevel']  # dependent column from the dataset

"""
Splitting the dataset into training and testing set
20% of the dataset will be used for testing(evaluation) and 80% of the data will be used for training purposes
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""---------------------------- LOGISTIC REGRESSION ----------------------------"""

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
# CALCULATING CONFUSION MATRIX AND ALGORITHM ACCURACY SCORE
log_cm = confusion_matrix(y_test, y_pred)
log_ac = accuracy_score(y_test, y_pred)

"""---------------------------- Decision Tree  ----------------------------"""

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
predictions = tree_classifier.predict(X_test)
# CALCULATING CONFUSION MATRIX AND ALGORITHM ACCURACY SCORE
dc_cm = confusion_matrix(y_test, predictions)
dc_ac = accuracy_score(y_test, predictions)

""" ---------------------------- KNN  ----------------------------"""

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
# CALCULATING CONFUSION MATRIX AND ALGORITHM ACCURACY SCORE
knn_cm = confusion_matrix(y_test, predictions)
knn_ac = accuracy_score(y_test, predictions)

""" ---------------------------- RANDOM FOREST  ----------------------------"""

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# CALCULATING CONFUSION MATRIX AND ALGORITHM ACCURACY SCORE
rf_cm = confusion_matrix(y_test, y_pred)
rf_ac = accuracy_score(y_test, y_pred)

""" ---------------------------- SUPPORT VECTOR MACHINE  ----------------------------"""

svc_model = SVC()
svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
# CALCULATING CONFUSION MATRIX AND ALGORITHM ACCURACY SCORE
svm_cm = confusion_matrix(y_test, predictions)
svm_ac = accuracy_score(y_test, predictions)

# predictions with respect to new dataset
data_new = {'Age': 25, 'SystolicBP': 130, 'DiastolicBP': 80, 'BS': 15.0, 'BodyTemp': 98.0,
            'HeartRate': 86}
index = [1]  # serial number


def l_reg_alg():
    logmodel_alg = LogisticRegression()
    logmodel_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    l_medical_details = logmodel_alg.predict(my_data)
    print(l_medical_details)

    if l_medical_details == 0:
        print(" Your Maternal Health risk is at Low Level")
    elif l_medical_details == 1:
        print(" Your Maternal Health risk is at Mid Level")
    else:
        print(" Your Maternal Health risk is at High Level")


def d_alg():
    tree_classifier_alg = DecisionTreeClassifier()
    tree_classifier_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    d_medical_details = tree_classifier_alg.predict(my_data)
    print(d_medical_details)

    if d_medical_details == 0:
        print(" Your Maternal Health risk is at Low Level")
    elif d_medical_details == 1:
        print(" Your Maternal Health risk is at Mid Level")
    else:
        print(" Your Maternal Health risk is at High Level")


def k_alg():
    knn_alg = KNeighborsClassifier(n_neighbors=11)
    knn_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    k_medical_details = knn_alg.predict(my_data)
    print(k_medical_details)

    if k_medical_details == 0:
        print(" Your Maternal Health risk is at Low Level")
    elif k_medical_details == 1:
        print(" Your Maternal Health risk is at Mid Level")
    else:
        print(" Your Maternal Health risk is at High Level")


def r_f_alg():
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    rf_medical_details = rf_classifier.predict(my_data)
    print(rf_medical_details)

    if rf_medical_details == 0:
        print(" Your Maternal Health risk is at Low Level")
    elif rf_medical_details == 1:
        print(" Your Maternal Health risk is at Mid Level")
    else:
        print(" Your Maternal Health risk is at High Level")


def s_v_alg():
    svc_model_alg = SVC()
    svc_model_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    sv_medical_details = svc_model_alg.predict(my_data)
    print(sv_medical_details)

    if sv_medical_details == 0:
        print(" Your Maternal Health risk is at Low Level")
    elif sv_medical_details == 1:
        print(" Your Maternal Health risk is at Mid Level")
    else:
        print(" Your Maternal Health risk is at High Level")


print(f"logistic regression Accuracy {log_ac * 100}")
print(f"Decision Tree Accuracy {dc_ac * 100}")
print(f"KNN Accuracy {knn_ac * 100}")
print(f"Random Forest Accuracy {rf_ac * 100}")
print(f"Support Vector Machine Accuracy {svm_ac * 100}")

# Condition to check which algorithm to use
if log_ac >= dc_ac and log_ac >= knn_ac and log_ac >= rf_ac and log_ac >= svm_ac:
    print(f"Using Algorithm : Logistic Regression with an accuracy {log_ac * 100}")
    l_reg_alg()
elif dc_ac >= log_ac and dc_ac >= knn_ac and dc_ac >= rf_ac and dc_ac >= svm_ac:
    print(f"Using Algorithm : Decision Tree with an accuracy {dc_ac * 100}")
    d_alg()
elif knn_ac >= log_ac and knn_ac >= dc_ac and knn_ac >= rf_ac and knn_ac >= svm_ac:
    print(f"Using Algorithm :K-Nearest Neighbor with an accuracy {knn_ac * 100}")
    k_alg()
elif rf_ac >= log_ac and rf_ac >= dc_ac and rf_ac >= knn_ac and rf_ac >= svm_ac:
    print(f"Using Algorithm :Random Forest with an accuracy {rf_ac * 100}")
    r_f_alg()
else:
    print(f"Using Algorithm :Support Vector Machine with an accuracy {svm_ac * 100}")
    s_v_alg()
