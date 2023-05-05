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

data = pd.read_csv("../heart.csv")

# Dividing the Dataset into Dependent and Independent Columns

X = data.drop(['output'], axis=1)  # independent features from the dataset
y = data['output']  # dependent column from the dataset

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
data_new = {'age': 30, 'sex': 0, 'cp': 3, 'trtbps': 136, 'chol': 133,
            'fbs': 0, 'restecg': 2, 'thalachh': 150, 'exng': 0, 'oldpeak': 2.3,
            'slp': 1, 'caa': 3, 'thall': 2}
index = [1]  # serial number


def l_reg_alg():
    logmodel_alg = LogisticRegression()
    logmodel_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    l_medical_details = logmodel_alg.predict(my_data)
    # rounded_price = np.round(my_data_price, 2)
    if l_medical_details == 1:
        print(" The Prediction detected possibility of Heart Attack")
    else:
        print(" You are completely fine, No sign of Heart related issues")


def d_alg():
    tree_classifier_alg = DecisionTreeClassifier()
    tree_classifier_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    d_medical_details = tree_classifier_alg.predict(my_data)
    # rounded_price = np.round(my_data_price, 2)
    if d_medical_details == 1:
        print(" The Prediction detected possibility of Heart Attack")
    else:
        print(" You are completely fine, No sign of Heart related issues")


def k_alg():
    knn_alg = KNeighborsClassifier(n_neighbors=11)
    knn_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    k_medical_details = knn_alg.predict(my_data)
    # rounded_price = np.round(my_data_price, 2)
    if k_medical_details == 1:
        print(" The Prediction detected possibility of Heart Attack")
    else:
        print(" You are completely fine, No sign of Heart related issues")


def r_f_alg():
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    rf_medical_details = rf_classifier.predict(my_data)
    # rounded_price = np.round(my_data_price, 2)
    if rf_medical_details == 1:
        print(" The Prediction detected possibility of Heart Attack")
    else:
        print(" You are completely fine, No sign of Heart related issues")


def s_v_alg():
    svc_model_alg = SVC()
    svc_model_alg.fit(X_train, y_train)
    my_data = pd.DataFrame(data_new, index)
    sv_medical_details = svc_model_alg.predict(my_data)

    # rounded_price = np.round(my_data_price, 2)
    if sv_medical_details == 1:
        print(" The Prediction detected possibility of Heart Attack")
    else:
        print(" You are completely fine, No sign of Heart related issues")


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
