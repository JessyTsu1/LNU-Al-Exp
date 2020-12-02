import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("./winequality-white.csv", sep=';')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :11], data.iloc[:, -1], test_size=0.3,
                                                    random_state=112)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
accuracy = accuracy_score(pred_logreg, y_test)
print("Logreg Accuracy Score %.2f" % accuracy)

cm = confusion_matrix(pred_logreg, y_test)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
accuracy = accuracy_score(pred_knn, y_test)
print("Knn Accuracy Score %.2f" % accuracy)

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
accuracy = accuracy_score(pred_svc, y_test)
print("SVC Accuracy Score %.2f" % accuracy)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred_tree = dtree.predict(X_test)
accuracy = accuracy_score(pred_tree, y_test)
print("DTree Accuracy Score %.2f" % accuracy)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
# print(rf)
# print(rf.fit(X_train,y_train))
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
accuracy = accuracy_score(pred_rf, y_test)
print("Random Forest Accuracy Score %.2f" % accuracy)
