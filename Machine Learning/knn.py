import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Cleaned dataeset import
df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())
df = df.reindex(sorted(df.columns), axis=1)

###############
## MODELLING ##
###############

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier().fit(X_train, y_train)

# To Classification Report
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))


# To ROC AUC
y_prob = knn_model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_prob))