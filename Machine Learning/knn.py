import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_validate


import warnings
warnings.filterwarnings("ignore")

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

# Cross Validation from all dataset
cv = cross_validate(estimator=knn_model, X=X, y=y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print("Accuracy: ", cv["test_accuracy"].mean())
print("F1: ", cv["test_f1"].mean())
print("ROC AUC: ", cv["test_roc_auc"].mean())

"""
Accuracy:  0.8307245386192754
F1:  0.7447436385524892
ROC AUC:  0.8690242165242166
"""

#################################
## Hyperparameter Optimization ##
#################################

print(knn_model.get_params())
knn_params = {"n_neighbors": np.arange(2, 100), 
              "p": [1, 2, 3]}
knn_gridSearch_best = GridSearchCV(estimator=knn_model, param_grid=knn_params, cv=10, n_jobs=-1, verbose=1).fit(X, y)

print(knn_gridSearch_best.best_params_)

"""
{'n_neighbors': 73, 'p': 1}
"""
#################
## Final Model ##
#################

knn_tuned = KNeighborsClassifier(**knn_gridSearch_best.best_params_).fit(X_train, y_train)

cv_results = cross_validate(knn_tuned,
                            X,
                            y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

print("Accuracy: ", cv_results["test_accuracy"].mean())
print("F1: ", cv_results["test_f1"].mean())
print("ROC AUC: ", cv_results["test_roc_auc"].mean())

"""
Accuracy:  0.8567669172932332
F1:  0.7826029873326528
ROC AUC:  0.9087293447293447
"""