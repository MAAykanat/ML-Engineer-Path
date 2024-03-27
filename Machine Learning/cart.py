import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, auc

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())

#################
### MODELLING ###
#################

####################
## Holdout Method ##
####################

# 1. Splitting the data
y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cart_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# 2. Prediction
# for Confusion Matrix y_pred
y_pred = cart_model.predict(X_test)
# for AUC y_prob
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test,y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_prob))

