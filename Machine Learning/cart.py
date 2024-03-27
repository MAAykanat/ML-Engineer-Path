import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())

#################
### MODELLING ###
#################

# 1. Splitting the data
y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

