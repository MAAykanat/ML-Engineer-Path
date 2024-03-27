import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())

#################
### MODELLING ###
#################

# 1. Splitting the data
y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

