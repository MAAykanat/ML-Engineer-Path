##################################################
# End-to-End Telco Churn Machine Learning Pipeline I
##################################################

##STEPS##

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

#############
### NOTES ###
#############

# 1. Convert TotalCharges to numeric.
# 2. Convert Churn to binary. (Yes: 1, No: 0)
# 3. Drop CustomerID. (It is cardinal)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import *

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")

#######################################
### 1.EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# 1.1. General Picture of the Dataset
# 1.2. Catch Numeric and Categorical Value
# 1.3. Catetorical Variables Analysis
# 1.4. Numeric Variables Analysis
# 1.5. Target Variable Analysis (Dependent Variable) - Categorical
# 1.6. Target Variable Analysis (Dependent Variable) - Numeric
# 1.7. Outlier Detection
# 1.8. Missing Value Analysis
# 1.9. Correlation Matrix

# 1.1. General Picture of the Dataset

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


"""
Target variable: Churn
Shape: (7043,21)
Types: 18 object, 2 float64, 1 int64
No missing values
"""

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

check_df(df)

# 1.2. Catch Numeric and Categorical Value

cat_cols, num_cols, cat_but_car = grap_column_names(df)

"""
Observations: 7043
Variables: 21
categorical_cols: 17
num_cols: 3
categorical_but_cardinal: 1
numeric_but_categorical: 2
"""

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 1.3. Catetorical Variables Analysis

for col in cat_cols:
    cat_summary(df, col)
print("#"*50)

