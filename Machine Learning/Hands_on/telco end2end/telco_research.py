################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")
print(df.head())

################################################
# 1. Exploratory Data Analysis (EDA)
################################################

# 1.1 General Picture of the Dataset
# 1.2 Catch Numeric and Categorical Value
# 1.3 Catetorical Variables Analysis
# 1.4 Numeric Variables Analysis
# 1.5 Target Variable Analysis (Dependent Variable) - Categorical
# 1.6 Target Variable Analysis (Dependent Variable) - Numeric
# 1.7 Outlier Detection
# 1.8 Missing Value Analysis
# 1.9 Correlation Matrix

# 1.1 General Picture of the Dataset

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
"""
Shape = (7043, 21)
Types = object(18), float64(2), int64(1)
NA = No missing value
"""

check_df(df)