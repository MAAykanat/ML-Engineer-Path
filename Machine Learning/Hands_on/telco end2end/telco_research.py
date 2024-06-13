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

# DONE --> 1. Convert TotalCharges to numeric.
# DONE --> 2. Convert Churn to binary. (Yes: 1, No: 0)
# DONE --> 3. Drop CustomerID. (It is cardinal)
# DONE --> ?Categorical? --> 4. Add handle outlier.
# DONE --> ?Categorical? --> 5. Add missing value imputation. Fillna with median.
# DONE --> 6. Add feature generation.
# DONE --> 7. Add encoding. (Label & One-Hot)
# DONE --> 8. Add standardization.


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

#########################################
### 1.EXPLORATORY DATA ANALYSIS - EDA ###
#########################################

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
target = ["CHURN"]

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

# 1.4. Numeric Variables Analysis

for col in num_cols:
    numerical_col_summary(df, col)
print("#"*50)

# 1.5. Target Variable Analysis (Dependent Variable) - Categorical

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

print("#"*50)

# 1.6. Target Variable Analysis (Dependent Variable) - Numeric

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

print("#"*50)

# 1.7. Outlier Detection

"""
There are no outliers in the dataset.
However outleier handle will be added to the pipeline.
"""

for col in num_cols:
    print(col, ":", check_outlier(df, col))

print("#"*50)

# 1.8. Missing Value Analysis

for col in df.columns:
    print(col, ":", df[col].isnull().sum())

print(df.isnull().sum())

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

missing_values_table(df, True)

print("#"*50)

# 1.9. Correlation Matrix

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function returns the columns that have a correlation higher than the threshold value.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    plot : bool, optional
        The default is False.
    corr_th : float, optional
        The default is 0.90.
    Returns
    -------
    drop_list : list
        The list of columns that have a correlation higher than the threshold value.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

df_corr = df.corr()

drop_list = high_correlated_cols(df, plot=False, corr_th=0.90)
print(drop_list)

##################################################
### 2.Data Preprocessing & Feature Engineering ###
##################################################

df.columns = [col.upper() for col in df.columns]

# There are 6 steps to be taken in the Feature Engineering process.
# 2.1. Missing Values
# 2.2. Outlier Values Analysis
# 2.3. Feature Generation
# 2.4. Encoding
# 2.5. Standardization
# 2.6. Save the Dataset

# 2.1. Missing Values
"""
Done. Check NOTES.
"""

# 2.2. Outlier Values Analysis
"""
DONE. Check NOTES.
"""

# 2.3. Feature Generation
print(df.head())

# Creating an annual categorical variable from a tenure variable
df.loc[(df["TENURE"]>=0) & (df["TENURE"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["TENURE"]>12) & (df["TENURE"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["TENURE"]>24) & (df["TENURE"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["TENURE"]>36) & (df["TENURE"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["TENURE"]>48) & (df["TENURE"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["TENURE"]>60) & (df["TENURE"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Specify customers with 1 or 2 years of contract as Engaged
df["NEW_ENGAGED"] = df["CONTRACT"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_NOPROT"] = df.apply(lambda x: 1 if (x["ONLINEBACKUP"] != "Yes") or (x["DEVICEPROTECTION"] != "Yes") or (x["TECHSUPPORT"] != "Yes") else 0, axis=1)

# Customers who have monthly contracts and are young
df["NEW_YOUNG_NOT_ENGAGED"] = df.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and (x["SENIORCITIZEN"] == 0) else 0, axis=1)


# Total number of services received by the person
df['NEW_TOTALSERVICES'] = (df[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY',
                                       'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT',
                                       'STREAMINGTV', 'STREAMINGMOVIES']]== 'Yes').sum(axis=1)


# People who receive any streaming service
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["STREAMINGTV"] == "Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AUTOPAYMENT"] = df["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Average monthly payment
df["NEW_AVG_CHARGES"] = df["TOTALCHARGES"] / (df["TENURE"] + 1)

# Rate of change the current price compared to the average price
df["NEW_INCREASE"] = df["NEW_AVG_CHARGES"] / df["MONTHLYCHARGES"]

# Fee per service
df["NEW_AVG_SERVICE_FEE"] = df["MONTHLYCHARGES"] / (df['NEW_TOTALSERVICES'] + 1)

print(df.head())
print(df.shape)

cat_cols, num_cols, cat_but_car = grap_column_names(df)

# 2.4. Encoding
# 2.4.1. Label Encoding

binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == "O"]

print(binary_cols)

for col in binary_cols:
    df = label_encoder(df, col)

# 2.4.2. One-Hot Encoding

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in target]

df = one_hot_encoder(df, cat_cols, drop_first=True)

print(df.head())
print(df.shape)

# 2.5. Standardization

# Standardization will be added to the pipeline.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())

# 2.6. Save the Dataset

# df.to_csv("Machine Learning/datasets/telco/telco_churn_cleaned.csv", index=False)

######################
### 3. Base Models ###
######################

def split_dataset(dataframe, target, test_size=0.20, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target, axis=1), dataframe[target], test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

print(target)

X_train, X_test, y_train, y_test = split_dataset(df, target=target)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3.1. Train-Test Split
# 3.2. Model Selection
# 3.3. Model Building

