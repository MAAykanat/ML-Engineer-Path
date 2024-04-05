import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_attribute = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_attributes.csv", sep=";")
df_labels = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_potential_labels.csv", sep=";")


df = pd.merge(df_attribute, df_labels, on=["player_id", "evaluator_id","match_id","task_response_id"])
print(df.head())
print(df.shape)
# (10730, 9)

df.drop(df[df["position_id"]==1].index, inplace=True)
print(df["potential_label"].value_counts())
# above_average = 136

print(df["potential_label"].value_counts()["below_average"]/df["potential_label"].value_counts().sum())
# 0.013 below_average ~1% of the data will be removed
df.drop(df[df["potential_label"]=="below_average"].index, inplace=True)

print(df.shape)
# (9894, 9)

table = pd.pivot_table(df, values="attribute_value", 
                       index = ["player_id", "position_id", "potential_label"], 
                       columns="attribute_id", aggfunc="count")
print(table.head())
table.reset_index(inplace=True)
print(table.head())

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Value
# 3. Catetorical Variables Analysis
# 4. Numeric Variables Analysis
# 5. Target Variable Analysis (Dependent Variable) - Categorical
# 6. Target Variable Analysis (Dependent Variable) - Numeric
# 7. Outlier Detection
# 8. Missing Value Analysis
# 9. Correlation Matrix

# 1. General Picture of the Dataset

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

check_df(table)
