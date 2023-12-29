import dataset_handle as dh

import pandas as pd
import numpy as np

def missing_values_table(dataframe, null_columns_name = False):
    """
    This function returns the number and percentage of missing values in a dataframe.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    null_columns_name : bool, optional
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)
    
    if null_columns_name:
        return null_columns    

df_titanic = dh.load_dataset("titanic.csv")

df_titanic.loc[[3]] # Specific row(3th) of the dataset 

df_titanic.isnull() # True-False matrix for null values
df_titanic.notnull() # True-False matrix for non-null values

df_titanic.isnull().sum() # number of null values in each column
df_titanic.notnull().sum() # number of non-null values in each column

df_titanic.isnull().sum().sort_values(ascending=False) # Sort the number of null values in each column
((df_titanic.isnull().sum() / df_titanic.shape[0])*100).sort_values(ascending=False) # Sort the percentage of null values in each column

na_columns = [col for col in df_titanic.columns if df_titanic[col].isnull().sum() > 0]

# print(missing_values_table(df_titanic))

########################
# HADLING MISSING VALUES
########################

########################
# Solution-1. Deleting
########################

df_titanic.dropna() # Delete all rows with null values

########################
# Solution-2. Filling
########################

df_titanic.fillna(0) # Fill all null values with 0

df_titanic["Age"].fillna(df_titanic["Age"].mean()) # Fill all null values with mean of the column
df_titanic["Age"].fillna(df_titanic["Age"].median()) # Fill all null values with median of the column

# Fill all null values with mean of the column if the column is not object type
# Numeric Varianbles Fill with Mean or Median
df_titanic.apply(lambda col : col.fillna(col.mean()) if (col.dtype != "O") else col, axis=0)

# Fill all null values with mode of the column if the column is object type and has less than 10 unique values
# Categorical Variables Fill with Mode
df_titanic=df_titanic.apply(lambda col: col.fillna(col.mode()[0]) if (col.dtype=="O" and len(col.unique()) <=10) else col ,axis=0)

###################
# Filling with Cateorical Breakdown 
###################

# Mean with Sex Breakdown
print(df_titanic.groupby("Sex")["Age"].mean())
print(df_titanic.groupby("Sex")["Age"].median())

# Fill all null values with mean in terms of Sex
df_titanic.loc[(df_titanic["Age"].isnull()) & (df_titanic["Sex"]=="female"), "Age"] = df_titanic.groupby("Sex")["Age"].mean()["female"]
df_titanic.loc[(df_titanic["Age"].isnull()) & (df_titanic["Sex"]=="male"), "Age"] = df_titanic.groupby("Sex")["Age"].mean()["male"]

print(df_titanic.isnull().sum())